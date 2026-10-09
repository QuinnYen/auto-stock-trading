"""訊號掃描：量化各訊號的毛利，決定哪些值得建規則（依 docs/訊號掃描事前約定 v1.md，2026-10-09 確認）。

只掃描開發期 2007-01-01～2016-12-31 的訊號日；驗證期與最後測試期不碰。
33 個（訊號 × 持有期）檢定全部寫入 data/experiments/trials.csv（period ＝ scan-dev），報告輸出到
data/experiments/scan_dev_report.txt。判斷條件、進場資格、成本門檻都照事前約定，不在這裡調整。

用法：python src/backtest/signal_scan.py
驗證期（依 docs/訊號驗證事前約定 v2.md）：python src/backtest/signal_scan.py --period val --val-confirm
  只驗證開發期通過的 7 個組合，事件門檻 100、5 年中至少 3 年為正；報告 scan_val_report.txt，trials.csv 標記 scan-val。
"""
import argparse
import csv
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import holdout  # noqa: E402
import prices  # noqa: E402
import strong_stocks as ss  # noqa: E402

DEV = holdout.PERIODS["dev"]
LIQ_MIN = 1e8              # 近 20 日平均成交金額
PRICE_CAP = 600.0          # 原始收盤價上限
LOCK = 0.095               # 開盤 ≥ 前收 +9.5% 視為漲停買不到
FEE_RT = 2 * 0.001425 * 0.3 + 0.003   # 手續費 3 折（買賣各一次）＋賣出證交稅
BH_Q = 0.10
POSITION = 5000.0          # 白話報告用的每次投入金額
MIN_MONTH_EVENTS = 5       # 一個月至少幾個事件才納入月序列
MIN_YEAR_EVENTS = 10
MIN_POS_YEARS = 6
VAL_MIN_POS_YEARS = 3       # 驗證期只有 5 年（事前約定 v2 第 4 節）
VAL_MIN_EVENTS = 100
MIN_BENCH_STOCKS = 10      # 同日基準至少要有幾檔股票
MIN_RANK_POOL = 20         # 月度排名的母體少於這個數就不排名
ALL_H = (3, 5, 10, 20, 40, 60)


@dataclass(frozen=True)
class Spec:
    sid: str
    name: str            # 短名稱（報告用）
    horizons: tuple
    slip: float          # 單邊滑價
    elig: str            # std：標準資格；no_attn：不排除注意股；none：不排除注意與處置
    min_events: int = 200


SPECS = [
    Spec("H2a", "爆量3倍且上漲", (5, 10, 20, 40, 60), 0.0025, "std"),
    Spec("H2b", "爆量5倍且上漲", (5, 10, 20, 40, 60), 0.0025, "std"),
    Spec("H3", "12個月動能前10%", (20, 40, 60), 0.0025, "std"),
    Spec("H4a", "52週高價比前10%", (20, 40, 60), 0.0025, "std"),
    Spec("H4b", "創250日新高", (5, 10, 20, 40, 60), 0.0025, "std"),
    Spec("H5", "超跌處置股末段", (3, 5, 10), 0.005, "none", 80),
    Spec("H6a", "首次注意股(漲幅類)", (5, 10, 20), 0.0025, "no_attn"),
    Spec("H6b", "首次注意股(跌幅類)", (5, 10, 20), 0.0025, "no_attn"),
    Spec("H8", "5日大跌反轉", (3, 5, 10), 0.0025, "std"),
]
N_TESTS = sum(len(s.horizons) for s in SPECS)

# 事前約定 v2 第 3 節：開發期通過的 7 個組合；驗證期事件門檻降為 100
VAL_COMBOS = {"H4a": (40, 60), "H4b": (20, 40, 60), "H8": (3, 5)}
VAL_SPECS = [replace(s, horizons=VAL_COMBOS[s.sid], min_events=VAL_MIN_EVENTS) for s in SPECS if s.sid in VAL_COMBOS]
N_VAL_TESTS = sum(len(s.horizons) for s in VAL_SPECS)


def cost_rt(slip):
    return FEE_RT + 2 * slip


# ---------------------------------------------------------------- 資料與報酬
def load(con):
    ids = [r[0] for r in con.execute("SELECT DISTINCT stock_id FROM stock_price WHERE stock_id GLOB '[1-9][0-9][0-9][0-9]'")]
    adj = prices.load_adjusted(con, ids + ["0050"])
    cal = pd.DatetimeIndex(sorted(adj[(adj.stock_id == "0050") & (adj.date >= prices.CAL_START)].date.unique()))
    adj = adj[adj.date.isin(set(cal.strftime("%Y-%m-%d")))]
    cols = {"O": "o", "H": "h", "L": "l", "C": "c", "V": "volume", "M": "money", "RC": "close", "RO": "open"}
    return {k: prices.wide(adj, c, cal).drop(columns="0050") for k, c in cols.items()}, cal


def build_eligibility(M, RC, C, attn_recent, disp_known):
    """進場資格（訊號日當天的資料）：近 20 日平均成交金額 ≥ 1 億、原始收盤價 ≤ 600、有價格；
    std 再排除近期注意股與已公布處置股，no_attn 只排除處置股（H6 用），none 都不排除（H5 用）。"""
    base = ((M.rolling(20, min_periods=15).mean() >= LIQ_MIN) & (RC <= PRICE_CAP) & C.notna()).to_numpy()
    return base, {"std": base & ~attn_recent & ~disp_known, "no_attn": base & ~disp_known, "none": base}


def entry_ok_matrix(O, C):
    """隔天有開盤價，且沒有鎖在漲停（開盤 ≥ 前收 ×1.095 視為買不到）。"""
    ok = np.zeros(O.shape, dtype=bool)
    ok[:-1] = np.isfinite(O[1:]) & np.isfinite(C[:-1]) & (O[1:] < C[:-1] * (1 + LOCK))
    return ok


def forward_returns(O_df, cal, horizons, delist_days=30):
    """t 日訊號 → t+1 開盤進場、t+1+h 開盤出場的報酬（還原價）。
    出場日無價取其後 5 個交易日內第一個開盤價；已下市無法出場以 −100% 計。"""
    n = len(O_df)
    O = O_df.to_numpy(dtype="float64")
    entry = np.full(O.shape, np.nan)
    entry[:-1] = O[1:]
    Ob = O_df.bfill(limit=5)
    last_idx = np.array([np.flatnonzero(np.isfinite(O[:, j]))[-1] if np.isfinite(O[:, j]).any() else -1
                         for j in range(O.shape[1])])
    cutoff = cal[-1] - pd.Timedelta(days=delist_days)
    ended_stock = np.array([(li >= 0 and cal[li] < cutoff) for li in last_idx])
    idx = np.arange(n)[:, None]
    out = {}
    for h in horizons:
        exit_ = Ob.shift(-(1 + h)).to_numpy(dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = exit_ / entry - 1
        ended = ((idx + 1 + h) > last_idx[None, :]) & ended_stock[None, :]
        out[h] = np.where(np.isnan(ret) & np.isfinite(entry) & ended, -1.0, ret)
    return out


def benchmark(ret, mask):
    """同一進場日、符合標準資格的全部股票的平均報酬；股票數不足 MIN_BENCH_STOCKS 時為 NaN。"""
    valid = mask & np.isfinite(ret)
    cnt = valid.sum(axis=1)
    sm = np.where(valid, ret, 0.0).sum(axis=1)
    return np.where(cnt >= MIN_BENCH_STOCKS, sm / np.maximum(cnt, 1), np.nan)


# ---------------------------------------------------------------- 訊號
def sig_h2(V, C, k):
    avg = V.rolling(20, min_periods=15).mean().shift(1)
    return ((V >= k * avg) & (C > C.shift(1)) & (C / C.shift(1) - 1 < LOCK)).to_numpy()


def month_end_indices(cal):
    s = pd.Series(np.arange(len(cal)), index=cal)
    return s.groupby([cal.year, cal.month]).last().to_numpy()


def top_fraction_monthly(score, elig, cal, frac=0.10):
    """每個月最後一個交易日，在符合資格的股票中按分數排名，取前 frac。"""
    out = np.zeros(score.shape, dtype=bool)
    for t in month_end_indices(cal):
        row = np.where(elig[t], score[t], np.nan)
        ok = np.isfinite(row)
        if ok.sum() < MIN_RANK_POOL:
            continue
        rank = pd.Series(row).rank(pct=True).to_numpy()
        out[t] = ok & (rank >= 1 - frac)
    return out


def sig_h3_score(C):
    return (C.shift(21) / C.shift(252) - 1).to_numpy()


def sig_h4a_score(C, H):
    return (C / H.rolling(250, min_periods=200).max()).to_numpy()


def sig_h4b(C):
    return (C >= C.rolling(250, min_periods=200).max().shift(1)).to_numpy()


def sig_h8(C):
    return ((C / C.shift(5) - 1 <= -0.10) & (C / C.shift(1) - 1 > -LOCK)).to_numpy()


def sig_h6(rows, cal, col_index, shape, gap=30):
    """首次被列注意股：過去 gap 個交易日內沒有任何注意紀錄；依注意原因分漲幅類與跌幅類。"""
    up, down = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    days = defaultdict(lambda: defaultdict(list))
    for sid, dt, info in rows:
        j = col_index.get(sid)
        if j is None:
            continue
        i = int(cal.searchsorted(pd.Timestamp(dt)))
        if i < shape[0]:
            days[j][i].append(info)
    for j, by_day in days.items():
        idxs = sorted(by_day)
        for k, i in enumerate(idxs):
            if k > 0 and i - idxs[k - 1] <= gap:
                continue
            infos = by_day[i]
            if any("漲幅" in x for x in infos):
                up[i, j] = True
            if any("跌幅" in x for x in infos):
                down[i, j] = True
    return up, down


def sig_h5(rows, C, cal, col_index):
    """超跌處置股末段：處置期間倒數第 4 個交易日收盤為訊號日；公告日前 10 日累積報酬 ≤ −10%。"""
    sig = np.zeros(C.shape, dtype=bool)
    for sid, ann, start, end in rows:
        j = col_index.get(sid)
        if j is None:
            continue
        i_a = int(cal.searchsorted(pd.Timestamp(ann)))
        i_s = int(cal.searchsorted(pd.Timestamp(start)))
        i_e = int(cal.searchsorted(pd.Timestamp(end), side="right")) - 1
        if i_e - i_s + 1 < 4 or i_a < 10 or i_a >= len(cal):
            continue
        r10 = C[i_a, j] / C[i_a - 10, j] - 1
        if np.isfinite(r10) and r10 <= -0.10:
            sig[i_e - 3, j] = True
    return sig


# ---------------------------------------------------------------- 事件與統計
def extract_events(sig, elig, entry_ok, ret, bench, h, lo, hi):
    """訊號 ∧ 資格 ∧ 隔天買得到 ∧ 有報酬 ∧ 有同日基準；同一檔在觸發後 h 個交易日內再次觸發則忽略。"""
    valid = sig & elig & entry_ok & np.isfinite(ret) & np.isfinite(bench)[:, None]
    valid[:lo] = False
    valid[hi:] = False
    ts, js = [], []
    for j in np.flatnonzero(valid.any(axis=0)):
        last = -10 ** 9
        for t in np.flatnonzero(valid[:, j]):
            if t - last >= h + 1:
                ts.append(int(t))
                js.append(int(j))
                last = t
    ts, js = np.array(ts, dtype=int), np.array(js, dtype=int)
    r = ret[ts, js] if len(ts) else np.array([])
    return ts, js, r, (r - bench[ts]) if len(ts) else np.array([])


def month_groups(ts, cal):
    key = cal.year.to_numpy()[ts] * 12 + cal.month.to_numpy()[ts]
    _, inv = np.unique(key, return_inverse=True)
    return inv, np.bincount(inv)


def monthly_means(vals, inv, cnt):
    keep = cnt >= MIN_MONTH_EVENTS
    return (np.bincount(inv, weights=vals, minlength=len(cnt)) / np.maximum(cnt, 1))[keep]


def nw_t(x, lags):
    """對月序列的平均值做 Newey-West（Bartlett 權重）調整的 t 值。"""
    x = np.asarray(x, dtype=float)
    k = len(x)
    if k < 2:
        return float("nan")
    e = x - x.mean()
    v = (e * e).sum() / k
    for lag in range(1, min(lags, k - 1) + 1):
        v += 2 * (1 - lag / (lags + 1)) * (e[lag:] * e[:-lag]).sum() / k
    return float("nan") if v <= 0 else float(x.mean() / math.sqrt(v / k))


def p_greater(t):
    return 1.0 if not np.isfinite(t) else 0.5 * math.erfc(t / math.sqrt(2))


def bh_reject(pvals, q=BH_Q):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    out = np.zeros(m, dtype=bool)
    if passed.any():
        out[order[: np.flatnonzero(passed).max() + 1]] = True
    return out


def year_stats(ts, ex, cal):
    years = cal.year.to_numpy()[ts]
    per = {int(y): ((years == y).sum(), ex[years == y].mean(), ex[years == y].sum()) for y in np.unique(years)}
    pos = sum(1 for n, mu, _ in per.values() if n >= MIN_YEAR_EVENTS and mu > 0)
    best = max(per, key=lambda y: per[y][2])
    excl = float(ex[years != best].mean()) if (years != best).any() else float("nan")
    return per, pos, best, excl


def evaluate(spec, h, ts, r, ex, cal):
    n = len(ts)
    res = {"sid": spec.sid, "name": spec.name, "h": h, "n": n, "cost": cost_rt(spec.slip), "min_events": spec.min_events}
    if n == 0:
        res.update(months=0, mean_ret=np.nan, mean_ex=np.nan, bench_mean=np.nan, t=np.nan, p_nw=1.0, hit=np.nan,
                   win_net=np.nan, net=np.nan, pos_years=0, best_year=None, excl_best=np.nan, per_year={})
        return res
    inv, cnt = month_groups(ts, cal)
    months = monthly_means(ex, inv, cnt)
    t = nw_t(months, math.ceil(h / 21))
    per, pos, best, excl = year_stats(ts, ex, cal)
    res.update(months=len(months), mean_ret=float(r.mean()), mean_ex=float(ex.mean()),
               bench_mean=float((r - ex).mean()), t=t, p_nw=p_greater(t), hit=float((ex > 0).mean()),
               win_net=float((r - res["cost"] > 0).mean()), net=float(r.mean() - res["cost"]),
               pos_years=pos, best_year=best, excl_best=excl, per_year=per)
    return res


def verdict(res, rejected, min_pos_years=MIN_POS_YEARS):
    """依事前約定第 6 節（驗證期為 v2 第 4 節）；回傳 (是否通過, 白話原因)。"""
    reasons = []
    if res["n"] < res["min_events"]:
        reasons.append(f"機會太少（只有 {res['n']} 次）")
    if not (rejected and res["mean_ex"] > 0):
        reasons.append("沒有明顯贏過隨機買股票" if res["mean_ex"] > 0 else "表現不如隨機買股票")
    if not res["net"] > 0:
        reasons.append("扣掉成本後平均是虧的")
    if res["n"] and res["pos_years"] < min_pos_years:
        reasons.append(f"只有 {res['pos_years']} 個年度有贏過隨機")
    if res["n"] and not res["excl_best"] > 0:
        reasons.append("拿掉最好的一年就不賺了")
    return (not reasons), "；".join(reasons)


# ---------------------------------------------------------------- 隨機化檢定（取代 Newey-West 的 p 值）
N_NULL = 3000        # 用來估計『隨機情況下的分布』的次數
N_CHECK = 200        # 另外抽幾組隨機結果當作『假的真實結果』，檢查整個流程的誤判率


def prepare_test(ts, h, pool, cal):
    inv, cnt = month_groups(ts, cal) if len(ts) else (None, None)
    return {"ts": ts, "h": h, "inv": inv, "cnt": cnt, "pool": pool}


def stat_mean(vals, inv, cnt):
    """統計量：把事件按月份聚合後，取各月平均超額報酬的平均（事件少於 5 個的月份略過）。"""
    m = monthly_means(vals, inv, cnt)
    return float(m.mean()) if len(m) else float("nan")


def null_stats(item, rng, n):
    """把每個事件換成『同一天、隨機一檔符合標準資格的股票』，算 n 次統計量。"""
    ret, bench, start, rcnt, cols = item["pool"]
    ts = item["ts"]
    out = np.empty(n)
    for k in range(n):
        pick = start[ts] + (rng.random(len(ts)) * rcnt[ts]).astype(int)
        out[k] = stat_mean(ret[ts, cols[pick]] - bench[ts], item["inv"], item["cnt"])
    return out


def randomization_pvalues(items, observed, rng, n_null=N_NULL, n_check=N_CHECK):
    """單尾 p ＝ (1 ＋ 隨機結果 ≥ 真實結果 的次數) ÷ (n_null ＋ 1)。
    另外用 n_check 組額外的隨機結果當『假的真實結果』，回傳流程的誤判率（至少一個組合通過 BH 的比例、各檢定 p<0.05 的比例）。"""
    m = len(items)
    pvals, check = np.ones(m), np.ones((m, n_check))
    for i, item in enumerate(items):
        if len(item["ts"]) == 0 or not np.isfinite(observed[i]):
            continue
        draws = null_stats(item, rng, n_null + n_check)
        null, fake = draws[:n_null], draws[n_null:]
        pvals[i] = (1 + (null >= observed[i]).sum()) / (n_null + 1)
        check[i] = (1 + (null[None, :] >= fake[:, None]).sum(axis=1)) / (n_null + 1)
    any_rej = float(np.mean([bh_reject(check[:, k]).any() for k in range(n_check)]))
    return pvals, any_rej, (check < 0.05).mean(axis=1)


def make_pool(ret, std_elig, entry_ok, bench):
    mask = std_elig & entry_ok & np.isfinite(ret) & np.isfinite(bench)[:, None]
    cnt = mask.sum(axis=1)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    _, cols = np.nonzero(mask)
    return ret, bench, start, cnt, cols


# ---------------------------------------------------------------- 報告
def money(x):
    return f"{x:+,.0f} 元"


def build_report(rows, funnel, placebo_info, n_tests, *, title="訊號掃描報告（開發期 2007～2016；依事前約定 v1）", years=10,
                 dev_compare=None, extra_summary=(), dev_method_note=True):
    """dev_compare：{(sid, h): (事件數, 扣成本後平均)}，有值時白話表多一欄『開發期（對照）』；extra_summary 放在一句話結論最前面。"""
    passed = [r for r in rows if r["pass"]]
    lines = [f"# {title}", f"產生時間：{datetime.now():%Y-%m-%d %H:%M}", ""]
    lines += ["## 一句話結論", *extra_summary]
    if passed:
        names = "、".join(f"{r['sid']} {r['name']}（持有 {r['h']} 天）" for r in passed)
        lines.append(f"- {n_tests} 個組合中，**有 {len(passed)} 個**通過全部篩選條件：{names}。通過只代表『值得進一步建規則做帳戶模擬』，還不是可以交易。")
    else:
        lines.append(f"- {n_tests} 個組合中，**沒有任何一個**通過篩選：在扣掉手續費、證交稅和滑價之後，沒有哪個訊號能穩定贏過『同一天隨機買股票』。")
    best = max(rows, key=lambda r: r["net"] if np.isfinite(r["net"]) else -9)
    pos_net = [r for r in rows if r["net"] > 0]
    lines.append(f"- 扣成本後平均每次是賺的組合有 {len(pos_net)} 個（共 {n_tests} 個）；但要同時『明顯贏過隨機買股票』才算有本事，不然只是行情好。")
    lines.append(f"- 帳面最好的是 {best['sid']} {best['name']}（持有 {best['h']} 天）：每次投入 {POSITION:,.0f} 元，扣成本後平均 {money(best['net'] * POSITION)}"
                 f"（{best['net']:+.2%}），一年約 {best['n'] / years:.0f} 次機會；結論：{'通過' if best['pass'] else '放棄，' + best['reason']}。")
    lines += ["", "## 怎麼讀這份報告",
              f"- **每次投入 {POSITION:,.0f} 元，扣成本後平均賺／賠**：假設每次訊號出現就買 {POSITION:,.0f} 元，持有幾天後賣出，已經扣掉手續費（3 折）、賣出證交稅 0.3% 和滑價（單邊 0.25%，來回總成本約 0.89%）。",
              "- **隨機買股票平均賺／賠**：同一天、同樣持有幾天，隨機買一檔符合資格的股票的平均結果。訊號要比這一欄好，才代表訊號有本事，而不是剛好碰上大盤上漲。",
              f"- **贏過隨機**：把每個訊號事件換成『同一天隨機挑一檔符合資格的股票』，重複 {N_NULL:,} 次，看真實結果在這些隨機結果裡有多罕見（越罕見越可能是真的），並且考慮到我們一次測了 {n_tests} 個組合（測越多越容易碰巧好看）。",
              "- 結論標示『通過』才進入下一步（建規則、做 2 萬元帳戶模擬、再用沒看過的年份驗證）；標示『放棄』就不再調整參數重試。", "",
              "## 白話對照表",
              f"| 訊號 | 持有 | {years} 年共幾次（每年約） | 每次投入 {POSITION:,.0f} 元，扣成本後平均 | 扣成本後賺錢的比例 | 隨機買股票平均 |"
              + (" 開發期（對照）|" if dev_compare else "") + " 結論 |",
              "|---|---|---|---|---|---|" + ("---|" if dev_compare else "") + "---|"]
    for r in rows:
        dc = ""
        if dev_compare:
            dn, dnet = dev_compare[(r["sid"], r["h"])]
            dc = f" {dn:,} 次、{money(dnet * POSITION)}（{dnet:+.2%}） |"
        if r["n"] == 0:
            lines.append(f"| {r['sid']} {r['name']} | {r['h']} 天 | 0 | — | — | — |{dc} ❌ 放棄：沒有符合條件的機會 |")
            continue
        rand = (r["bench_mean"] - r["cost"]) * POSITION
        lines.append(f"| {r['sid']} {r['name']} | {r['h']} 天 | {r['n']:,}（{r['n'] / years:.0f}） | {money(r['net'] * POSITION)}（{r['net']:+.2%}） | "
                     f"{r['win_net']:.0%} | {money(rand)} |{dc} {'✅ 通過' if r['pass'] else '❌ 放棄：' + r['reason']} |")
    lines += ["", "---", "## 技術細節（事前約定第 8 節）",
              "| 檢定 | 事件數 | 月數 | 平均報酬 | 平均超額 | 超額>0 的比例 | NW t（僅供參考） | p(單尾，隨機化) | BH(q=0.10) | 成本門檻 | 正超額年數 | 去掉最佳年的平均超額 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["n"] == 0:
            lines.append(f"| {r['sid']} h={r['h']} | 0 | — | — | — | — | — | — | — | {r['cost']:.2%} | — | — |")
            continue
        lines.append(f"| {r['sid']} h={r['h']} | {r['n']} | {r['months']} | {r['mean_ret']:+.2%} | {r['mean_ex']:+.2%} | {r['hit']:.0%} | "
                     f"{r['t']:+.2f} | {r['p']:.4f} | {'通過' if r['bh'] else '未通過'} | {r['cost']:.2%} | {r['pos_years']} | {r['excl_best']:+.2%} |")
    lines += ["", "逐年平均超額報酬（事件數 ≥ 10 才列）："]
    for r in rows:
        if r["n"]:
            ys = "  ".join(f"{y}:{mu:+.1%}" for y, (n, mu, _) in sorted(r["per_year"].items()) if n >= MIN_YEAR_EVENTS)
            lines.append(f"- {r['sid']} h={r['h']}：{ys}")
    lines += ["", "進場資格漏斗（期間內『股票日』數量）："]
    lines += [f"- {k}：{v:,}" for k, v in funnel.items()]
    lines += ["", "統計流程健全性檢查（另抽 "
              f"{N_CHECK} 組隨機結果當『假的真實結果』，走完全部檢定與多重檢定流程）：",
              f"- 至少有 1 個組合通過 BH 的比例：{placebo_info[0]:.1%}（理論上應不超過約 10%）",
              f"- 各組合『p<0.05』的平均比例：{placebo_info[1].mean():.1%}（理論上約 5%）"]
    if dev_method_note:
        lines += ["- 方法說明：原本事前約定用 Newey-West t 檢定；第一次掃描的健全性檢查顯示誤判率偏高（至少一個通過 BH 的比例 25%，"
              "事件少的 H6a／H6b 單項誤判率 8%～12%），依約定在『看任何訊號結果之前』改為隨機化檢定。訊號、成本與篩選條件沒有改動。"]
    return "\n".join(lines)


def prepare(con, period=DEV):
    """載入資料並建立資格、報酬、基準、訊號（掃描與診斷共用）；period 為訊號日範圍（預設開發期）。"""
    print("載入資料…", flush=True)
    d, cal = load(con)
    cols = list(d["C"].columns)
    col_index = {c: i for i, c in enumerate(cols)}
    shape = d["C"].shape
    lo = int(cal.searchsorted(pd.Timestamp(period[0])))
    hi = int(cal.searchsorted(pd.Timestamp(period[1]), side="right"))
    O, C = d["O"].to_numpy(dtype="float64"), d["C"].to_numpy(dtype="float64")
    print(f"股票 {shape[1]} 檔；訊號日索引 {lo}～{hi - 1}（{cal[lo].date()}～{cal[hi - 1].date()}）", flush=True)

    _, attn_recent, disp_known, _ = ss.alert_matrices(con, cols, cal, 5)
    base, elig = build_eligibility(d["M"], d["RC"], d["C"], attn_recent, disp_known)
    entry_ok = entry_ok_matrix(O, C)
    rets = forward_returns(d["O"], cal, ALL_H)
    bench = {h: benchmark(rets[h], elig["std"] & entry_ok) for h in ALL_H}
    pools = {h: make_pool(rets[h], elig["std"], entry_ok, bench[h]) for h in ALL_H}

    attn_rows = con.execute("SELECT stock_id, date, info FROM attention").fetchall()
    disp_rows = con.execute("SELECT stock_id, announce_date, period_start, period_end FROM disposition").fetchall()
    h6_up, h6_down = sig_h6(attn_rows, cal, col_index, shape)
    sigs = {
        "H2a": sig_h2(d["V"], d["C"], 3), "H2b": sig_h2(d["V"], d["C"], 5),
        "H3": top_fraction_monthly(sig_h3_score(d["C"]), elig["std"], cal),
        "H4a": top_fraction_monthly(sig_h4a_score(d["C"], d["H"]), elig["std"], cal),
        "H4b": sig_h4b(d["C"]), "H5": sig_h5(disp_rows, C, cal, col_index),
        "H6a": h6_up, "H6b": h6_down, "H8": sig_h8(d["C"]),
    }

    return {"cal": cal, "shape": shape, "lo": lo, "hi": hi, "C": C, "elig": elig, "base": base, "entry_ok": entry_ok,
            "rets": rets, "bench": bench, "pools": pools, "sigs": sigs,
            "cols": cols, "RC": d["RC"].to_numpy(dtype="float64"), "d": d}


def collect_events(ctx, specs=SPECS):
    out = []
    for spec in specs:
        for h in spec.horizons:
            ts, js, r, ex = extract_events(ctx["sigs"][spec.sid], ctx["elig"][spec.elig], ctx["entry_ok"], ctx["rets"][h],
                                           ctx["bench"][h], h, ctx["lo"], ctx["hi"])
            out.append((spec, h, ts, js, r, ex))
    return out


def val_conclusions(rows):
    """事前約定 v2 第 5 節：H4 族 5 個組合至少 2 個通過才算 52 週新高優勢在驗證期仍成立；H8 兩個都通過才算成立。"""
    ok = {(r["sid"], r["h"]): r["pass"] for r in rows}
    n_h4 = sum(v for (sid, _), v in ok.items() if sid in ("H4a", "H4b"))
    h4 = n_h4 >= 2
    h8 = ok[("H8", 3)] and ok[("H8", 5)]
    return h4, h8, [
        f"- 7 個組合中 **{sum(ok.values())} 個**通過驗證期篩選（事前約定 v2 第 4 節）。",
        f"- 52 週新高這一族（H4a、H4b 共 5 個組合）通過 {n_h4} 個：**{'優勢在驗證期仍然成立' if h4 else '優勢未能在驗證期重現，放棄此族，不換參數重試'}**（需至少 2 個）。",
        f"- 短期反轉（H8 兩個組合）：**{'成立' if h8 else '不成立'}**（需兩個都通過）。"]


def load_dev_compare(path=None):
    """從 trials.csv 讀開發期（scan-dev）每個組合的事件數與扣成本後平均，供驗證期報告並列對照。"""
    out = {}
    with (path or ss.OUT / "trials.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["period"] == "scan-dev":
                out[(row["version"], int(row["exit"][2:]))] = (int(row["trades"]), float(row["avg_ret"]) if row["avg_ret"] else float("nan"))
    return out


def apply_verdicts(results, pvals, min_pos_years):
    """對全部檢定做 BH，再逐一套用通過條件，結果寫回 results。"""
    for r, p_, rj in zip(results, pvals, bh_reject(pvals)):
        r["p"], r["bh"] = float(p_), bool(rj)
        r["pass"], r["reason"] = verdict(r, bool(rj), min_pos_years)


def scan_config(name):
    """各期間的掃描設定：(檢定清單, 檢定數, 訊號日期間, 正超額年數門檻)。"""
    if name == "val":
        return VAL_SPECS, N_VAL_TESTS, holdout.PERIODS["val"], VAL_MIN_POS_YEARS
    return SPECS, N_TESTS, DEV, MIN_POS_YEARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "val"], default="dev")
    ap.add_argument("--val-confirm", action="store_true")
    args = ap.parse_args()
    holdout.require_access(args.period, "signal_scan", val_confirm=args.val_confirm)
    val = args.period == "val"
    specs, n_tests, period, min_pos_years = scan_config(args.period)
    con = sqlite3.connect(prices.DB)
    ctx = prepare(con, period)
    cal, shape, lo, hi, C = ctx["cal"], ctx["shape"], ctx["lo"], ctx["hi"], ctx["C"]
    elig, base, entry_ok, rets, bench, pools = (ctx[k] for k in ("elig", "base", "entry_ok", "rets", "bench", "pools"))
    assert period[0] <= str(cal[lo].date()) and str(cal[hi - 1].date()) <= period[1], "訊號日範圍超出指定期間"
    results, items, observed = [], [], []
    for spec, h, ts, js, r, ex in collect_events(ctx, specs):
        res = evaluate(spec, h, ts, r, ex, cal)
        item = prepare_test(ts, h, pools[h], cal)
        results.append(res)
        items.append(item)
        observed.append(stat_mean(ex, item["inv"], item["cnt"]) if len(ts) else float("nan"))
        print(f"  {spec.sid} h={h}：{len(ts)} 個事件", flush=True)
    assert len(results) == n_tests == (7 if val else 33)

    print("隨機化檢定…", flush=True)
    pvals, any_rej, raw_rate = randomization_pvalues(items, observed, np.random.default_rng(0))
    apply_verdicts(results, pvals, min_pos_years)
    placebo_info = (any_rej, raw_rate)

    window = np.zeros(shape, dtype=bool)
    window[lo:hi] = True
    funnel = {"有價格": int((window & np.isfinite(C)).sum()), "且流動性 ≥ 1 億、原始價 ≤ 600 元": int((window & base).sum()),
              "且非近期注意股、非處置股（標準資格）": int((window & elig["std"]).sum()),
              "且隔天買得到（標準資格）": int((window & elig["std"] & entry_ok).sum())}
    if val:
        _, _, summary = val_conclusions(results)
        text = build_report(results, funnel, placebo_info, n_tests, title="訊號驗證報告（驗證期 2017～2021；依事前約定 v2）", years=5,
                            dev_compare=load_dev_compare(), extra_summary=summary, dev_method_note=False)
    else:
        text = build_report(results, funnel, placebo_info, n_tests)
    ss.OUT.mkdir(parents=True, exist_ok=True)
    (ss.OUT / f"scan_{args.period}_report.txt").write_text(text, encoding="utf-8")
    for r in results:
        ss.log_trial({"time": datetime.now().isoformat(timespec="seconds"), "period": f"scan-{args.period}", "version": r["sid"],
                      "exit": f"h={r['h']}", "slip": (r["cost"] - FEE_RT) / 2, "sig": r["name"], "cfg": "",
                      "cagr": "", "mdd": "", "triggers": "", "trades": r["n"], "win": r["win_net"],
                      "avg_net": r["net"] * POSITION if r["n"] else "", "avg_ret": r["net"]})
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
