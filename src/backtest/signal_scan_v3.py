"""訊號掃描 v3：月營收（H1）、低券資比（H9）、股利（H10）事件（依 docs/訊號掃描事前約定 v3.md，2026-10-09 確認）。

沿用 signal_scan.py 的進場資格、報酬、同日基準、隨機化檢定、BH 與通過條件；只掃開發期 2007-01-01～2016-12-31。
18 個（訊號 × 持有期）檢定全部寫入 data/experiments/trials.csv（period ＝ scan3-dev），報告輸出到 scan3_dev_report.txt。
本程式沒有驗證期／最後測試期的選項。

約定沒有明寫、實作時採取的保守做法：
  - 營收年增率需要的任一個月份缺資料（或去年同月營收 ≤ 0）時，該窗口不產生訊號；年增率先四捨五入到 10 位再與門檻比較。
  - 同一檔股票同一公告日的多列股利加總視為一次公告。
  - 公告日／營收公開日早於交易日曆起點者略過。

用法：python src/backtest/signal_scan_v3.py
"""
import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import holdout  # noqa: E402
import prices  # noqa: E402
import signal_scan as s1  # noqa: E402
import strong_stocks as ss  # noqa: E402

H1A_YOY, H1B_YOY, H1C_YOY = 0.20, 0.10, 0.10
H1C_MONTHS = 6
H9_FRAC = 0.10
H10A_YIELD = 0.05
H10B_RATIO = 1.2
HORIZONS = (20, 40, 60)

SPECS3 = [
    s1.Spec("H1a", "營收年增創高", HORIZONS, 0.0025, "std"),
    s1.Spec("H1b", "營收由負轉正", HORIZONS, 0.0025, "std"),
    s1.Spec("H1c", "營收連6月成長", HORIZONS, 0.0025, "std"),
    s1.Spec("H9a", "低券資比前10%", HORIZONS, 0.0025, "std"),
    s1.Spec("H10a", "高殖利率公告", HORIZONS, 0.0025, "std"),
    s1.Spec("H10b", "股利大增公告", HORIZONS, 0.0025, "std"),
]
N_TESTS3 = sum(len(s.horizons) for s in SPECS3)


def scan_config():
    """(檢定清單, 檢定數, 訊號日期間, 正超額年數門檻)；只有開發期。"""
    return SPECS3, N_TESTS3, s1.DEV, s1.MIN_POS_YEARS


# ---------------------------------------------------------------- H1 營收
def release_index(year, month, cal):
    """M 月營收視為次月 10 日收盤時已知；回傳訊號日（次月 10 日當天或之後第一個交易日）在 cal 的位置。
    超出日曆結尾或早於日曆起點時回傳 len(cal)（沒有有效訊號日）。"""
    ry, rm = (year, month + 1) if month < 12 else (year + 1, 1)
    d = pd.Timestamp(ry, rm, 10)
    if d < cal[0]:
        return len(cal)
    return int(cal.searchsorted(d))


def revenue_signals(rows, cal, col_index, shape):
    """rows：(stock_id, revenue_year, revenue_month, revenue)。回傳 {H1a, H1b, H1c} 布林矩陣（訊號日 × 股票）。"""
    out = {k: np.zeros(shape, dtype=bool) for k in ("H1a", "H1b", "H1c")}
    by_stock = {}
    for sid, y, m, rev in rows:
        if sid in col_index:
            by_stock.setdefault(sid, {})[y * 12 + m - 1] = rev
    for sid, rev in by_stock.items():
        j = col_index[sid]
        yoy = {mi: round(v / rev[mi - 12] - 1, 10) for mi, v in rev.items() if rev.get(mi - 12, 0) > 0}
        for mi, g in yoy.items():
            win12 = [yoy.get(mi - k) for k in range(12)]
            win6 = [yoy.get(mi - k) for k in range(H1C_MONTHS)]
            prev = yoy.get(mi - 1)
            a = g >= H1A_YOY and None not in win12 and g >= max(win12)
            b = prev is not None and prev < 0 and g >= H1B_YOY
            c = None not in win6 and min(win6) >= H1C_YOY
            if not (a or b or c):
                continue
            i = release_index(mi // 12, mi % 12 + 1, cal)
            if i < shape[0]:
                out["H1a"][i, j] |= a
                out["H1b"][i, j] |= b
                out["H1c"][i, j] |= c
    return out


# ---------------------------------------------------------------- H9 券資比
def margin_ratio_matrix(rows, cal, col_index, shape):
    """rows：(stock_id, date, 融資餘額, 融券餘額)。券資比 ＝ 融券 ÷ 融資；融資為 0、日期不在日曆、代號不在欄位者為 NaN。"""
    ratio = np.full(shape, np.nan)
    day_idx = {d: i for i, d in enumerate(cal.strftime("%Y-%m-%d"))}
    for sid, day, margin, short in rows:
        j, i = col_index.get(sid), day_idx.get(day)
        if j is not None and i is not None and margin and margin > 0:
            ratio[i, j] = short / margin
    return ratio


def sig_h9a(ratio, elig, cal):
    """每月最後一個交易日，在符合資格且有券資比的股票中，取券資比 ≤ 第 10 百分位值者（並列全部納入）。"""
    out = np.zeros(ratio.shape, dtype=bool)
    for t in s1.month_end_indices(cal):
        row = np.where(elig[t], ratio[t], np.nan)
        ok = np.isfinite(row)
        if ok.sum() < s1.MIN_RANK_POOL:
            continue
        out[t] = ok & (row <= np.quantile(row[ok], H9_FRAC))
    return out


# ---------------------------------------------------------------- H10 股利
def dividend_signals(rows, RC, cal, col_index, shape):
    """rows：(stock_id, 公告日字串, 現金盈餘配發, 現金公積配發)。同股同公告日加總為一次公告。
    訊號日 ＝ 公告日當天或之後第一個交易日；H10a 殖利率用訊號日原始收盤；H10b 比較同股上一次公告。"""
    a_out, b_out = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    by_stock = {}
    for sid, ann, c_e, c_s in rows:
        if sid in col_index:
            d = by_stock.setdefault(sid, {})
            d[ann] = d.get(ann, 0.0) + (c_e or 0.0) + (c_s or 0.0)
    for sid, anns in by_stock.items():
        j, prev = col_index[sid], None
        for ann in sorted(anns):
            cash = anns[ann]
            d = pd.Timestamp(ann)
            i = int(cal.searchsorted(d)) if d >= cal[0] else len(cal)
            if i < shape[0]:
                close = RC[i, j]
                if cash > 0 and np.isfinite(close) and close > 0 and round(cash / close, 10) >= H10A_YIELD:
                    a_out[i, j] = True
                if prev is not None and prev > 0 and round(cash / prev, 10) >= H10B_RATIO:
                    b_out[i, j] = True
            prev = cash
    return a_out, b_out


# ---------------------------------------------------------------- 資料與主程式
def prepare3(con, period=s1.DEV):
    ctx = s1.prepare(con, period)
    cal, shape = ctx["cal"], ctx["shape"]
    col_index = {c: i for i, c in enumerate(ctx["cols"])}
    print("建立 H1、H9、H10 訊號…", flush=True)
    sigs = revenue_signals(con.execute("SELECT stock_id, revenue_year, revenue_month, revenue FROM month_revenue").fetchall(),
                           cal, col_index, shape)
    ends = [i for i in s1.month_end_indices(cal) if ctx["lo"] <= i < ctx["hi"]]
    days = [str(cal[i].date()) for i in ends]
    marks = ",".join("?" * len(days))
    mrows = con.execute(f"SELECT stock_id, date, margin_today_balance, short_today_balance FROM margin WHERE date IN ({marks})", days).fetchall()
    sigs["H9a"] = sig_h9a(margin_ratio_matrix(mrows, cal, col_index, shape), ctx["elig"]["std"], cal)
    drows = con.execute("SELECT stock_id, announcement_date, cash_earnings_distribution, cash_statutory_surplus FROM dividend_policy").fetchall()
    sigs["H10a"], sigs["H10b"] = dividend_signals(drows, ctx["RC"], cal, col_index, shape)
    ctx["sigs"] = {**ctx["sigs"], **sigs}
    return ctx


NOTES = ["", "---", "## 實作細節說明（約定未明寫處的保守做法）",
         "- 營收年增率需要的任一個月份缺資料（或去年同月營收 ≤ 0）時，該窗口不產生 H1 訊號；年增率先四捨五入到 10 位再與門檻比較。",
         "- H9a 的母體需 ≥ 20 檔符合資格且有券資比的股票（沿用 v1 的排名母體下限），並列於第 10 百分位值者全部納入。",
         "- H10 同一檔股票同一公告日的多列股利加總視為一次公告；公告日早於交易日曆起點者略過。"]


def main():
    argparse.ArgumentParser().parse_args()
    holdout.require_access("dev", "signal_scan_v3")
    specs, n_tests, period, min_pos_years = scan_config()
    con = sqlite3.connect(prices.DB)
    ctx = prepare3(con, period)
    cal, shape, lo, hi, C = ctx["cal"], ctx["shape"], ctx["lo"], ctx["hi"], ctx["C"]
    elig, base, entry_ok, pools = (ctx[k] for k in ("elig", "base", "entry_ok", "pools"))
    assert period[0] <= str(cal[lo].date()) and str(cal[hi - 1].date()) <= period[1], "訊號日範圍超出開發期"
    results, items, observed = [], [], []
    for spec, h, ts, js, r, ex in s1.collect_events(ctx, specs):
        res = s1.evaluate(spec, h, ts, r, ex, cal)
        item = s1.prepare_test(ts, h, pools[h], cal)
        results.append(res)
        items.append(item)
        observed.append(s1.stat_mean(ex, item["inv"], item["cnt"]) if len(ts) else float("nan"))
        print(f"  {spec.sid} h={h}：{len(ts)} 個事件", flush=True)
    assert len(results) == n_tests == 18

    print("隨機化檢定…", flush=True)
    pvals, any_rej, raw_rate = s1.randomization_pvalues(items, observed, np.random.default_rng(0))
    s1.apply_verdicts(results, pvals, min_pos_years)

    window = np.zeros(shape, dtype=bool)
    window[lo:hi] = True
    funnel = {"有價格": int((window & np.isfinite(C)).sum()), "且流動性 ≥ 1 億、原始價 ≤ 600 元": int((window & base).sum()),
              "且非近期注意股、非處置股（標準資格）": int((window & elig["std"]).sum()),
              "且隔天買得到（標準資格）": int((window & elig["std"] & entry_ok).sum())}
    text = s1.build_report(results, funnel, (any_rej, raw_rate), n_tests,
                           title="訊號掃描報告 v3：營收、融資券、股利（開發期 2007～2016；依事前約定 v3）", years=10, dev_method_note=False)
    text = "\n".join([text, *NOTES])
    ss.OUT.mkdir(parents=True, exist_ok=True)
    (ss.OUT / "scan3_dev_report.txt").write_text(text, encoding="utf-8")
    for r in results:
        ss.log_trial({"time": datetime.now().isoformat(timespec="seconds"), "period": "scan3-dev", "version": r["sid"],
                      "exit": f"h={r['h']}", "slip": (r["cost"] - s1.FEE_RT) / 2, "sig": r["name"], "cfg": "",
                      "cagr": "", "mdd": "", "triggers": "", "trades": r["n"], "win": r["win_net"],
                      "avg_net": r["net"] * s1.POSITION if r["n"] else "", "avg_ret": r["net"]})
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
