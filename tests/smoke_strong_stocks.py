"""強勢股回測引擎的冒煙測試：證明引擎沒有算錯（不是證明策略賺錢）。

報告：data/experiments/smoke_report.txt
內容：檢查結果總表 → 資料覆蓋 → 訊號漏斗 → 正確性檢查（前視偏誤、警示旗標、帳戶不變量、獨立重算交易）
      → 交易明細抽樣 → 績效摘要 → 敏感度測試（逐一拿掉強勢條件、參數一次動一個）。
用法：python tests/smoke_strong_stocks.py
使用開發期（2007～2016）與目前已下載的部分股票；所有績效數字僅供除錯，不是結論。
試驗紀錄寫入 trials.csv，period 標記為 smoke（不計入正式試驗次數）。
"""
import random
import sqlite3
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backtest"))
import account  # noqa: E402
import strong_stocks as ss  # noqa: E402

REPORT = ss.OUT / "smoke_report.txt"
PERIOD = "dev"
SENS_SLIP = 0.0025
VERSIONS = ("breakout", "pullback")
checks: list[tuple[str, bool, str]] = []
sections: list[tuple[str, list[str]]] = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


def emit(title, lines):
    sections.append((title, lines))
    write_report(done=False)
    print(f"完成：{title}", flush=True)


def write_report(done):
    head = ["# 強勢股回測引擎冒煙測試報告",
            f"產生時間：{datetime.now():%Y-%m-%d %H:%M}（{'完成' if done else '進行中'}）",
            "資料：目前已下載的部分上市股票（不是全市場）；期間：開發期 2007～2016。績效數字僅供除錯，不是結論。", ""]
    ok = sum(1 for c in checks if c[1])
    head += [f"## 檢查結果總表：{ok}／{len(checks)} 通過"]
    head += [f"- [{'通過' if c[1] else '失敗'}] {c[0]}" + (f"：{c[2]}" if c[2] else "") for c in checks]
    body = []
    for title, lines in sections:
        body += ["", f"## {title}", *lines]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(head + body), encoding="utf-8")


# ---------------------------------------------------------------- 獨立重算（不經過引擎的矩陣程式）
def stock_series(con, sid, cal):
    df = pd.read_sql("SELECT date,open,high,low,close,volume,money FROM stock_price "
                     "WHERE stock_id=? AND close>0 AND open>0 ORDER BY date", con, params=(sid,))
    ev = con.execute("""SELECT date,k FROM (
        SELECT date, reference_price/before_price k FROM dividend_result WHERE stock_id=? AND before_price>0 AND reference_price>0
        UNION ALL SELECT date, post_reduction_ref_price/last_close FROM capital_reduction WHERE stock_id=? AND last_close>0 AND post_reduction_ref_price>0
        UNION ALL SELECT date, after_price/before_price FROM split_price WHERE stock_id=? AND before_price>0 AND after_price>0
        UNION ALL SELECT p.date, p.after_ref_close/p.before_close FROM par_value_change p WHERE p.stock_id=? AND p.before_close>0
          AND p.after_ref_close>0 AND NOT EXISTS (SELECT 1 FROM split_price s WHERE s.stock_id=p.stock_id AND s.date=p.date))""",
                      (sid,) * 4).fetchall()
    df["f"] = [float(np.prod([k for ed, k in ev if ed > dt])) if ev else 1.0 for dt in df.date]
    out = pd.DataFrame({"ro": df.open, "rh": df.high, "rl": df.low, "rc": df.close, "o": df.open * df.f,
                        "h": df.high * df.f, "l": df.low * df.f, "c": df.close * df.f, "v": df.volume, "m": df.money})
    out.index = pd.DatetimeIndex(df.date)
    return out.reindex(cal)


def roll(s, n, fn):
    return getattr(s.rolling(n, min_periods=int(n * 0.8)), fn)()


def verify_trade(con, t, ver, sig, cfg, cols, cal, mkt_s):
    """回傳失敗訊息列表（空 ＝ 全部通過）。無法驗證的項目以 note 回報。"""
    sid, e, x, T = cols[t["stock"]], t["entry_idx"], t["exit_idx"], t["entry_idx"] - 1
    s = stock_series(con, sid, cal)
    if s.c.iloc[max(T - 260, 0):T + 1].isna().any():
        return None, ["訊號前 260 日有缺漏日，略過獨立驗證"]
    fails = []
    close = lambda a, b, rtol=1e-4: bool(np.isclose(a, b, rtol=rtol))  # noqa: E731
    if not close(t["entry_raw"], s.ro.iloc[e] * (1 + cfg.slip), 1e-5):
        fails.append(f"進場原始價 {t['entry_raw']:.4f} ≠ 資料庫開盤 {s.ro.iloc[e]:.4f}")
    if not close(t["entry_adj"], s.o.iloc[e] * (1 + cfg.slip)):
        fails.append("進場還原價與獨立重算不同")
    # --- 訊號日 T 的條件
    c, h, l_, o, v, m = s.c, s.h, s.l, s.o, s.v, s.m
    ma10, ma20, ma60 = roll(c, 10, "mean"), roll(c, 20, "mean"), roll(c, 60, "mean")
    if ver == "breakout":
        if not c.iloc[T] >= c.iloc[T - sig.breakout_n + 1:T + 1].max():
            fails.append("T 日收盤不是 N 日新高")
        if not v.iloc[T] >= sig.vol_k * v.iloc[T - 5:T].mean():
            fails.append("T 日成交量未達前 5 日均量 k 倍")
    else:
        pm = roll(c, sig.pullback_ma, "mean")
        if not (l_.iloc[T] <= pm.iloc[T] * (1 + sig.pullback_tol) and c.iloc[T] >= pm.iloc[T] and c.iloc[T] > o.iloc[T]):
            fails.append("T 日不符合回檔條件")
    if sig.ma_trend and not (c.iloc[T] > ma20.iloc[T] > ma60.iloc[T] and ma20.iloc[T] > ma20.iloc[T - 5]
                             and ma60.iloc[T] > ma60.iloc[T - 5]):
        fails.append("T 日不符合均線多頭")
    if sig.near_high > 0 and not c.iloc[T] / h.iloc[T - 249:T + 1].max() >= sig.near_high:
        fails.append("T 日距 52 週高超過門檻")
    if sig.beat_market:
        mret = mkt_s.iloc[T] / mkt_s.iloc[T - sig.rs_n] - 1
        if not c.iloc[T] / c.iloc[T - sig.rs_n] - 1 > mret:
            fails.append("T 日漲幅未勝過大盤")
    if sig.market_filter and not mkt_s.iloc[T] > roll(mkt_s, 60, "mean").iloc[T]:
        fails.append("T 日大盤低於 60 日均線")
    if not m.iloc[T - 19:T + 1].mean() >= sig.liq_min:
        fails.append("T 日流動性不足")
    if not c.iloc[T] / ma20.iloc[T] - 1 <= sig.bias_max:
        fails.append("T 日乖離過大")
    daily = c.pct_change(fill_method=None)
    if daily.iloc[T - 2:T + 1].max() >= 0.095 or c.iloc[T] / c.iloc[T - 3] - 1 >= 0.15:
        fails.append("T 日屬追高")
    dates = [cal[i].strftime("%Y-%m-%d") for i in range(T - sig.attn_days + 1, T + 1)]
    n_att = con.execute(f"SELECT COUNT(*) FROM attention WHERE stock_id=? AND date IN ({','.join('?' * len(dates))})",
                        (sid, *dates)).fetchone()[0]
    if n_att:
        fails.append("進場前 N 日內有注意股公告")
    n_disp = con.execute("SELECT COUNT(*) FROM disposition WHERE stock_id=? AND announce_date<=? AND period_end>=?",
                         (sid, cal[T].strftime("%Y-%m-%d"), cal[T + 1].strftime("%Y-%m-%d"))).fetchone()[0]
    if n_disp:
        fails.append("進場時已有處置股公告")
    # --- 出場
    stop = t["entry_adj"] * (1 - cfg.stop_pct)
    r = t["reason"]
    early = (l_.iloc[e:x] <= stop).any() if x > e else False
    if r == "停損":
        if not l_.iloc[x] <= stop:
            fails.append("停損日最低價未碰到停損價")
        if not close(t["exit_adj"], min(o.iloc[x], stop)):
            fails.append("停損成交價應為 min(開盤, 停損價)")
        if (l_.iloc[e:x] <= stop).any():
            fails.append("更早的日子已碰到停損價")
    else:
        if early:
            fails.append("持有期間已碰到停損價卻未停損")
        if r in ("均線出場", "時間停損", "注意股", "處置股", "停機") and not close(t["exit_adj"], o.iloc[x]):
            fails.append("開盤出場價與獨立重算不同")
        if r == "均線出場":
            mx = roll(c, cfg.ma_exit, "mean")
            first = next((i for i in range(e, x) if c.iloc[i] < mx.iloc[i]), None)
            if first is None or first != x - 1:
                fails.append(f"均線出場日不符（首次跌破日 {first}，出場日 {x}）")
        elif r == "時間停損":
            if x - e != cfg.time_stop:
                fails.append(f"持有 {x - e} 日，應為 {cfg.time_stop} 日")
        elif r == "注意股":
            n = con.execute("SELECT COUNT(*) FROM attention WHERE stock_id=? AND date=?",
                            (sid, cal[x - 1].strftime("%Y-%m-%d"))).fetchone()[0]
            if not n:
                fails.append("出場前一日沒有注意股公告")
        elif r == "處置股":
            n = con.execute("SELECT COUNT(*) FROM disposition WHERE stock_id=? AND announce_date<=? AND period_end>=?",
                            (sid, cal[x - 1].strftime("%Y-%m-%d"), cal[x].strftime("%Y-%m-%d"))).fetchone()[0]
            if not n:
                fails.append("出場前一日沒有處置公告")
        elif r == "停利":
            if not (h.iloc[x] >= t["entry_adj"] * (1 + cfg.tp_pct) and close(t["exit_adj"], max(o.iloc[x], t["entry_adj"] * (1 + cfg.tp_pct)))):
                fails.append("停利條件或價格不符")
    # --- 損益恆等式
    amount = t["shares"] * t["exit_raw"]
    if not np.isclose(t["net"], amount - t["fee_sell"] - t["tax"] - t["basis"], atol=1e-6):
        fails.append("淨損益恆等式不成立")
    if r != "下市全損" and not close(t["exit_raw"], t["exit_adj"] * t["scale"] * (1 - cfg.slip), 1e-6):
        fails.append("出場原始價 ≠ 還原價 × scale × (1−滑價)")
    return fails, []


def invariants(res, cfg, label, last_close):
    t = res.trades
    rate = account.FEE_BASE * cfg.discount
    check(f"{label}｜現金不為負", np.nanmin(res.cash) >= -1e-6, f"最小現金 {np.nanmin(res.cash):.2f}")
    check(f"{label}｜持股不超過 {cfg.max_pos} 檔", res.n_held.max() <= cfg.max_pos, f"最大 {res.n_held.max()}")
    check(f"{label}｜單筆風險 ≤ {cfg.risk_pct:.0%} 總資金",
          all(x["shares"] * cfg.stop_pct * x["entry_raw"] <= cfg.risk_pct * x["eq_at_entry"] * (1 + 1e-9) for x in t),
          f"{len(t)} 筆")
    check(f"{label}｜單檔部位 ≤ 總資金 1/3",
          all(x["shares"] * x["entry_raw"] <= cfg.pos_cap * x["eq_at_entry"] * (1 + 1e-9) for x in t))
    check(f"{label}｜買進手續費符合公式",
          all(abs(x["fee_buy"] - max(x["shares"] * x["entry_raw"] * rate, account.MIN_FEE)) < 1e-6 for x in t))
    check(f"{label}｜賣出手續費與證交稅符合公式",
          all(abs(x["fee_sell"] - (max(x["shares"] * x["exit_raw"] * rate, account.MIN_FEE) if x["exit_raw"] > 0 else 0)) < 1e-6
              and abs(x["tax"] - x["shares"] * x["exit_raw"] * account.TAX) < 1e-6 for x in t))
    cash_expected = res.capital + sum(x["net"] for x in t) - sum(p["basis"] for p in res.open_positions)
    check(f"{label}｜現金對帳（本金＋已平倉淨損益−未平倉成本）", abs(res.cash[-1] - cash_expected) < 1e-4,
          f"差 {res.cash[-1] - cash_expected:.6f}")
    eq_expected = res.cash[-1] + sum(p["shares"] * p["scale"] * last_close[p["stock"]] for p in res.open_positions)
    check(f"{label}｜期末淨值＝現金＋持股市值", abs(res.equity[-1] - eq_expected) < 1e-4)


# ---------------------------------------------------------------- 主流程
def main():
    con = sqlite3.connect(ss.prices.DB)
    print("載入資料…", flush=True)
    d, mkt, cal = ss.load_data(con)
    cols = list(d["C"].columns)
    start, end = (int(cal.searchsorted(pd.Timestamp(ss.PERIODS[PERIOD][0]))),
                  int(cal.searchsorted(pd.Timestamp(ss.PERIODS[PERIOD][1]), side="right")))
    sig0, cfg0 = ss.Sig(), account.Cfg()
    alert_cache: dict[int, tuple] = {}

    def alerts_for(days):
        if days not in alert_cache:
            alert_cache[days] = ss.alert_matrices(con, cols, cal, days)
        return alert_cache[days]

    def build(ver, sig, cfg):
        a = alerts_for(sig.attn_days)
        return ss.build(d, mkt, cal, a[1], a[2], sig, ver, cfg.ma_exit), a

    # ---- A. 資料覆蓋
    yrs = d["C"].index.year
    cover = d["C"].notna().groupby(yrs).any().sum(axis=1)
    codes = sorted(int(c) for c in cols)
    attn0, _, disp_known0, disp_in0 = alerts_for(sig0.attn_days)
    emit("A. 資料覆蓋", [
        f"股票 {len(cols)} 檔（代號 {codes[0]}～{codes[-1]}）；日曆 {cal[0].date()}～{cal[-1].date()}（{len(cal)} 個交易日）",
        "各年有價格的股票數：" + "  ".join(f"{y}:{n}" for y, n in cover.items() if 2005 <= y <= 2016),
        f"注意股標記 {int(attn0.sum())} 個股票日；處置已公布（含期間內）標記 {int(disp_known0.sum())} 個股票日；"
        f"處置期間內 {int(disp_in0.sum())} 個股票日",
        "提醒：下載尚未完成，股票池只涵蓋較小的代號，不是全市場。"])

    # ---- B. 訊號漏斗
    funnel_lines, built_cache = [], {}
    for ver in VERSIONS:
        b, a = build(ver, sig0, cfg0)
        built_cache[ver] = b
        p = {k: v[start:end] for k, v in b["parts"].items()}
        cnt = lambda m: int(m.sum())  # noqa: E731
        L0 = p["價格存在"]
        L1 = L0 & p["流動性"] & p["均線60日有值"]
        L2 = L1 & p["大盤濾網"]
        L3 = L2 & p["非近期注意股"] & p["非處置股"]
        strong = p["B1均線多頭"] & p["B2接近52週高"] & p["B3漲幅排名"] & p["B4勝過大盤"]
        L4 = L3 & strong
        L5 = L4 & p["進場觸發"]
        L6 = L5 & p["未追高"] & p["乖離合格"]
        funnel_lines += [f"-- {ver}（開發期股票日數，括號為占上一層的比例）--"]
        prev = None
        for name, m in (("有價格", L0), ("流動性≥門檻且均線有值", L1), ("且大盤在 60 日均線之上", L2),
                        ("且非近期注意股、非處置股", L3), ("且四個強勢條件全部通過", L4), ("且進場觸發", L5),
                        ("且未追高、乖離合格（最終候選）", L6)):
            n = cnt(m)
            funnel_lines.append(f"  {name}：{n}" + (f"（{n / prev:.1%}）" if prev else ""))
            prev = n if n else 1
        funnel_lines.append("  各強勢條件單獨通過率（在『非注意處置』的股票日中）："
                            + "、".join(f"{k} {cnt(L3 & p[k]) / max(cnt(L3), 1):.1%}"
                                       for k in ("B1均線多頭", "B2接近52週高", "B3漲幅排名", "B4勝過大盤", "進場觸發")))
        sig_days = L6.any(axis=1)
        by_year = pd.Series(L6.sum(axis=1), index=cal[start:end]).groupby(cal[start:end].year).sum()
        funnel_lines.append("  最終候選每年股票日：" + "  ".join(f"{y}:{n}" for y, n in by_year.items()))
        check(f"漏斗｜{ver} 開發期有最終候選", cnt(L6) > 0, f"{cnt(L6)} 個股票日，涵蓋 {int(sig_days.sum())} 天")
    emit("B. 訊號漏斗", funnel_lines)

    # ---- C1. 前視偏誤：把 T 日之後的資料全部切掉，T 日（含以前）的訊號必須完全相同
    la_lines = []
    for ver in VERSIONS:
        for cutoff in ("2010-06-30", "2014-03-31"):
            T = int(cal.searchsorted(pd.Timestamp(cutoff)))
            dt_ = {k: v.iloc[:T + 1] for k, v in d.items()}
            a = alerts_for(sig0.attn_days)
            bt = ss.build(dt_, mkt.iloc[:T + 1], cal[:T + 1], a[1][:T + 1], a[2][:T + 1], sig0, ver, cfg0.ma_exit)
            same = (np.array_equal(bt["entry"], built_cache[ver]["entry"][:T + 1])
                    and np.array_equal(bt["elig"], built_cache[ver]["elig"][:T + 1]))
            check(f"前視偏誤｜{ver}：切掉 {cutoff} 之後的資料，訊號不變", same)
            la_lines.append(f"  {ver} 切點 {cutoff}（索引 {T}）：{'一致' if same else '不一致'}")
    emit("C1. 前視偏誤測試", ["把切點之後的價格與成交量全部移除，重新計算訊號，切點以前的進場與合格矩陣必須與完整資料相同。",
                           "（注意股／處置股的矩陣是由公告表直接標記，不含價格資訊，另於 C2 檢查。）", *la_lines])

    # ---- C2. 注意股／處置股旗標：用暴力逐列方式獨立重建，與矩陣比對
    rng = random.Random(0)
    sample_ids = rng.sample(cols, 12)
    col = {s_: i for i, s_ in enumerate(cols)}
    ok_attn = ok_dk = ok_di = True
    for sid in sample_ids:
        j = col[sid]
        exp_attn = np.zeros(len(cal), dtype=bool)
        for (dt,) in con.execute("SELECT date FROM attention WHERE stock_id=?", (sid,)):
            i = cal.searchsorted(pd.Timestamp(dt))
            if i < len(cal):
                exp_attn[i] = True
        exp_known, exp_in = np.zeros(len(cal), dtype=bool), np.zeros(len(cal), dtype=bool)
        for ann, st, en in con.execute("SELECT announce_date,period_start,period_end FROM disposition WHERE stock_id=?", (sid,)):
            for i in range(len(cal) - 1):
                ci, ni = cal[i], cal[i + 1]
                if ci >= pd.Timestamp(ann) and ni <= pd.Timestamp(en):
                    exp_known[i] = True
            for i in range(len(cal)):
                if pd.Timestamp(st) <= cal[i] <= pd.Timestamp(en):
                    exp_in[i] = True
        ok_attn &= np.array_equal(attn0[:, j], exp_attn)
        ok_dk &= np.array_equal(disp_known0[:, j], exp_known)
        ok_di &= np.array_equal(disp_in0[:, j], exp_in)
    check("旗標｜注意股標記與逐列重建一致", ok_attn, f"抽 {len(sample_ids)} 檔")
    check("旗標｜處置『已公布且未結束』標記與逐列重建一致", ok_dk)
    check("旗標｜處置『期間內』標記與逐列重建一致", ok_di)
    emit("C2. 注意股／處置股旗標", [f"抽樣股票：{', '.join(sample_ids)}；以逐列暴力方式重建旗標並與矩陣逐元素比對。"])

    # ---- C3/D/E. 基準回測：帳戶不變量、交易獨立重算、明細抽樣、績效摘要
    last_close = d["C"].iloc[:end].ffill().iloc[-1].to_numpy(dtype="float64")
    mkt_s = mkt.copy()
    perf_lines, detail_lines, inv_done = [], [], False
    results: dict = {}
    for ver in VERSIONS:
        b = built_cache[ver]
        for ename, tp in (("b(無固定停利)", None), ("a(停利2%)", 0.02)):
            for slip in (0.0, SENS_SLIP):
                cfg = account.Cfg(**{**asdict(cfg0), "tp_pct": tp, "slip": slip})
                m = ss.market(d, cal, b, cfg, alerts_for(sig0.attn_days))
                res = account.simulate(m, b["entry"], b["rank"], b["elig"], cfg, start, end)
                results[(ver, ename, slip)] = (res, cfg)
                label = f"{ver} / {ename} / 滑價 {slip:.2%}"
                lines, st = ss.summarize(label, res, cal, start)
                perf_lines += ["", *lines, "    事件計數：" + "、".join(f"{k} {v}" for k, v in sorted(res.counters.items()))]
                ss.log_trial({"time": datetime.now().isoformat(timespec="seconds"), "period": "smoke", "version": ver,
                              "exit": ename, "slip": slip, "sig": str(asdict(sig0)), "cfg": str(asdict(cfg)),
                              **{k: st[k] for k in ("cagr", "mdd", "triggers", "trades", "win", "avg_net", "avg_ret")}})
                if slip == 0.0:
                    invariants(res, cfg, label, last_close)
                print(f"  基準 {label}：{len(res.trades)} 筆", flush=True)

    # 獨立重算抽樣（只用 slip=0、b 版本，涵蓋各種出場原因）
    for ver in VERSIONS:
        res, cfg = results[(ver, "b(無固定停利)", 0.0)]
        res_a, cfg_a = results[(ver, "a(停利2%)", 0.0)]
        pool = [(t, cfg) for t in res.trades] + [(t, cfg_a) for t in res_a.trades if t["reason"] == "停利"]
        by_reason: dict[str, list] = {}
        for t, c_ in pool:
            by_reason.setdefault(t["reason"], []).append((t, c_))
        picks = []
        for reason, lst in sorted(by_reason.items()):
            rng.shuffle(lst)
            picks += lst[:4]
        detail_lines += ["", f"-- {ver}：抽樣 {len(picks)} 筆（各出場原因最多 4 筆），逐筆以資料庫原始資料獨立重算 --",
                         "  代號   進場日      進場價    股數  停損價(還原)  出場日      出場價    原因      淨損益   驗證"]
        n_ok = n_fail = n_skip = 0
        for t, c_ in picks:
            fails, notes = verify_trade(con, t, ver, sig0, c_, cols, cal, mkt_s)
            status = "略過(" + notes[0] + ")" if fails is None else ("通過" if not fails else "失敗：" + "；".join(fails))
            n_ok += fails == [] if fails is not None else 0
            n_fail += bool(fails)
            n_skip += fails is None
            detail_lines.append(f"  {cols[t['stock']]:<6} {cal[t['entry_idx']].date()}  {t['entry_raw']:>8.2f} {t['shares']:>5} "
                                f"{t['stop_adj']:>12.2f}  {cal[t['exit_idx']].date()}  {t['exit_raw']:>8.2f}  {t['reason']:<6} "
                                f"{t['net']:>8.1f}   {status}")
        check(f"獨立重算｜{ver} 抽樣交易的進場、出場、損益全部與資料庫重算一致", n_fail == 0 and n_ok > 0,
              f"通過 {n_ok}、失敗 {n_fail}、略過 {n_skip}")
    emit("C3. 帳戶不變量與邊界情況計數", [
        "不變量（現金、持股數、風險、手續費、對帳）逐項列在最上方總表。", *perf_lines])
    emit("D. 交易明細抽樣與獨立重算", detail_lines)

    # ---- F. 敏感度測試（滑價固定 0.25%，出場方式 b）
    base_exit = {"tp_pct": None}
    ablations = {
        "基準": {},
        "拿掉 B1 均線多頭": {"ma_trend": False},
        "拿掉 B2 接近 52 週高": {"near_high": 0.0},
        "拿掉 B3 漲幅排名": {"rs_top": 1.0},
        "拿掉 B4 勝過大盤": {"beat_market": False},
        "拿掉大盤濾網": {"market_filter": False},
        "拿掉乖離上限": {"bias_max": 9.9},
        "四個強勢條件全拿掉（只剩觸發）": {"ma_trend": False, "near_high": 0.0, "rs_top": 1.0, "beat_market": False},
    }
    params = {
        "停損 5%": {"stop_pct": 0.05}, "停損 10%": {"stop_pct": 0.10}, "停損 15%": {"stop_pct": 0.15},
        "均線出場 5 日": {"ma_exit": 5}, "不用均線出場": {"ma_exit": None},
        "時間停損 5 日": {"time_stop": 5}, "時間停損 15 日": {"time_stop": 15},
        "固定停利 1.5%": {"tp_pct": 0.015}, "固定停利 3%": {"tp_pct": 0.03},
        "持股被列注意股不出場": {"exit_on_attention": False},
        "流動性 0.5 億": {"liq_min": 5e7}, "流動性 3 億": {"liq_min": 3e8},
        "注意股排除 10 日": {"attn_days": 10}, "乖離上限 10%": {"bias_max": 0.10}, "乖離上限 20%": {"bias_max": 0.20},
        "52 週高門檻 0.85": {"near_high": 0.85}, "52 週高門檻 0.95": {"near_high": 0.95},
        "漲幅排名前 10%": {"rs_top": 0.1}, "漲幅排名前 30%": {"rs_top": 0.3},
        "漲幅天數 20 日": {"rs_n": 20}, "漲幅天數 120 日": {"rs_n": 120},
    }
    per_version = {"breakout": {"突破量比 1.2 倍": {"vol_k": 1.2}, "突破量比 2 倍": {"vol_k": 2.0},
                                "突破 60 日新高": {"breakout_n": 60}},
                   "pullback": {"回檔均線 20 日": {"pullback_ma": 20}, "回檔容許 2%": {"pullback_tol": 0.02}}}
    sens_lines = ["滑價 0.25%（單邊），出場方式 b（無固定停利，除非該列調整），開發期。每列都是一個設定，記入 trials.csv（period=smoke）。",
                  "欄位：交易筆數｜每筆平均淨利（元）｜每筆平均報酬｜年化｜最大回撤｜停機線每年次數",
                  "及格線（使用者 2026-10-08 同意）：每筆平均淨利 > 0 且單邊滑價 0.25% 下仍 > 0。"]
    cfg_keys = {f for f in asdict(cfg0)}
    for ver in VERSIONS:
        sens_lines += ["", f"### {ver}"]
        todo = [("【逐一拿掉條件】" + k, v) for k, v in ablations.items()]
        todo += [("【參數】" + k, v) for k, v in {**params, **per_version[ver]}.items()]
        for name, ov in todo:
            sig = ss.Sig(**{**asdict(sig0), **{k: v for k, v in ov.items() if k not in cfg_keys}})
            cfg = account.Cfg(**{**asdict(cfg0), "slip": SENS_SLIP, **base_exit, **{k: v for k, v in ov.items() if k in cfg_keys}})
            b, a = build(ver, sig, cfg)
            m = ss.market(d, cal, b, cfg, a)
            res = account.simulate(m, b["entry"], b["rank"], b["elig"], cfg, start, end)
            lines, st = ss.summarize(name, res, cal, start)
            yrs_n = (end - start) / 245
            mark = "✓" if st["avg_net"] > 0 else "✗"
            sens_lines.append(f"  {name:<28}{st['trades']:>5} 筆｜{st['avg_net']:>+7.1f} 元 {mark}｜{st['avg_ret']:>+7.2%}｜"
                              f"{st['cagr']:>+7.1%}｜{st['mdd']:>4.0%}｜{st['triggers'] / yrs_n:.1f}")
            ss.log_trial({"time": datetime.now().isoformat(timespec="seconds"), "period": "smoke", "version": ver,
                          "exit": name, "slip": SENS_SLIP, "sig": str(asdict(sig)), "cfg": str(asdict(cfg)),
                          **{k: st[k] for k in ("cagr", "mdd", "triggers", "trades", "win", "avg_net", "avg_ret")}})
            print(f"  敏感度 {ver} {name}", flush=True)
        # 隨機挑選對照（相同出場與過濾）
        b, a = build(ver, sig0, cfg0)
        cfg = account.Cfg(**{**asdict(cfg0), "slip": SENS_SLIP})
        m = ss.market(d, cal, b, cfg, a)
        rr = np.random.default_rng(0)
        cagrs, avgs = [], []
        for _ in range(ss.N_CONTROL):
            r_ = account.simulate(m, np.ones_like(b["entry"]), rr.random(b["rank"].shape), b["elig_base"], cfg, start, end)
            _, s2 = ss.summarize("c", r_, cal, start)
            cagrs.append(s2["cagr"]); avgs.append(s2["avg_net"])
        sens_lines.append(f"  【隨機挑選對照 {ss.N_CONTROL} 次】每筆平均淨利中位數 {np.median(avgs):+.1f} 元，"
                          f"年化中位數 {np.median(cagrs):+.1%}（5%～95% {np.percentile(cagrs, 5):+.1%}～{np.percentile(cagrs, 95):+.1%}）")
    emit("F. 敏感度測試（部分資料，僅供除錯）", sens_lines)
    write_report(done=True)
    ok = sum(c[1] for c in checks)
    print(f"檢查 {ok}/{len(checks)} 通過；報告：{REPORT}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
