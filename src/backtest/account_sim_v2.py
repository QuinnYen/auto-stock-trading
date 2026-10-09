"""帳戶模擬 v2 執行器（依 docs/帳戶模擬事前約定 v2.md，2026-10-09 確認）：3 個規則 × 2 個結構 ＝ 6 次正式試驗。

規則：A 創 250 日新高（H4b）、B 52 週高價比前 10% 月度（H4a）、D 月營收年增創高（H1a），皆持有 60 天。
結構：S2 ＝ 3 檔（每檔淨值 1/3）、S4 ＝ 6 檔（每檔淨值 1/6），帳戶停機線都是 20%；其餘（單檔收盤停損 15%、停機後停 20 日、
單邊滑價 0.25%、同日訊號依種子隨機挑、種子 1～20）沿用 v1。每個組合 20 個種子合起來記 1 次試驗（trials.csv，period ＝ acct2-dev）。
隨機進場對照只取決於結構（與規則無關），每個結構算一次，6 個組合各自引用。

預設只跑開發期 2007～2016。驗證期（依 docs/帳戶模擬事前約定 v3.md）只驗證 A-S2 一個組合，需 --val-confirm 並寫入帳本；
通過條件改為：期末中位數 ≥ 25,526 元、5 年中至少 3 年為正、最大回檔 ≤ 40%，其餘同 v2。
最後測試期（依 docs/帳戶模擬事前約定 v4（最後測試期）.md）同樣只跑 A-S2，需 --final-confirm 並寫入帳本，只能執行一次；
期末門檻依模擬交易日數換算（20,000 × 1.05^(N÷245)），其餘同驗證期。

用法：python src/backtest/account_sim_v2.py
      python src/backtest/account_sim_v2.py --period val --val-confirm
      python src/backtest/account_sim_v2.py --period final --final-confirm
報告：data/experiments/acct2_dev_report.txt／acct3_val_report.txt／acct4_final_report.txt
"""
import argparse
import functools
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import account  # noqa: E402
import account_sim as sim  # noqa: E402
import holdout  # noqa: E402
import prices  # noqa: E402
import signal_scan_v3 as v3  # noqa: E402
import strong_stocks as ss  # noqa: E402

START = account.START_CAPITAL
SLIP, STOP_PCT, N_SEEDS = sim.SLIP, sim.STOP_PCT, sim.N_SEEDS
MIN_WIN_RUNS, MIN_POS_YEARS = sim.MIN_WIN_RUNS, sim.MIN_POS_YEARS
MIN_FINAL_DEV = START * 1.05 ** 10      # 事前約定 v2 第 7 節：年化 5%、十年
MAX_MDD = 0.25                          # 事前約定 v2 第 7 節：停機線 20% ＋ 隔天開盤跳空容許 5 個百分點
MIN_FINAL_VAL = START * 1.05 ** 5       # 事前約定 v3 第 5 節：驗證期 5 年、年化 5%
MIN_POS_YEARS_VAL = 3                   # 5 個年度中至少 3 年為正
MAX_MDD_VAL = 0.40                      # 事前約定 v3：使用者放寬到 40%

STRUCTS = {"S2": (3, 0.20), "S4": (6, 0.20)}     # 結構 → (最多持股, 帳戶停機線)
RULES2 = [sim.Rule("A", "創 250 日新高（H4b）持有 60 天", 60),
          sim.Rule("B", "52 週高價比前 10%、月度（H4a）持有 60 天", 60),
          sim.Rule("D", "月營收年增創高（H1a）持有 60 天", 60)]
COMBOS = [sim.Rule(f"{r.sid}-{s}", f"{r.name}｜{STRUCTS[s][0]} 檔、停機線 {STRUCTS[s][1]:.0%}", r.h)
          for r in RULES2 for s in STRUCTS]
# v1（3 檔、停機線 10%）開發期的參考數字，取自 data/experiments/acct_dev_report.txt（不重跑）：(期末中位數, 最大回檔)
V1_REF = {"A": (37421, 0.44), "B": (25646, 0.53)}


def make_cfg2(h, struct, slip=SLIP, stop_pct=STOP_PCT):
    n, line = STRUCTS[struct]
    return account.Cfg(risk_sized=False, stop_close=True, stop_pct=stop_pct, ma_exit=None, exit_on_attention=False,
                       time_stop=h, max_pos=n, pos_cap=1 / n, slip=slip, stop_line=line, restart_after=20)


def judge2(finals, control_finals, years_ret, mdd, min_final=MIN_FINAL_DEV, min_pos_years=MIN_POS_YEARS, max_mdd=MAX_MDD):
    """事前約定 v2 第 7 節的 6 個條件；回傳 [(說明, 是否成立)]。years_ret 為中位數路徑的逐年報酬率。
    驗證期的門檻差異（期末金額、為正年數、回檔上限）由參數帶入，見 judge_val。"""
    finals, years_ret = np.asarray(finals), np.asarray(years_ret)
    med = np.median(finals)
    rest = np.delete(years_ret, int(np.argmax(years_ret)))
    return [(f"期末淨值中位數 {med:,.0f} ≥ {min_final:,.0f}（年化約 5%）", bool(med >= min_final)),
            (f"20 次中 {int((finals > START).sum())} 次賺錢（需 ≥ {MIN_WIN_RUNS}）", bool((finals > START).sum() >= MIN_WIN_RUNS)),
            (f"中位數高於隨機進場對照（{med:,.0f} vs {np.median(control_finals):,.0f}）", bool(med > np.median(control_finals))),
            (f"{int((years_ret > 0).sum())}／{len(years_ret)} 年為正（需 ≥ {min_pos_years}）", bool((years_ret > 0).sum() >= min_pos_years)),
            (f"最大回檔 {mdd:.1%}（需 ≤ {max_mdd:.0%}）", bool(mdd <= max_mdd)),
            (f"去掉最好的一年後剩餘年度累計 {np.prod(1 + rest) - 1:+.0%}（需 > 0）", bool(np.prod(1 + rest) > 1))]


def judge_val(finals, control_finals, years_ret, mdd):
    """事前約定 v3 第 5 節的驗證期 6 個條件。"""
    return judge2(finals, control_finals, years_ret, mdd, min_final=MIN_FINAL_VAL, min_pos_years=MIN_POS_YEARS_VAL, max_mdd=MAX_MDD_VAL)


def judge_final(finals, control_finals, years_ret, mdd, n_days):
    """事前約定 v4 第 4 節的最後測試期 6 個條件：期末門檻依模擬交易日數 N 換算為 20,000 × 1.05^(N÷245)。"""
    return judge2(finals, control_finals, years_ret, mdd, min_final=START * 1.05 ** (n_days / 245),
                  min_pos_years=MIN_POS_YEARS_VAL, max_mdd=MAX_MDD_VAL)


def bind_judge(period, judge, n_days):
    """最後測試期的門檻依模擬交易日數換算，把 n_days 綁進判定函式；其他期間原樣回傳。"""
    return functools.partial(judge, n_days=n_days) if period == "final" else judge


def mode_config(period):
    """各期間的設定：期間、要跑的組合、通過條件函式、trials.csv 標記、報告檔名。驗證期只有 A-S2。"""
    if period == "val":
        return {"period": holdout.PERIODS["val"], "combos": [c for c in COMBOS if c.sid == "A-S2"], "judge": judge_val,
                "tag": "acct3-val", "report": "acct3_val_report.txt", "title": "帳戶模擬 v3 驗證報告（驗證期"}
    if period == "final":
        return {"period": holdout.PERIODS["final"], "combos": [c for c in COMBOS if c.sid == "A-S2"], "judge": judge_final,
                "tag": "acct4-final", "report": "acct4_final_report.txt", "title": "帳戶模擬 v4 最後測試報告（最後測試期"}
    return {"period": holdout.PERIODS["dev"], "combos": COMBOS, "judge": judge2, "tag": "acct2-dev",
            "report": "acct2_dev_report.txt", "title": "帳戶模擬 v2 報告（開發期"}


def pick_preferred(passed_ids):
    """事前約定 v2 第 7 節與決定 4：多個組合通過時，S2（3 檔）優先於 S4；同一結構下 A ＞ B ＞ D。沒有通過者回傳 None。"""
    order = [c.sid for s in STRUCTS for c in COMBOS if c.sid.endswith("-" + s)]
    ranked = [i for i in order if i in set(passed_ids)]
    return ranked[0] if ranked else None


def buy_hold(close, capital=START):
    """買進持有參考線：回傳 (期末金額, 最大回檔)。"""
    close = np.asarray(close, dtype="float64")
    return float(capital * close[-1] / close[0]), sim.max_drawdown(close)


def d_entry(con, cal, cols, shape):
    """規則 D 的進場訊號：H1a 營收年增創高（訊號日為次月 10 日當天或之後第一個交易日）。"""
    rows = con.execute("SELECT stock_id, revenue_year, revenue_month, revenue FROM month_revenue").fetchall()
    return v3.revenue_signals(rows, cal, {c: i for i, c in enumerate(cols)}, shape)["H1a"]


def prepare2(con):
    ctx = sim.prepare(con)
    ctx["entries"]["D"] = d_entry(con, ctx["cal"], ctx["cols"], ctx["mkt"].C.shape)
    return ctx


def run_combo(ctx, combo, controls, lo, hi, seeds):
    """跑一個組合：回傳 (照規則的 Result 清單, 隨機進場對照, 敏感度情境)。對照只取決於結構，算過就存進 controls 重複使用。"""
    rid, sname = combo.sid.split("-")
    entry, h, mkt, std = ctx["entries"][rid], combo.h, ctx["mkt"], ctx["std"]
    if sname not in controls:
        controls[sname] = sim.run_control(mkt, std, make_cfg2(h, sname), lo, hi, seeds)
    res = sim.run_seeds(mkt, entry, std, make_cfg2(h, sname), lo, hi, seeds)
    extra = {"滑價 0%": sim.run_seeds(mkt, entry, std, make_cfg2(h, sname, slip=0.0), lo, hi, seeds),
             "滑價 0.5%": sim.run_seeds(mkt, entry, std, make_cfg2(h, sname, slip=0.005), lo, hi, seeds),
             "不設單檔停損": sim.run_seeds(mkt, entry, std, make_cfg2(h, sname, stop_pct=1.0), lo, hi, seeds)}
    return res, controls[sname], extra


def summarize_combo(combo, results, control, extra, cal, lo, judge_fn=judge2):
    return sim.summarize_rule(combo, results, control, extra, cal, lo, judge_fn=judge_fn, title=f"■ 組合 {combo.sid}：{combo.name}")


def bh_0050(con, cal, lo, hi):
    adj = prices.load_adjusted(con, ["0050"])
    s = adj[adj.stock_id == "0050"].set_index("date")["c"]
    s.index = pd.to_datetime(s.index)
    return buy_hold(s[(s.index >= cal[lo]) & (s.index <= cal[hi - 1])].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "val", "final"], default="dev")
    ap.add_argument("--val-confirm", action="store_true")
    ap.add_argument("--final-confirm", action="store_true")
    args = ap.parse_args()
    holdout.require_access(args.period, "account_sim_v2", val_confirm=args.val_confirm, final_confirm=args.final_confirm)
    cfg_m = mode_config(args.period)
    start, end = cfg_m["period"]
    con = sqlite3.connect(prices.DB)
    ctx = prepare2(con)
    cal = ctx["cal"]
    lo = int(cal.searchsorted(pd.Timestamp(start)))
    hi = int(cal.searchsorted(pd.Timestamp(end), side="right"))
    assert start <= str(cal[lo].date()) and str(cal[hi - 1].date()) <= end, "模擬期間超出指定範圍"
    judge_fn = bind_judge(args.period, cfg_m["judge"], hi - lo)
    seeds = range(1, N_SEEDS + 1)
    controls, body, summary = {}, [], []
    for combo in cfg_m["combos"]:
        t0 = time.time()
        res, ctrl, extra = run_combo(ctx, combo, controls, lo, hi, seeds)
        sname = combo.sid.split("-")[1]
        h = combo.h
        lines, passed, stat = summarize_combo(combo, res, ctrl, extra, cal, lo, judge_fn=judge_fn)
        body += lines + [""]
        summary.append((combo, passed, stat))
        ss.log_trial({"time": datetime.now().isoformat(timespec="seconds"), "period": cfg_m["tag"], "version": combo.sid,
                      "exit": f"h={h} stop{STOP_PCT:.0%} line{STRUCTS[sname][1]:.0%} slots={STRUCTS[sname][0]}", "slip": SLIP,
                      "sig": combo.name, "cfg": f"seeds={N_SEEDS}", "cagr": stat["cagr"], "mdd": stat["mdd"],
                      "triggers": stat["triggers"], "trades": stat["trades"], "win": stat["win"],
                      "avg_net": stat["avg_net"], "avg_ret": stat["avg_ret"]})
        print(f"組合 {combo.sid} 完成，{time.time() - t0:.0f} 秒", flush=True)

    bh_final, bh_mdd = bh_0050(con, cal, lo, hi)
    last = str(cal[hi - 1].date())
    head = [f"# {cfg_m['title']} {start}～{min(end, last)}，本金 {START:,.0f} 元）", ""]
    if args.period == "final":
        combo, ok, st = summary[0]
        head += [f"【白話摘要】{combo.sid} 在最後測試期**{'通過' if ok else '不通過'}**："
                 f"2 萬元照規則做約 {max(1, round((hi - lo) / 245, 1))} 年，20 次模擬的期末中位數 {st['final']:,.0f} 元（年化 {st['cagr']:+.1%}），最大回檔 {st['mdd']:.1%}。"
                 + ("三段獨立期間都通過；仍需紙上交易確認實際成交，才考慮實盤。" if ok else "依約定放棄此規則，不換參數重試。"),
                 f"參考：同期間買進持有 0050，2 萬元變 {bh_final:,.0f} 元，最大回檔 {bh_mdd:.0%}（只是參考，不是通過條件）。",
                 "對照：開發期 2007～2016 期末中位數 36,110 元（年化 +6.0%），最大回檔 39.7%；驗證期 2017～2021 期末中位數 31,486 元（年化 +9.5%），最大回檔 35.9%。", ""]
    elif args.period == "val":
        combo, ok, st = summary[0]
        head += [f"【白話摘要】{combo.sid} 在驗證期**{'通過' if ok else '不通過'}**："
                 f"2 萬元照規則做 5 年，20 次模擬的期末中位數 {st['final']:,.0f} 元（年化 {st['cagr']:+.1%}），最大回檔 {st['mdd']:.1%}。"
                 + ("依約定可進最後測試期（2022 起，只跑一次，尚未執行）。" if ok else "依約定放棄此規則，不換參數重試。"),
                 f"參考：同期間買進持有 0050，2 萬元變 {bh_final:,.0f} 元，最大回檔 {bh_mdd:.0%}（只是參考，不是通過條件）。",
                 "開發期對照（A-S2）：期末中位數 36,110 元（年化 +6.0%），最大回檔 39.7%，18／20 次賺錢。",
                 "提醒：2017～2021 是多頭，隨機進場對照也可能賺錢，所以『高於隨機進場對照』比絕對金額重要。", ""]
    else:
        chosen = pick_preferred([c.sid for c, ok, _ in summary if ok])
        head += ["【白話摘要】6 個組合中 **" + str(sum(ok for _, ok, _ in summary)) + " 個**通過全部條件。"
                 + (f"依偏好順序進驗證期的是 **{chosen}**。" if chosen else "沒有任何組合通過：在 2 萬元與這些限制下，本訊號族沒有可行的帳戶結構（依約定不換參數重試）。"),
                 f"參考：同期間買進持有 0050，2 萬元變 {bh_final:,.0f} 元，最大回檔 {bh_mdd:.0%}（只是參考，不是通過條件）。", "",
                 "| 組合 | 期末中位數 | 年化 | 最大回檔 | 停機線觸發 | 結論 | v1 對照（3 檔、停機線 10%） |", "|---|---|---|---|---|---|---|"]
        for c, ok, st in summary:
            ref = V1_REF.get(c.sid.split("-")[0])
            head.append(f"| {c.sid} | {st['final']:,.0f} 元 | {st['cagr']:+.1%} | {st['mdd']:.0%} | {st['triggers']} 次 | "
                        f"{'✅ 通過' if ok else '❌ 不通過'} | " + (f"{ref[0]:,} 元、回檔 {ref[1]:.0%}" if ref else "—（D 未跑過 v1）") + " |")
        head.append("")
    head += ["假設：每檔買淨值的 1/3（S2）或 1/6（S4）；收盤跌破進場價 15% 隔天開盤賣；帳戶停機線 20%（停 20 日後人工重啟）；"
             "單邊滑價 0.25%、手續費 3 折（每筆最低 1 元）、賣出證交稅 0.3%；持股遇處置期間賣出再加 1% 滑價；"
             "同日訊號多於空位時依種子隨機排序；S4 每檔預算較小，原始股價上限約 333 元（S2 約 667 元）。", ""]
    text = "\n".join(head + body)
    ss.OUT.mkdir(parents=True, exist_ok=True)
    (ss.OUT / cfg_m["report"]).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
