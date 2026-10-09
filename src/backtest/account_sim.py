"""帳戶模擬執行器（依 docs/帳戶模擬事前約定 v1.md）：三個代表規則 × 20 個隨機種子 ＋ 隨機進場對照。

規則 A：創 250 日新高持有 60 天；B：52 週高價比前 10%（月度）持有 60 天；C：5 日大跌 ≥10% 反轉持有 5 天。
每檔買淨值 1/3、收盤跌破進場價 15% 隔天開盤停損、帳戶停機線 10%、單邊滑價 0.25%；同日訊號多於空位時依種子隨機排序。
開發期通過條件見 judge()；同一規則的 20 組種子合起來記 1 次正式試驗（trials.csv，period ＝ acct-dev）。

用法：python src/backtest/account_sim.py [--period dev|val|final] [--final-confirm] [--seeds 20]
報告：data/experiments/acct_<period>_report.txt
"""
import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import account  # noqa: E402
import holdout  # noqa: E402
import prices  # noqa: E402
import signal_scan as sc  # noqa: E402
import strong_stocks as ss  # noqa: E402

SLIP = 0.0025
STOP_PCT = 0.15
N_SEEDS = 20
MIN_WIN_RUNS = 15           # 20 次中至少幾次期末淨值 > 本金
MIN_POS_YEARS = 6
MAX_MDD = 0.30
START = account.START_CAPITAL


@dataclass(frozen=True)
class Rule:
    sid: str
    name: str
    h: int


RULES = [Rule("A", "創 250 日新高（H4b）持有 60 天", 60),
         Rule("B", "52 週高價比前 10%、月度（H4a）持有 60 天", 60),
         Rule("C", "5 日大跌 ≥10% 反轉（H8）持有 5 天", 5)]


def make_cfg(h, slip=SLIP, stop_pct=STOP_PCT):
    return account.Cfg(risk_sized=False, stop_close=True, stop_pct=stop_pct, ma_exit=None, exit_on_attention=False,
                       time_stop=h, pos_cap=1 / 3, slip=slip, stop_line=0.10, restart_after=20)


def seeded_rank(shape, seed):
    return np.random.default_rng(seed).random(shape)


# ---------------------------------------------------------------- 績效整理
def max_drawdown(equity):
    eq = np.asarray(equity, dtype="float64")
    return float((1 - eq / np.maximum.accumulate(eq)).max())


def year_table(equity, dates, capital=START):
    """每年的 (報酬率, 賺賠金額)；第一年以本金為起點，其餘以上一年底淨值為起點。"""
    s = pd.Series(np.asarray(equity, dtype="float64"), index=pd.DatetimeIndex(dates))
    ends = s.groupby(s.index.year).last()
    starts = ends.shift(1).fillna(capital)
    return pd.DataFrame({"ret": ends / starts - 1, "pnl": ends - starts})


def median_index(finals):
    """中位數路徑：期末淨值由小到大排序後的第 ceil(n/2) 名（20 次即第 10 名，取偏低的中位數）。"""
    order = np.argsort(finals, kind="stable")
    return int(order[(len(finals) + 1) // 2 - 1])


def judge(finals, control_finals, years_ret, mdd):
    """事前約定第 7 節的 6 個條件；回傳 [(說明, 是否成立)]。years_ret 為中位數路徑的逐年報酬率。"""
    finals, years_ret = np.asarray(finals), np.asarray(years_ret)
    rest = np.delete(years_ret, int(np.argmax(years_ret)))
    return [(f"期末淨值中位數 {np.median(finals):,.0f} > {START:,.0f}", bool(np.median(finals) > START)),
            (f"20 次中 {int((finals > START).sum())} 次賺錢（需 ≥ {MIN_WIN_RUNS}）", bool((finals > START).sum() >= MIN_WIN_RUNS)),
            (f"中位數高於隨機進場對照（{np.median(finals):,.0f} vs {np.median(control_finals):,.0f}）",
             bool(np.median(finals) > np.median(control_finals))),
            (f"{int((years_ret > 0).sum())}／{len(years_ret)} 年為正（需 ≥ {MIN_POS_YEARS}）", bool((years_ret > 0).sum() >= MIN_POS_YEARS)),
            (f"最大回檔 {mdd:.0%}（需 ≤ {MAX_MDD:.0%}）", bool(mdd <= MAX_MDD)),
            (f"去掉最好的一年後剩餘年度累計 {np.prod(1 + rest) - 1:+.0%}（需 > 0）", bool(np.prod(1 + rest) > 1))]


# ---------------------------------------------------------------- 執行
def run_seeds(mkt, entry, elig, cfg, lo, hi, seeds):
    """每個種子跑一次帳戶模擬；回傳 Result 清單。"""
    return [account.simulate(mkt, entry, seeded_rank(entry.shape, s), elig, cfg, lo, hi) for s in seeds]


def run_control(mkt, std, cfg, lo, hi, seeds):
    """隨機進場對照：每天所有合格股票都是候選，同樣的資金、停損與持有期。"""
    return run_seeds(mkt, std, std, cfg, lo, hi, seeds)


def finals_of(results):
    return np.array([r.equity[-1] for r in results])


def prepare(con):
    print("載入資料…", flush=True)
    d, cal = sc.load(con)
    cols = list(d["C"].columns)
    attn, attn_recent, disp_known, disp_in = ss.alert_matrices(con, cols, cal, 5)
    _, elig = sc.build_eligibility(d["M"], d["RC"], d["C"], attn_recent, disp_known)
    f = lambda k: d[k].to_numpy(dtype="float64")  # noqa: E731
    last_idx = np.array([np.flatnonzero(np.isfinite(f("O")[:, j]))[-1] if np.isfinite(f("O")[:, j]).any() else -1
                         for j in range(len(cols))])
    zeros = np.zeros(d["C"].shape, dtype=bool)
    mkt = account.Market(O=f("O"), H=f("H"), L=f("L"), C=f("C"), RO=f("RO"), ma_break=zeros, attn_exit=zeros,
                         disp_exit=zeros, disp_in=disp_in, last_idx=last_idx)
    std = elig["std"]
    entries = {"A": sc.sig_h4b(d["C"]), "B": sc.top_fraction_monthly(sc.sig_h4a_score(d["C"], d["H"]), std, cal),
               "C": sc.sig_h8(d["C"])}
    return {"mkt": mkt, "cal": cal, "std": std, "entries": entries, "cols": cols, "d": d}


def summarize_rule(rule, results, control, extra, cal, lo, judge_fn=None, title=None):
    """一個規則的完整整理：判斷、中位數路徑、白話文字。extra ＝ {情境名: [Result]}。
    judge_fn：通過條件函式（預設 v1 的 judge）；title：第一行標題（預設「■ 規則 …」）。"""
    finals, cfin = finals_of(results), finals_of(control)
    mi = median_index(finals)
    med = results[mi]
    dates = cal[lo:lo + len(med.equity)]
    yt = year_table(med.equity, dates)
    mdd = max_drawdown(med.equity)
    checks = (judge_fn or judge)(finals, cfin, yt.ret.to_numpy(), mdd)
    passed = all(ok for _, ok in checks)
    trades = pd.DataFrame(med.trades)
    years = len(med.equity) / 245
    med_final = float(np.median(finals))
    cagr = (med_final / START) ** (1 / years) - 1

    def line(label, fin):
        return f"{label}：中位數 {np.median(fin):,.0f} 元（{np.median(fin) - START:+,.0f}），最差 {fin.min():,.0f}，最好 {fin.max():,.0f}"

    lines = [title or f"■ 規則 {rule.sid}：{rule.name}",
             f"  用 2 萬元照這個規則做 {max(1, round(years))} 年，{len(finals)} 組隨機挑選的結果：",
             "  " + line("照規則", finals),
             "  " + line("隨機進場對照（同樣的資金與停損，只是隨機挑股票）", cfin),
             f"  賺錢的次數：{int((finals > START).sum())}／{len(finals)}；中位數路徑最大回檔 {mdd:.0%}",
             "  中位數路徑逐年賺賠：" + "  ".join(f"{y}:{r.pnl:+,.0f}元" for y, r in yt.iterrows())]
    if len(trades):
        lines.append(f"  中位數路徑：{len(trades)} 筆交易，勝率 {(trades.net > 0).mean():.0%}，每筆平均淨利 "
                     f"{trades.net.mean():+,.0f} 元，手續費與稅合計 {med.cost_total:,.0f} 元，停機線 {len(med.triggers)} 次，"
                     f"平均持有資金比例 {np.nanmean((med.equity - med.cash) / med.equity):.0%}")
        lines.append("  出場原因：" + "、".join(f"{k} {v}" for k, v in trades.reason.value_counts().items()))
    for label, res in extra.items():
        lines.append("  " + line(f"敏感度（{label}）", finals_of(res)))
    lines.append("  通過條件：")
    lines += [f"    [{'✓' if ok else '✗'}] {txt}" for txt, ok in checks]
    lines.append(f"  結論：{'通過，可進驗證期' if passed else '不通過，放棄此規則（不換參數重試）'}")
    stat = {"cagr": cagr, "mdd": mdd, "triggers": len(med.triggers), "trades": len(trades),
            "win": float((trades.net > 0).mean()) if len(trades) else 0.0,
            "avg_net": float(trades.net.mean()) if len(trades) else 0.0,
            "avg_ret": float((trades.net / trades.basis).mean()) if len(trades) else 0.0, "final": med_final}
    return lines, passed, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "val", "final"], default="dev")
    ap.add_argument("--val-confirm", action="store_true")
    ap.add_argument("--final-confirm", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--rules", default="ABC")
    args = ap.parse_args()
    holdout.require_access(args.period, "account_sim", val_confirm=args.val_confirm, final_confirm=args.final_confirm)
    start, end = ss.PERIODS[args.period]
    con = sqlite3.connect(prices.DB)
    ctx = prepare(con)
    cal = ctx["cal"]
    lo = int(cal.searchsorted(pd.Timestamp(start)))
    hi = int(cal.searchsorted(pd.Timestamp(end), side="right"))
    seeds = range(1, args.seeds + 1)
    body, summary = [], []
    for rule in RULES:
        if rule.sid not in args.rules:
            continue
        t0 = time.time()
        entry = ctx["entries"][rule.sid]
        cfg = make_cfg(rule.h)
        res = run_seeds(ctx["mkt"], entry, ctx["std"], cfg, lo, hi, seeds)
        ctrl = run_control(ctx["mkt"], ctx["std"], cfg, lo, hi, seeds)
        extra = {"滑價 0%": run_seeds(ctx["mkt"], entry, ctx["std"], make_cfg(rule.h, slip=0.0), lo, hi, seeds),
                 "滑價 0.5%": run_seeds(ctx["mkt"], entry, ctx["std"], make_cfg(rule.h, slip=0.005), lo, hi, seeds),
                 "不設單檔停損": run_seeds(ctx["mkt"], entry, ctx["std"], make_cfg(rule.h, stop_pct=1.0), lo, hi, seeds)}
        lines, passed, stat = summarize_rule(rule, res, ctrl, extra, cal, lo)
        body += lines + [""]
        summary.append((rule, passed, stat))
        ss.log_trial({"time": datetime.now().isoformat(timespec="seconds"), "period": f"acct-{args.period}",
                      "version": rule.sid, "exit": f"h={rule.h} stop{STOP_PCT:.0%}", "slip": SLIP, "sig": rule.name,
                      "cfg": f"seeds={args.seeds}", "cagr": stat["cagr"], "mdd": stat["mdd"], "triggers": stat["triggers"],
                      "trades": stat["trades"], "win": stat["win"], "avg_net": stat["avg_net"], "avg_ret": stat["avg_ret"]})
        print(f"規則 {rule.sid} 完成，{time.time() - t0:.0f} 秒", flush=True)
    head = [f"# 帳戶模擬報告（{args.period}：{start}～{min(end, str(cal[hi - 1].date()))}，本金 {START:,.0f} 元）", "",
            "【白話摘要】" + "；".join(f"規則 {r.sid} 中位數期末 {s['final']:,.0f} 元（{s['final'] - START:+,.0f}），"
                                       f"{'通過' if p else '不通過'}" for r, p, s in summary), "",
            "假設：每檔買淨值 1/3；收盤跌破進場價 15% 隔天開盤賣；帳戶停機線 10%（停 20 日後人工重啟）；單邊滑價 0.25%、"
            "手續費 3 折（每筆最低 1 元）、賣出證交稅 0.3%；持股遇處置期間賣出再加 1% 滑價（沿用引擎的保守假設）；"
            "同日訊號多於空位時依種子隨機排序。", ""]
    text = "\n".join(head + body)
    ss.OUT.mkdir(parents=True, exist_ok=True)
    (ss.OUT / f"acct_{args.period}_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
