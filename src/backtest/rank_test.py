"""規則 A 的同日排序測試（依 docs/訊號掃描事前約定 v4.md 第 5 節）：用分數排序取代隨機挑選，與隨機挑選 20 個種子比較。

規則 A（創 250 日新高持有 60 天）× 結構 S2（3 檔）、S4（6 檔），停機線 20%，其餘同 docs/帳戶模擬事前約定 v2.md；只跑開發期。
排序：R1 壓縮度（近 20 日最高價÷最低價，越小越優先；含今天）、R2 速率（近 5 日報酬 − 近 20 日報酬，越大越優先）。
判定：排序結果的期末淨值高於 20 個隨機種子期末淨值由小到大的第 18 名，才算『有改善』；其他指標只報告。
4 次模擬各記 1 次帳戶層級試驗（trials.csv，period ＝ rank-dev）。本程式沒有驗證期／最後測試期選項。

用法：python src/backtest/rank_test.py
報告：data/experiments/rank_dev_report.txt
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
import account  # noqa: E402
import account_sim as sim  # noqa: E402
import account_sim_v2 as v2  # noqa: E402
import holdout  # noqa: E402
import prices  # noqa: E402
import signal_scan_v4 as v4  # noqa: E402
import strong_stocks as ss  # noqa: E402

RANKS = {"R1": "壓縮度（近 20 日高低比，越小越優先）", "R2": "速率（近 5 日報酬 − 近 20 日報酬，越大越優先）"}
STRUCTS_R = ("S2", "S4")


def rank_scores(H, L, C):
    """回傳 {R1, R2} 分數矩陣（日 × 股），分數越高越優先；視窗不足為 NaN。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = -(v4.roll_rows(H, 20, "max") / v4.roll_rows(L, 20, "min"))
        r2 = (C / v4.shift_rows(C, 5) - 1) - (C / v4.shift_rows(C, 20) - 1)
    return {"R1": r1, "R2": r2}


def improves(final, random_finals):
    """有改善 ＝ 期末淨值高於隨機種子期末淨值由小到大的第 18 名（前 10%）。"""
    return bool(final > np.sort(np.asarray(random_finals, dtype="float64"))[17])


def run_ranked(mkt, entry, score, elig, cfg, lo, hi):
    return account.simulate(mkt, entry, score, elig, cfg, lo, hi)


def stat_of(res):
    eq = np.asarray(res.equity, dtype="float64")
    years = len(eq) / 245
    t = pd.DataFrame(res.trades)
    return {"final": float(eq[-1]), "cagr": float((eq[-1] / account.START_CAPITAL) ** (1 / years) - 1), "mdd": sim.max_drawdown(eq),
            "trades": len(t), "win": float((t.net > 0).mean()) if len(t) else 0.0,
            "avg_ret": float((t.net / t.basis).mean()) if len(t) else 0.0, "triggers": len(res.triggers)}


def main():
    argparse.ArgumentParser().parse_args()
    holdout.require_access("dev", "rank_test")
    start, end = holdout.PERIODS["dev"]
    con = sqlite3.connect(prices.DB)
    ctx = sim.prepare(con)
    cal, mkt, std, entry = ctx["cal"], ctx["mkt"], ctx["std"], ctx["entries"]["A"]
    lo = int(cal.searchsorted(pd.Timestamp(start)))
    hi = int(cal.searchsorted(pd.Timestamp(end), side="right"))
    assert start <= str(cal[lo].date()) and str(cal[hi - 1].date()) <= end, "期間超出開發期"
    scores = rank_scores(mkt.H, mkt.L, mkt.C)
    seeds = range(1, sim.N_SEEDS + 1)
    lines = [f"# 規則 A 同日排序測試報告（開發期 {start}～{str(cal[hi - 1].date())}；依事前約定 v4 第 5 節）", "",
             "**判定**：排序結果的期末淨值需高於 20 個隨機種子期末淨值的第 18 名（隨機結果的前 10%）才算『有改善』；即使有改善也不直接進驗證期。", "",
             "| 結構 | 挑選方式 | 期末（2 萬元起） | 年化 | 最大回檔 | 交易筆數 | 勝率 | 每筆平均報酬 | 停機線 | 有改善？ |", "|---|---|---|---|---|---|---|---|---|---|"]
    for sname in STRUCTS_R:
        cfg = v2.make_cfg2(60, sname)
        rand = sim.run_seeds(mkt, entry, std, cfg, lo, hi, seeds)
        rf = sim.finals_of(rand)
        med = rand[sim.median_index(rf)]
        sm = stat_of(med)
        lines.append(f"| {sname} | 隨機挑選（20 個種子中位數） | {sm['final']:,.0f} 元（範圍 {rf.min():,.0f}～{rf.max():,.0f}；第 18 名 {np.sort(rf)[17]:,.0f}） | "
                     f"{sm['cagr']:+.1%} | {sm['mdd']:.0%} | {sm['trades']} | {sm['win']:.0%} | {sm['avg_ret']:+.2%} | {sm['triggers']} | — |")
        for rid in RANKS:
            res = run_ranked(mkt, entry, scores[rid], std, cfg, lo, hi)
            st = stat_of(res)
            ok = improves(st["final"], rf)
            lines.append(f"| {sname} | {rid} {RANKS[rid]} | {st['final']:,.0f} 元 | {st['cagr']:+.1%} | {st['mdd']:.0%} | {st['trades']} | "
                         f"{st['win']:.0%} | {st['avg_ret']:+.2%} | {st['triggers']} | {'✅ 有' if ok else '❌ 沒有'} |")
            ss.log_trial({"time": datetime.now().isoformat(timespec="seconds"), "period": "rank-dev", "version": f"A-{sname}-{rid}",
                          "exit": "h=60 stop15% line20% ranked", "slip": v2.SLIP, "sig": RANKS[rid], "cfg": f"{sname}",
                          "cagr": st["cagr"], "mdd": st["mdd"], "triggers": st["triggers"], "trades": st["trades"], "win": st["win"],
                          "avg_net": "", "avg_ret": st["avg_ret"]})
            print(f"{sname} {rid} 完成", flush=True)
        lines.append("")
    lines += ["說明：R1 的『過去 20 日』含今天；分數無法計算（歷史不足）者排在最後。隨機挑選的 20 個種子與 v2 相同（種子 1～20）。"]
    text = "\n".join(lines)
    ss.OUT.mkdir(parents=True, exist_ok=True)
    (ss.OUT / "rank_dev_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
