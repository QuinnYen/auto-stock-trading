"""資金×持股數探索：短期強勢股規則（突破型、回檔型，見 docs/台股強勢股規則草案 v0.1.md）在不同起始資金與持股數下的帳戶模擬。

動機（使用者 2026-10-09）：2 萬元不行，就放寬資金限制看看這條規則本身的結論。
設計：4 種起始資金 × 持股數上限 3／10／30 檔 × 突破型／回檔型；每檔買當時淨值的 1/N（固定比例）；拿掉帳戶停機線；
其餘沿用強勢股規則基準（單檔停損 7%、收盤跌破 10 日均線隔天賣、持有滿 10 日賣、注意／處置股出場、不設停利、大盤濾網）；
滑價 0% 與 0.25% 各跑一次，另加隨機挑股對照（同樣的出場與過濾）。

只跑開發期 2007～2016，沒有驗證期／最後測試期選項。這是探索，不做通過與否的判定；每個設定（版本×資金×持股數）記一次試驗
（trials.csv，period ＝ explore-dev）。已知限制：資金很大而持股數很少時，單筆金額可能占個股成交量相當比例，
固定 0.25% 滑價會低估實際衝擊。

用法：python src/backtest/explore_capital.py
報告：data/experiments/explore_capital_dev_report.txt
"""
import argparse
import json
import sqlite3
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import account  # noqa: E402
import account_sim_v2 as v2  # noqa: E402
import holdout  # noqa: E402
import prices  # noqa: E402
import strong_stocks as ss  # noqa: E402

CAPITALS = (20_000, 200_000, 2_000_000, 20_000_000)
SLOTS = (3, 10, 30)
VERSIONS = ("breakout", "pullback")
SLIPS_X = (0.0, 0.0025)
N_CONTROL_X = 8
VERSION_NAME = {"breakout": "突破型", "pullback": "回檔型"}


def make_cfg_x(n, slip):
    return account.Cfg(risk_sized=False, stop_close=False, stop_pct=0.07, ma_exit=10, tp_pct=None, time_stop=10,
                       exit_on_attention=True, max_pos=n, pos_cap=1 / n, slip=slip, stop_line=1.0, restart_after=20)


def grid():
    return [(ver, cap, n) for ver in VERSIONS for cap in CAPITALS for n in SLOTS]


def summarize_x(res, capital):
    eq = np.asarray(res.equity, dtype="float64")
    years = len(eq) / 245
    mult = float(eq[-1] / capital)
    t = pd.DataFrame(res.trades)
    return {"final": float(eq[-1]), "mult": mult, "cagr": mult ** (1 / years) - 1,
            "mdd": float((1 - eq / np.maximum.accumulate(eq)).max()), "trades": len(t),
            "win": float((t.net > 0).mean()) if len(t) else 0.0, "avg_net": float(t.net.mean()) if len(t) else 0.0,
            "avg_ret": float((t.net / t.basis).mean()) if len(t) else 0.0, "avg_held": float(np.mean(res.n_held)),
            "cost": float(res.cost_total)}


def build_report(rows, control, bh, period):
    """rows：[{cap, n, ver, s0, s25}]（s0＝滑價 0%、s25＝滑價 0.25% 的 summarize_x 結果）；control：{(cap, n): {...}}。"""
    lines = [f"# 資金×持股數探索報告（開發期 {period[0]}～{period[1]}；短期強勢股規則，帳戶停機線已拿掉）", "",
             "**這是探索，不是通過／不通過的判定**：沒有事前約定的過關標準，結果只用來回答『放寬資金限制後，這條規則本身有沒有賺頭』。",
             f"參考：同期間買進持有 0050，2 萬元變 {bh[0]:,.0f} 元（約 {bh[0] / 20000:.2f} 倍），最大回檔 {bh[1]:.0%}。",
             "限制：資金大而持股數少時，單筆金額占個股成交量比例高，固定 0.25% 滑價會低估實際衝擊；2 萬元分 30 檔每檔不到 700 元，"
             "受『每檔至少 10 股』限制，實際持股數會遠低於 30（見『平均持股』）。", ""]
    for ver in VERSIONS:
        lines += [f"## {VERSION_NAME[ver]}", "",
                  "| 起始資金 | 持股上限 | 平均持股 | 期末倍數（滑價 0%） | 年化（0%） | 期末倍數（滑價 0.25%） | 年化（0.25%） | 最大回檔（0.25%） | 交易筆數 | 勝率 | 每筆平均報酬（0.25%） | 隨機挑股年化（0.25%，中位數） |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in rows if x["ver"] == ver]:
            s0, s25, c = r["s0"], r["s25"], control.get((r["cap"], r["n"]))
            lines.append(f"| {r['cap']:,} 元 | {r['n']} 檔 | {s25['avg_held']:.1f} | {s0['mult']:.2f} | {s0['cagr']:+.1%} | {s25['mult']:.2f} | "
                         f"{s25['cagr']:+.1%} | {s25['mdd']:.0%} | {s25['trades']:,} | {s25['win']:.0%} | {s25['avg_ret']:+.2%} | "
                         + (f"{c['cagr_med']:+.1%}（{c['cagr_lo']:+.1%}～{c['cagr_hi']:+.1%}）" if c else "—") + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    argparse.ArgumentParser().parse_args()
    holdout.require_access("dev", "explore_capital")
    start, end = ss.PERIODS["dev"]
    con = sqlite3.connect(prices.DB)
    print("載入資料…", flush=True)
    d, mkt, cal = ss.load_data(con)
    lo = int(cal.searchsorted(pd.Timestamp(start)))
    hi = int(cal.searchsorted(pd.Timestamp(end), side="right"))
    assert start <= str(cal[lo].date()) and str(cal[hi - 1].date()) <= end, "期間超出開發期"
    sig = ss.Sig()
    alerts = ss.alert_matrices(con, list(d["C"].columns), cal, sig.attn_days)
    rows, control = [], {}
    market_obj = None
    for ver in VERSIONS:
        built = ss.build(d, mkt, cal, alerts[1], alerts[2], sig, ver, 10)
        market_obj = ss.market(d, cal, built, make_cfg_x(3, 0.0), alerts)
        for _, cap, n in [g for g in grid() if g[0] == ver]:
            t0 = time.time()
            s = {}
            for slip in SLIPS_X:
                res = account.simulate(market_obj, built["entry"], built["rank"], built["elig"], make_cfg_x(n, slip), lo, hi, cap)
                s[slip] = summarize_x(res, cap)
            rows.append({"cap": cap, "n": n, "ver": ver, "s0": s[0.0], "s25": s[0.0025]})
            ss.log_trial({"time": datetime.now().isoformat(timespec="seconds"), "period": "explore-dev", "version": ver,
                          "exit": f"cap={cap} slots={n}", "slip": 0.0025, "sig": json.dumps(asdict(sig)),
                          "cfg": json.dumps(asdict(make_cfg_x(n, 0.0025))), "cagr": s[0.0025]["cagr"], "mdd": s[0.0025]["mdd"],
                          "triggers": 0, "trades": s[0.0025]["trades"], "win": s[0.0025]["win"],
                          "avg_net": s[0.0025]["avg_net"], "avg_ret": s[0.0025]["avg_ret"]})
            print(f"{ver} 資金 {cap:,} 持股 {n}：{time.time() - t0:.0f} 秒", flush=True)
        if ver == VERSIONS[0]:
            for cap in CAPITALS:
                for n in SLOTS:
                    rng = np.random.default_rng(0)
                    cagrs, mdds = [], []
                    for _ in range(N_CONTROL_X):
                        r = account.simulate(market_obj, np.ones_like(built["entry"]), rng.random(built["rank"].shape), built["elig_base"],
                                             make_cfg_x(n, 0.0025), lo, hi, cap)
                        st = summarize_x(r, cap)
                        cagrs.append(st["cagr"])
                        mdds.append(st["mdd"])
                    control[(cap, n)] = {"cagr_med": float(np.median(cagrs)), "cagr_lo": float(min(cagrs)), "cagr_hi": float(max(cagrs)),
                                         "mdd_med": float(np.median(mdds))}
                    print(f"隨機對照 資金 {cap:,} 持股 {n} 完成", flush=True)
    text = build_report(rows, control, v2.bh_0050(con, cal, lo, hi), (start, str(cal[hi - 1].date())))
    ss.OUT.mkdir(parents=True, exist_ok=True)
    (ss.OUT / "explore_capital_dev_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
