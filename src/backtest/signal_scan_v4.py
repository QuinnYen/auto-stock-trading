"""訊號掃描 v4：量價心法 K1（盤整後帶量長紅突破）、K2（突破後量縮不破低的回檔）（依 docs/訊號掃描事前約定 v4.md，2026-10-09 確認）。

沿用 signal_scan.py 的進場資格、報酬、同日基準、隨機化檢定、BH 與通過條件；只掃開發期 2007-01-01～2016-12-31。
8 個（訊號 × 持有期 10／20／40／60）檢定全部寫入 data/experiments/trials.csv（period ＝ scan4-dev），報告 scan4_dev_report.txt。
本程式沒有驗證期／最後測試期的選項。

K1（訊號日 t，以下 C、O、V 為還原價與成交量，「前 N 日」不含今天）：
  前 60 日收盤的（最高÷最低−1）≤ 20%；今日收盤 > 前 60 日收盤最高；今日量 ≥ 前 20 日均量 × 2.5；（C−O）÷O ≥ 5%。
K2：關鍵日固定為 t−5 且該日滿足 K1；t−2、t−1、t 的收盤都 ≥ 關鍵日最低價；t−2、t−1、t 每日收盤不高於前一日；
  今日量 ≤ 關鍵日量 × 35%；（H−L）÷前一日收盤 < 3%；|收盤 − 5 日均線| ÷ 5 日均線 ≤ 2%。
窗口資料不足（含停牌缺值）時不產生訊號；比例先四捨五入到 10 位再與門檻比較，避免浮點誤差吃掉邊界。

用法：python src/backtest/signal_scan_v4.py
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

HORIZONS = (10, 20, 40, 60)
SPECS4 = [s1.Spec("K1", "盤整後帶量長紅突破", HORIZONS, 0.0025, "std"),
          s1.Spec("K2", "突破後量縮不破低回檔", HORIZONS, 0.0025, "std")]
N_TESTS4 = sum(len(s.horizons) for s in SPECS4)


def scan_config():
    """(檢定清單, 檢定數, 訊號日期間, 正超額年數門檻)；只有開發期。"""
    return SPECS4, N_TESTS4, s1.DEV, s1.MIN_POS_YEARS


def shift_rows(a, k):
    """把陣列（日 × 股）往下移 k 列，前 k 列補 NaN。"""
    out = np.full(a.shape, np.nan)
    out[k:] = a[:-k]
    return out


def roll_rows(a, n, how):
    """沿日期方向的滾動統計（含當列、視窗必須填滿 n 個有效值）。"""
    return getattr(pd.DataFrame(a).rolling(n, min_periods=n), how)().to_numpy()


def k1_signal(O, C, V):
    prev_c, prev_v = shift_rows(C, 1), shift_rows(V, 1)
    hi, lo = roll_rows(prev_c, 60, "max"), roll_rows(prev_c, 60, "min")
    avg = roll_rows(prev_v, 20, "mean")
    with np.errstate(divide="ignore", invalid="ignore"):
        compress = np.round(hi / lo - 1, 10) <= 0.20
        big_vol = np.round(V / avg, 10) >= 2.5
        body = np.round((C - O) / O, 10) >= 0.05
    return compress & (C > hi) & big_vol & body


def k2_signal(k1, H, L, C, V):
    key = shift_rows(k1.astype(float), 5) == 1.0
    Lk, Vk = shift_rows(L, 5), shift_rows(V, 5)
    c1, c2, c3 = shift_rows(C, 1), shift_rows(C, 2), shift_rows(C, 3)
    ma5 = roll_rows(C, 5, "mean")
    with np.errstate(divide="ignore", invalid="ignore"):
        holds = (C >= Lk) & (c1 >= Lk) & (c2 >= Lk)
        falling = (C <= c1) & (c1 <= c2) & (c2 <= c3)
        quiet = np.round(V / Vk, 10) <= 0.35
        narrow = np.round((H - L) / c1, 10) < 0.03
        near = np.round(np.abs(C - ma5) / ma5, 10) <= 0.02
    return key & holds & falling & quiet & narrow & near


def prepare4(con, period=s1.DEV):
    ctx = s1.prepare(con, period)
    a = {k: ctx["d"][k].to_numpy(dtype="float64") for k in ("O", "H", "L", "C", "V")}
    print("建立 K1、K2 訊號…", flush=True)
    k1 = k1_signal(a["O"], a["C"], a["V"])
    k2 = k2_signal(k1, a["H"], a["L"], a["C"], a["V"])
    ctx["sigs"] = {**ctx["sigs"], "K1": k1, "K2": k2}
    return ctx


NOTES = ["", "---", "## 實作細節說明（約定未明寫處的處理）",
         "- 視窗內有缺值（停牌）或歷史不足 60 日時不產生訊號；比例先四捨五入到 10 位再與門檻比較。",
         "- K2 的 5 日均線含當日；關鍵日固定為 t−5 且需滿足 K1 的全部條件。",
         "- 心法來源為使用者貼入的第三方建議，績效宣稱未採信；本報告為其在開發期的獨立檢驗。"]


def main():
    argparse.ArgumentParser().parse_args()
    holdout.require_access("dev", "signal_scan_v4")
    specs, n_tests, period, min_pos_years = scan_config()
    con = sqlite3.connect(prices.DB)
    ctx = prepare4(con, period)
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
    assert len(results) == n_tests == 8

    print("隨機化檢定…", flush=True)
    pvals, any_rej, raw_rate = s1.randomization_pvalues(items, observed, np.random.default_rng(0))
    s1.apply_verdicts(results, pvals, min_pos_years)

    window = np.zeros(shape, dtype=bool)
    window[lo:hi] = True
    funnel = {"有價格": int((window & np.isfinite(C)).sum()), "且流動性 ≥ 1 億、原始價 ≤ 600 元": int((window & base).sum()),
              "且非近期注意股、非處置股（標準資格）": int((window & elig["std"]).sum()),
              "且隔天買得到（標準資格）": int((window & elig["std"] & entry_ok).sum())}
    for sid in ("K1", "K2"):
        funnel[f"{sid} 訊號日（未套資格）"] = int((window & ctx["sigs"][sid]).sum())
    text = s1.build_report(results, funnel, (any_rej, raw_rate), n_tests,
                           title="訊號掃描報告 v4：量價心法（開發期 2007～2016；依事前約定 v4）", years=10, dev_method_note=False)
    text = "\n".join([text, *NOTES])
    ss.OUT.mkdir(parents=True, exist_ok=True)
    (ss.OUT / "scan4_dev_report.txt").write_text(text, encoding="utf-8")
    for r in results:
        ss.log_trial({"time": datetime.now().isoformat(timespec="seconds"), "period": "scan4-dev", "version": r["sid"],
                      "exit": f"h={r['h']}", "slip": (r["cost"] - s1.FEE_RT) / 2, "sig": r["name"], "cfg": "",
                      "cagr": "", "mdd": "", "triggers": "", "trades": r["n"], "win": r["win_net"],
                      "avg_net": r["net"] * s1.POSITION if r["n"] else "", "avg_ret": r["net"]})
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
