"""訊號掃描 v3（營收 H1、低券資比 H9、股利 H10）的測試（依 docs/訊號掃描事前約定 v3.md）。

全部用人工構造資料、手算預期值；不讀真實 market.db。碰到入口腳本的檢查一律走 tests/sandbox.py。
用法：python tests/test_signal_scan_v3.py        報告：data/experiments/signal_scan_v3_tests_report.txt
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "backtest"))
sys.path.insert(0, str(ROOT / "tests"))
import signal_scan as s1  # noqa: E402
import signal_scan_v3 as v3  # noqa: E402
from sandbox import run_sandboxed  # noqa: E402


def expect(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


def hits(M, cal, cols):
    """布林矩陣 → {(日期字串, 股票)}。"""
    return {(str(cal[i].date()), cols[j]) for i, j in zip(*np.nonzero(M))}


# ---------------------------------------------------------------- H1 營收
CAL_R = pd.bdate_range("2019-01-01", "2021-12-31")


def rev_rows(sid, series):
    return [(sid, y, m, v) for (y, m), v in series.items()]


def base_series(zero_2019_07=False):
    """2018、2019 每月 1000；2020、2021 見下（年增率手算寫在各測試註解）。"""
    s = {}
    for y in (2018, 2019):
        for m in range(1, 13):
            s[(y, m)] = 1000
    if zero_2019_07:
        s[(2019, 7)] = 0
    r20 = [1050, 1100, 1150, 1120, 1100, 950, 1300, 1250, 1100, 1100, 1100, 1100]
    for m, v in enumerate(r20, 1):
        s[(2020, m)] = v
    # 2021：1 月 1155（年增 10%），2 月 1000（年增 −9.09%），3～12 月與 2020 同（年增 0）
    s[(2021, 1)], s[(2021, 2)] = 1155, 1000
    for m in range(3, 13):
        s[(2021, m)] = r20[m - 1]
    return s


def rev_signals(rows, cols=("A",), cal=CAL_R):
    ci = {c: i for i, c in enumerate(cols)}
    return v3.revenue_signals(rows, cal, ci, (len(cal), len(cols)))


def t01_revenue_h1a_h1b_h1c_hand_computed():
    """2020 年增率 .05 .10 .15 .12 .10 −.05 .30 .25 .10 .10 .10 .10；2021-01 為 .10：H1a 只在 7 月，H1b 只在 7 月，H1c 在 12 月與次年 1 月。"""
    sig = rev_signals(rev_rows("A", base_series()))
    cols = ["A"]
    # 7 月營收 → 8/10（週一）；12 月 → 2021-01-10 是週日 → 01-11；2021-01 → 02-10
    expect(hits(sig["H1a"], CAL_R, cols) == {("2020-08-10", "A")}, hits(sig["H1a"], CAL_R, cols))
    expect(hits(sig["H1b"], CAL_R, cols) == {("2020-08-10", "A")}, hits(sig["H1b"], CAL_R, cols))
    expect(hits(sig["H1c"], CAL_R, cols) == {("2021-01-11", "A"), ("2021-02-10", "A")}, hits(sig["H1c"], CAL_R, cols))


def t02_signal_day_is_first_trading_day_on_or_after_10th():
    """訊號日 ＝ 次月 10 日當天或之後第一個交易日（週末順延）；絕不早於 10 日。"""
    cal = pd.bdate_range("2020-01-01", "2020-12-31")
    expect(cal[v3.release_index(2020, 9, cal)] == pd.Timestamp("2020-10-12"), "10/10 是週六 → 10/12")
    expect(cal[v3.release_index(2020, 7, cal)] == pd.Timestamp("2020-08-10"), "8/10 是週一 → 當天")
    expect(cal[v3.release_index(2020, 12 - 1, cal)] == pd.Timestamp("2020-12-10"), "12/10 是週四 → 當天")
    expect(v3.release_index(2020, 12, cal) == len(cal), "超出日曆 → 回傳 len(cal)")


def t03_yoy_threshold_boundaries_are_inclusive():
    """年增率恰為 20%（H1a）與恰為 10%（H1b、H1c）算達標，不被浮點誤差吃掉。"""
    s = {}
    for y in (2018, 2019):
        for m in range(1, 13):
            s[(y, m)] = 1000
    # 2020：1～6 月 −5%，7 月 +10%（H1b 邊界），8～12 月 0；2021：1 月 1200 → vs 2020-01(950)；改以專用序列測 20%
    for m in range(1, 13):
        s[(2020, m)] = 950 if m <= 6 else (1100 if m == 7 else 1000)
    sig = rev_signals(rev_rows("A", s))
    expect(hits(sig["H1b"], CAL_R, ["A"]) == {("2020-08-10", "A")}, "上月 −5%、本月恰 +10% → H1b")
    # H1a 邊界：2020 全年 +0%，2020-07 恰 +20% 且為窗口最高
    s2 = {k: 1000 for k in s}
    s2[(2020, 7)] = 1200
    sig2 = rev_signals(rev_rows("A", s2))
    expect(hits(sig2["H1a"], CAL_R, ["A"]) == {("2020-08-10", "A")}, "恰 +20% 且最高 → H1a")
    s2[(2020, 7)] = 1199
    expect(not rev_signals(rev_rows("A", s2))["H1a"].any(), "+19.9% 不算")


def t04_zero_or_missing_base_gives_no_signal():
    """去年同月營收為 0 → 該月年增率不存在；窗口內缺任何一個月就不產生 H1a／H1b／H1c（保守）。"""
    sig = rev_signals(rev_rows("B", base_series(zero_2019_07=True)), cols=("B",))
    cols = ["B"]
    expect(not sig["H1a"].any(), hits(sig["H1a"], CAL_R, cols))
    expect(not sig["H1b"].any(), hits(sig["H1b"], CAL_R, cols))
    # H1c：2020-08～2021-01 連續 6 個月年增率都存在且 ≥10% → 只有 2021-02-10
    expect(hits(sig["H1c"], CAL_R, cols) == {("2021-02-10", "B")}, hits(sig["H1c"], CAL_R, cols))


def t05_no_history_no_signal_and_columns_independent():
    """沒有前一年資料的股票沒有訊號；多檔股票互不影響。"""
    rows = rev_rows("A", base_series()) + rev_rows("C", {(2020, m): 5000 for m in range(1, 13)})
    sig = rev_signals(rows, cols=("A", "C"))
    expect(not sig["H1a"][:, 1].any() and not sig["H1b"][:, 1].any() and not sig["H1c"][:, 1].any(), "C 沒有 2019 年資料")
    expect(hits(sig["H1a"], CAL_R, ["A", "C"]) == {("2020-08-10", "A")}, "A 不受 C 影響")


def t06_revenue_no_lookahead():
    """只用某日以前已公開的營收與日曆重算，該日以前的訊號與全資料算出的完全一致。"""
    rows = rev_rows("A", base_series())
    full = rev_signals(rows)
    for T_date in ("2020-08-10", "2020-12-31", "2021-02-10", "2021-02-09"):
        T = int(CAL_R.searchsorted(pd.Timestamp(T_date)))
        cut = CAL_R[:T + 1]
        # 已公開 ＝ 次月 10 日 ≤ T 日（日曆起點之前就公開的舊營收當然已知）
        known = [r for r in rows if pd.Timestamp(r[1] + (r[2] == 12), r[2] % 12 + 1, 10) <= CAL_R[T]]
        part = rev_signals(known, cal=cut)
        for k in ("H1a", "H1b", "H1c"):
            expect((part[k] == full[k][:T + 1]).all(), f"{k} 在 {T_date} 前不一致")


# ---------------------------------------------------------------- H9 券資比
CAL_M = pd.bdate_range("2021-01-01", "2021-03-31")
MONTH_ENDS = {"2021-01-29": None, "2021-02-26": None, "2021-03-31": None}


def margin_rows(day, margin, shorts):
    return [(f"S{j:02d}", day, margin[j] if isinstance(margin, list) else margin, s) for j, s in enumerate(shorts)]


def run_h9(rows, elig=None, n=25):
    cols = [f"S{j:02d}" for j in range(n)]
    ci = {c: i for i, c in enumerate(cols)}
    ratio = v3.margin_ratio_matrix(rows, CAL_M, ci, (len(CAL_M), n))
    el = np.ones((len(CAL_M), n), dtype=bool) if elig is None else elig
    return v3.sig_h9a(ratio, el, CAL_M), cols, ratio


def t07_margin_ratio_matrix_values():
    """券資比 ＝ 融券餘額 ÷ 融資餘額；融資餘額 0 → 不存在（NaN）；日曆外的日期忽略。"""
    rows = [("S00", "2021-01-04", 1000, 50), ("S01", "2021-01-04", 0, 0), ("S02", "2021-01-04", 0, 5),
            ("S00", "2021-01-02", 1000, 999)]  # 週六，不在日曆
    ci = {"S00": 0, "S01": 1, "S02": 2}
    r = v3.margin_ratio_matrix(rows, CAL_M, ci, (len(CAL_M), 3))
    i = int(CAL_M.searchsorted(pd.Timestamp("2021-01-04")))
    expect(r[i, 0] == 0.05 and np.isnan(r[i, 1]) and np.isnan(r[i, 2]), r[i])
    expect(np.isnan(r).sum() == r.size - 1, "只有一個有效格")


def t08_h9a_percentile_and_ties_included():
    """月底：25 檔；1 月 3 檔券資比 0（門檻 0.012 → 選 3 檔，超過 10%）；2 月門檻恰等於並列值 0.05 → 5 檔全選。"""
    jan = margin_rows("2021-01-29", 1000, [0, 0, 0] + [10 * j for j in range(3, 25)])
    feb_shorts = [10, 20, 50, 50, 50] + [10 * j for j in range(6, 26)]   # 60,70,…,250
    feb = margin_rows("2021-02-26", 1000, feb_shorts)
    sig, cols, _ = run_h9(jan + feb)
    got = hits(sig, CAL_M, cols)
    expect({d for d, _ in got} == {"2021-01-29", "2021-02-26"}, f"只在月底：{got}")
    expect({s for d, s in got if d == "2021-01-29"} == {"S00", "S01", "S02"}, got)
    # 2 月排序後第 3 小（索引 2）＝ 第 4 小 ＝ 0.05；位置 2.4 的線性內插 ＝ 0.05
    expect({s for d, s in got if d == "2021-02-26"} == {"S00", "S01", "S02", "S03", "S04"}, got)


def t09_h9a_skips_zero_margin_and_ineligible():
    """3 月：融資餘額 0 的 2 檔不納入母體（有效 23 檔）；不符資格的股票不納入母體也不會被選。"""
    shorts = [0, 5] + [10 * j for j in range(2, 25)]
    margins = [0, 0] + [1000] * 23
    mar = margin_rows("2021-03-31", margins, shorts)
    sig, cols, _ = run_h9(mar)
    got = hits(sig, CAL_M, cols)
    # 有效比率 .02,.03,…,.24（23 個）；位置 2.2 → .04 與 .05 之間 → .042；≤ .042 的是 .02 .03 .04 → S02,S03,S04
    expect({s for _, s in got} == {"S02", "S03", "S04"}, got)
    # 資格：2 月 S00（最低）不合資格 → 母體 24 檔，剩 20,50,50,50,60…；位置 2.3 → 50 ＝ 並列值 → 選 S01(20)、S02～S04(50)
    feb = margin_rows("2021-02-26", 1000, [10, 20, 50, 50, 50] + [10 * j for j in range(6, 26)])
    el = np.ones((len(CAL_M), 25), dtype=bool)
    i = int(CAL_M.searchsorted(pd.Timestamp("2021-02-26")))
    el[i, 0] = False
    sig2, cols2, _ = run_h9(feb, elig=el)
    expect({s for _, s in hits(sig2, CAL_M, cols2)} == {"S01", "S02", "S03", "S04"}, hits(sig2, CAL_M, cols2))


def t10_h9a_small_pool_gives_no_signal():
    """符合資格且有券資比的股票少於 20 檔 → 不排名、不產生訊號（沿用 v1 的 MIN_RANK_POOL）。"""
    rows = margin_rows("2021-01-29", 1000, [10 * j for j in range(19)])
    sig, _, _ = run_h9(rows, n=19)
    expect(not sig.any())


def t11_h9a_no_lookahead():
    """月底訊號只用當日以前的融資券資料：把月底之後的資料全部丟掉，訊號不變。"""
    jan = margin_rows("2021-01-29", 1000, [0, 0, 0] + [10 * j for j in range(3, 25)])
    later = margin_rows("2021-02-26", 1000, [10 * j for j in range(25)])
    full, cols, _ = run_h9(jan + later)
    i = int(CAL_M.searchsorted(pd.Timestamp("2021-01-29")))
    part, _, _ = run_h9(jan)
    expect((part[:i + 1] == full[:i + 1]).all())


# ---------------------------------------------------------------- H10 股利
CAL_D = pd.bdate_range("2021-01-01", "2021-12-31")
COLS_D = ["A", "B", "C", "D", "E", "F", "G"]
CLOSE_D = {"A": 50, "B": 100, "C": 40, "D": 10, "E": 100, "F": 20, "G": 40}


def div_matrices(rows, close=None, cal=CAL_D, cols=COLS_D):
    close = close or CLOSE_D
    RC = np.tile(np.array([close[c] for c in cols], dtype=float), (len(cal), 1))
    ci = {c: i for i, c in enumerate(cols)}
    return v3.dividend_signals(rows, RC, cal, ci, RC.shape), RC


DIV_ROWS = [
    ("A", "2021-03-10", 2.0, 0.5),                                            # 2.5/50 ＝ 5.0% 恰好達標
    ("B", "2021-04-10", 4.9, 0.0),                                            # 週六公告 → 04-12；4.9% 不達標
    ("C", "2021-02-05", 1.0, 0.0), ("C", "2021-08-05", 1.2, 0.0),             # 1.2 ÷ 1.0 ＝ 1.2 倍 恰好達標
    ("C", "2021-11-05", 1.3, 0.0),                                            # 1.3 ÷ 1.2 ＜ 1.2
    ("D", "2021-03-05", 0.0, 0.0), ("D", "2021-09-06", 1.0, 0.0),             # 前次為 0 → 無 H10b；1.0/10 ＝ 10% → H10a
    ("E", "2021-05-05", 3.0, 0.0),                                            # 無前次；3% 不達標
    ("F", "2021-06-07", 0.6, 0.0), ("F", "2021-06-07", 0.5, 0.0),             # 同日兩列加總 1.1 → 5.5%
    ("G", "2020-12-30", 1.0, 0.0), ("G", "2021-06-01", 1.5, 0.0),             # 前次在日曆開始前仍算；1.5 ÷ 1.0
    ("G", "2022-01-05", 9.0, 0.0),                                            # 日曆結束後 → 略過、不報錯
]


def t12_h10a_yield_threshold_and_announcement_day():
    """H10a：現金股利（盈餘＋公積）÷ 訊號日原始收盤 ≥ 5%；週末公告順延到下個交易日；同日多列加總。"""
    (a, b), _ = div_matrices(DIV_ROWS)
    expect(hits(a, CAL_D, COLS_D) == {("2021-03-10", "A"), ("2021-09-06", "D"), ("2021-06-07", "F")}, hits(a, CAL_D, COLS_D))


def t13_h10b_ratio_threshold_and_previous_announcement():
    """H10b：上次現金股利 > 0 且本次 ≥ 1.2 倍；沒有前次、前次為 0、不足 1.2 倍都不算；日曆開始前的前次仍可用。"""
    (a, b), _ = div_matrices(DIV_ROWS)
    expect(hits(b, CAL_D, COLS_D) == {("2021-08-05", "C"), ("2021-06-01", "G")}, hits(b, CAL_D, COLS_D))


def t14_dividend_before_calendar_start_is_skipped():
    """公告日早於日曆起點的公告本身不產生訊號（否則會被錯放在日曆第一天）。"""
    (a, b), _ = div_matrices([("A", "2020-12-30", 5.0, 0.0)])
    expect(not a.any() and not b.any())


def t15_dividend_missing_price_no_h10a():
    """訊號日沒有原始收盤價 → 殖利率不存在 → 不產生 H10a。"""
    rows = [("A", "2021-03-10", 2.0, 0.5)]
    ci = {c: i for i, c in enumerate(COLS_D)}
    RC = np.tile(np.array([CLOSE_D[c] for c in COLS_D], dtype=float), (len(CAL_D), 1))
    RC[:, 0] = np.nan
    a, _ = v3.dividend_signals(rows, RC, CAL_D, ci, RC.shape)
    expect(not a.any())


def t16_dividend_no_lookahead():
    """只用公告日 ≤ 某日的公告重算，該日以前的訊號與全資料一致。"""
    (fa, fb), RC = div_matrices(DIV_ROWS)
    ci = {c: i for i, c in enumerate(COLS_D)}
    for T_date in ("2021-03-10", "2021-06-01", "2021-08-05", "2021-12-31"):
        T = int(CAL_D.searchsorted(pd.Timestamp(T_date)))
        known = [r for r in DIV_ROWS if r[1] <= T_date]
        pa, pb = v3.dividend_signals(known, RC[:T + 1], CAL_D[:T + 1], ci, RC[:T + 1].shape)
        expect((pa == fa[:T + 1]).all() and (pb == fb[:T + 1]).all(), f"{T_date} 前不一致")


def t16b_h1b_requires_strictly_negative_previous_month():
    """H1b：上月年增率恰為 0（不是負）、本月 +10% → 不算由負轉正。"""
    s = {(y, m): 1000 for y in (2018, 2019, 2020) for m in range(1, 13)}
    s[(2020, 5)] = 1100          # 4 月年增 0%，5 月 +10%
    expect(not rev_signals(rev_rows("A", s))["H1b"].any(), "上月 0% 不算負")
    s[(2020, 4)] = 999           # 4 月 −0.1%
    expect(hits(rev_signals(rev_rows("A", s))["H1b"], CAL_R, ["A"]) == {("2020-06-10", "A")}, "上月 −0.1% → H1b")


def t16c_h10a_uses_signal_day_close_not_neighbor_days():
    """H10a 殖利率用『訊號日當天』原始收盤：前一天、後一天的價格不同也不能混用。"""
    rows = [("A", "2021-03-10", 2.5, 0.0), ("B", "2021-03-10", 2.5, 0.0)]
    cols = ["A", "B"]
    ci = {c: i for i, c in enumerate(cols)}
    RC = np.full((len(CAL_D), 2), 50.0)
    i = int(CAL_D.searchsorted(pd.Timestamp("2021-03-10")))
    RC[i - 1, 0], RC[i, 0], RC[i + 1, 0] = 20.0, 100.0, 20.0     # A：當天 100 → 2.5%，不算；前後天 20 → 12.5%
    RC[i - 1, 1], RC[i, 1], RC[i + 1, 1] = 100.0, 20.0, 100.0    # B：當天 20 → 12.5%，算；前後天 100 → 2.5%
    a, _ = v3.dividend_signals(rows, RC, CAL_D, ci, RC.shape)
    expect(hits(a, CAL_D, cols) == {("2021-03-10", "B")}, hits(a, CAL_D, cols))


# ---------------------------------------------------------------- 組態與入口
def t17_config_18_tests_dev_only():
    """18 個檢定：H1a／H1b／H1c／H9a／H10a／H10b 各 20、40、60 日；標準資格、單邊滑價 0.25%、事件門檻 200；期間為開發期。"""
    specs, n, period, min_years = v3.scan_config()
    expect(n == 18 and sum(len(s.horizons) for s in specs) == 18, n)
    expect([s.sid for s in specs] == ["H1a", "H1b", "H1c", "H9a", "H10a", "H10b"], [s.sid for s in specs])
    expect(all(s.horizons == (20, 40, 60) and s.elig == "std" and s.slip == 0.0025 and s.min_events == 200 for s in specs))
    expect(period == s1.holdout.PERIODS["dev"] and min_years == 6, (period, min_years))


def t18_entry_script_refuses_validation_period():
    """入口腳本沒有 --period 選項；硬塞 --period val 或 final 會被拒絕，且不寫任何檔案（在沙盒中執行）。"""
    for extra in (["--period", "val"], ["--period", "final"], ["--val-confirm"]):
        r, files = run_sandboxed("signal_scan_v3.py", *extra)
        expect(r.returncode != 0, f"{extra} 應失敗")
        expect(files == [], f"{extra} 不該留下檔案：{files}")


def t19_entry_script_dev_run_in_sandbox_writes_nothing_real():
    """沙盒（空資料庫）中直接執行會因缺資料表而失敗，且不留下 trials.csv 或報告。"""
    r, files = run_sandboxed("signal_scan_v3.py")
    expect(r.returncode != 0)
    expect(not any(f.endswith(("trials.csv", "scan3_dev_report.txt")) for f in files), files)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t") and k[1:3].isdigit() and callable(v)]


def main():
    lines, passed = [], 0
    for fn in TESTS:
        title = (fn.__doc__ or fn.__name__).strip().splitlines()[0]
        try:
            fn()
            passed += 1
            lines.append(f"[通過] {fn.__name__}：{title}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"[失敗] {fn.__name__}：{title}\n        {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                lines.append("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    text = "\n".join([f"# 訊號掃描 v3 測試報告：{passed}／{len(TESTS)} 通過", *lines])
    out = ROOT / "data" / "experiments" / "signal_scan_v3_tests_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
