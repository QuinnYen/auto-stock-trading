"""訊號掃描（src/backtest/signal_scan.py）的測試：統計函式用已知答案驗證、事件與報酬用人工資料手算驗證、
價格型訊號用『切掉未來資料重算』驗證沒有前視偏誤、安慰劑流程用純雜訊與注入效果檢查大小與檢定力。

用法：python tests/test_signal_scan.py        報告：data/experiments/scan_tests_report.txt
"""
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "backtest"))
import signal_scan as sc  # noqa: E402


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def frame(arr, cal, cols=None):
    arr = np.asarray(arr, dtype=float)
    return pd.DataFrame(arr, index=cal, columns=cols or [f"s{i}" for i in range(arr.shape[1])])


# ---------------------------------------------------------------- 統計函式（已知答案）
def t01_newey_west():
    """lags=0 時等於 平均 ÷ (母體標準差 ÷ √k)；有落後期時與逐項展開的獨立算法一致；正自相關會讓 t 變小。"""
    x = [1, 2, 3, 4, 5, 6]
    expect(close(sc.nw_t(x, 0), 3.5 / (np.std(x) / math.sqrt(6))), f"{sc.nw_t(x, 0)}")
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=40)) * 0.1 + 1.0
    m, e, k, L = y.mean(), y - y.mean(), len(y), 3
    gamma = lambda l: sum(e[i] * e[i - l] for i in range(l, k)) / k  # noqa: E731
    v = gamma(0) + 2 * sum((1 - l / (L + 1)) * gamma(l) for l in range(1, L + 1))
    expect(close(sc.nw_t(y, L), m / math.sqrt(v / k)), "落後期展開不一致")
    expect(abs(sc.nw_t(y, 3)) < abs(sc.nw_t(y, 0)), "正自相關下 t 應變小")
    expect(math.isnan(sc.nw_t([1.0], 1)), "只有 1 個月應回 NaN")


def t02_benjamini_hochberg():
    """已知答案：p ＝ [.01,.04,.03,.005] q ＝ .1 全拒絕；[.01,.04,.2,.5] 只拒絕前兩個；結果依原順序回傳。"""
    expect(sc.bh_reject([0.01, 0.04, 0.03, 0.005], 0.1).tolist() == [True] * 4, "案例 1")
    expect(sc.bh_reject([0.01, 0.04, 0.2, 0.5], 0.1).tolist() == [True, True, False, False], "案例 2")
    expect(sc.bh_reject([0.5, 0.01, 0.2, 0.04], 0.1).tolist() == [False, True, False, True], "原順序")
    expect(not sc.bh_reject([0.3, 0.4, 0.9], 0.1).any(), "全不顯著")
    # BH 的『向上延伸』特性：p(3) 超標但 p(4) 達標時，仍拒絕前 4 個
    expect(sc.bh_reject([0.001, 0.30, 0.012, 0.02], 0.1).tolist() == [True, False, True, True], "向上延伸")


def t03_p_value():
    expect(close(sc.p_greater(0.0), 0.5), "t=0")
    expect(abs(sc.p_greater(1.6448536) - 0.05) < 1e-6, "t=1.645")
    expect(sc.p_greater(float("nan")) == 1.0, "NaN 應回 1")


# ---------------------------------------------------------------- 報酬與資格
def t04_forward_returns():
    """t 日訊號 → O[t+1] 進場、O[t+1+h] 出場；出場日停牌取後續第一個開盤價；已下市以 −100%；仍存活的尾端為 NaN。"""
    n = 30
    cal = pd.bdate_range("2020-01-01", periods=n)
    base = 10.0 + np.arange(n)
    a = base.copy()                                   # 存活
    b = base.copy(); b[8] = np.nan                    # 第 8 天停牌
    c = base.copy(); c[7:] = np.nan                   # 最後價格日 6（已下市）
    O = frame(np.column_stack([a, b, c]), cal)
    r = sc.forward_returns(O, cal, (3,))[3]
    expect(close(r[0, 0], a[4] / a[1] - 1), f"A {r[0, 0]}")
    expect(close(r[4, 1], b[9] / b[5] - 1), f"B 出場日停牌應取下一個價格，得 {r[4, 1]}")
    expect(close(r[2, 2], c[6] / c[3] - 1), "C 出場日仍有價")
    expect(r[3, 2] == -1.0 and r[4, 2] == -1.0 and r[5, 2] == -1.0, f"C 下市應 −100%：{r[3:6, 2]}")
    expect(np.isnan(r[6, 2]) and np.isnan(r[10, 2]), "下市後隔天已無開盤價，無法進場，應為 NaN")
    expect(np.isnan(r[n - 1, 0]) and np.isnan(r[n - 4, 0]), "存活股尾端應為 NaN")


def t05_entry_ok():
    """隔天開盤 < 前收 ×1.095 才買得到；=109.5 買不到；NaN 買不到；最後一天沒有隔天。"""
    C = np.full((4, 3), 100.0)
    O = np.array([[100.0] * 3, [109.4, 109.5, np.nan], [100.0] * 3, [100.0] * 3])
    ok = sc.entry_ok_matrix(O, C)
    expect(ok[0].tolist() == [True, False, False], f"{ok[0]}")
    expect(ok[1].tolist() == [True, True, True] and not ok[3].any(), f"{ok[1]} {ok[3]}")


def t06_benchmark():
    """同日基準＝符合資格且有報酬的股票平均；不足 10 檔為 NaN。"""
    ret = np.tile(np.arange(12, dtype=float) / 100, (3, 1))
    mask = np.ones((3, 12), dtype=bool)
    mask[1, :4] = False                       # 第 2 天只剩 8 檔
    ret[2, 11] = np.nan
    b = sc.benchmark(ret, mask)
    expect(close(b[0], np.mean(np.arange(12) / 100)), f"{b[0]}")
    expect(np.isnan(b[1]), "不足 10 檔應為 NaN")
    expect(close(b[2], np.mean(np.arange(11) / 100)), f"{b[2]}")


# ---------------------------------------------------------------- 訊號
def t07_h2_volume_spike():
    """量 ≥ k × 前 20 日均量（不含當日）、收盤上漲、單日漲幅 < 9.5%。"""
    cal = pd.bdate_range("2020-01-01", periods=30)
    V = np.full((30, 5), 100.0); C = np.full((30, 5), 50.0)
    V[20, :] = 300.0                          # 剛好 3 倍
    C[20, 0], C[20, 1], C[20, 2], C[20, 3] = 51.0, 49.0, 55.0, 51.0   # 上漲、下跌、+10%、上漲；第 5 檔收平盤
    V[20, 3] = 299.0                          # 第 4 檔量不足 3 倍
    s3 = sc.sig_h2(frame(V, cal), frame(C, cal), 3)
    expect(s3[20].tolist() == [True, False, False, False, False], f"3 倍：{s3[20]}")
    s5 = sc.sig_h2(frame(V, cal), frame(C, cal), 5)
    expect(not s5[20].any(), "5 倍門檻不應成立")
    expect(not s3[:20].any(), "平量日不應有訊號")


def t08_monthly_top_decile():
    """每月最後一個交易日，在符合資格的股票中取分數前 10%；母體少於 20 檔不排名；非月底沒有訊號。"""
    cal = pd.bdate_range("2020-01-01", periods=70)
    score = np.tile(np.arange(30, dtype=float), (70, 1))
    elig = np.ones((70, 30), dtype=bool); elig[:, 29] = False      # 分數最高的一檔不符合資格
    s = sc.top_fraction_monthly(score, elig, cal)
    ends = set(sc.month_end_indices(cal).tolist())
    expect(len(ends) >= 3, "至少應有 3 個月")
    for t in range(70):
        picked = np.flatnonzero(s[t]).tolist()
        if t in ends:
            expect(picked == [26, 27, 28], f"t={t}：{picked}")
        else:
            expect(not picked, f"非月底 t={t} 不應有訊號")
    small = sc.top_fraction_monthly(score[:, :10], elig[:, :10], cal)
    expect(not small.any(), "母體不足 20 檔不應排名")


def t09_new_high_250():
    """收盤價 ≥ 前 250 日最高收盤價（不含當日）；樣本不足 200 日不判斷。"""
    cal = pd.bdate_range("2018-01-01", periods=400)
    C = np.arange(400, dtype=float) + 100.0
    C[300] = C[299] - 5                       # 回檔
    C[301] = C[299]                           # 平盤創高（相等算數）
    sig = sc.sig_h4b(frame(C[:, None], cal))
    expect(not sig[:200].any(), "前 200 日不應判斷")
    expect(sig[250, 0] and not sig[300, 0] and sig[301, 0] and sig[302, 0], f"{sig[250, 0]} {sig[300, 0]} {sig[301, 0]} {sig[302, 0]}")


def t10_h8_reversal():
    """5 日累積 ≤ −10% 且當日跌幅未達 −9.5%（未跌停鎖死）。"""
    cal = pd.bdate_range("2020-01-01", periods=20)
    C = np.full((20, 2), 100.0)
    C[10, 0] = 89.0; C[9, 0] = 95.0           # 5 日 −11%，單日 −6.3% → 成立
    C[10, 1] = 89.0; C[9, 1] = 100.0          # 單日 −11% → 鎖死不成立
    C = np.column_stack([C, np.full(20, 100.0)]); C[10, 2] = 93.0     # 第 3 檔只跌 7%，不成立
    s = sc.sig_h8(frame(C, cal))
    expect(s[10].tolist() == [True, False, False], f"{s[10]}")


def t11_h6_first_attention():
    """首次被列注意股：前 30 個交易日內沒有任何注意紀錄；原因含『漲幅』『跌幅』分類，可同時成立。"""
    cal = pd.bdate_range("2019-01-01", periods=200)
    d = lambda i: cal[i].strftime("%Y-%m-%d")  # noqa: E731
    rows = [("A", d(10), "累積收盤價漲幅達40%"), ("A", d(12), "累積收盤價跌幅達30%"),   # 12 在 10 後 2 日內 → 忽略
            ("A", d(50), "累積收盤價跌幅達30%"),                                        # 距上一筆 38 日 → 新一輪
            ("A", d(81), "漲幅 跌幅 都有"), ("A", d(112), "累積收盤價漲幅"),            # 81 距 50 為 31 日、112 距 81 為 31 日 → 新一輪
            ("A", d(142), "累積收盤價跌幅"), ("A", d(173), "累積收盤價漲幅"),            # 142 距 112 剛好 30 日 → 忽略；173 距 142 為 31 日 → 新一輪
            ("B", d(20), "成交量放大")]                                                  # 非漲跌幅原因
    up, down = sc.sig_h6(rows, cal, {"A": 0, "B": 1}, (200, 2))
    expect(np.flatnonzero(up[:, 0]).tolist() == [10, 81, 112, 173], f"漲幅類 {np.flatnonzero(up[:, 0])}")
    expect(np.flatnonzero(down[:, 0]).tolist() == [50, 81], f"跌幅類 {np.flatnonzero(down[:, 0])}")
    expect(not up[:, 1].any() and not down[:, 1].any(), "其他原因兩類都不成立")


def t12_h5_disposition_tail():
    """處置期間倒數第 4 個交易日為訊號日；公告日前 10 日累積報酬 ≤ −10%；期間不足 4 個交易日不成立。"""
    cal = pd.bdate_range("2019-01-01", periods=120)
    d = lambda i: cal[i].strftime("%Y-%m-%d")  # noqa: E731
    C = np.full((120, 3), 100.0)
    C[31:, 0] = 85.0                          # 第 0 檔：公告日(40) 的前 10 日(30) 為 100 → 85（−15%）
    C[31:, 1] = 95.0                          # 第 1 檔：只跌 5%
    C = np.column_stack([C, C[:, 0], C[:, 0]])  # 第 3、4 檔與第 0 檔一樣超跌，只差在處置期間長度
    rows = [("A", d(40), d(41), d(55)), ("B", d(40), d(41), d(55)),   # B 跌得不夠
            ("C", d(40), d(41), d(43)),                                # 只有 3 個交易日 → 不成立
            ("D", d(40), d(41), d(44))]                                # 剛好 4 個交易日 → 訊號日 41
    sig = sc.sig_h5(rows, C, cal, {"A": 0, "B": 1, "C": 3, "D": 4})      # 第 2 欄是沒用到的平盤股
    expect(np.flatnonzero(sig[:, 4]).tolist() == [41], f"D：{np.flatnonzero(sig[:, 4])}（4 個交易日 41～44，倒數第 4 天 41）")
    expect(np.flatnonzero(sig[:, 0]).tolist() == [52], f"A：{np.flatnonzero(sig[:, 0])}（期間最後一天 55，倒數第 4 天 52）")
    expect(not sig[:, 1].any() and not sig[:, 3].any(), "B（跌不夠）、C（期間只有 3 個交易日）不應成立")


def t13_price_signals_no_lookahead():
    """價格型訊號：只用月底以前的資料重算，月底及以前的訊號必須與完整資料相同（H2、H3、H4a、H4b、H8）。"""
    rng = np.random.default_rng(5)
    n, m = 420, 40
    cal = pd.bdate_range("2019-01-01", periods=n)
    C = 50 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (n, m)), axis=0))
    V = rng.integers(1000, 5000, (n, m)).astype(float)
    V[rng.random((n, m)) < 0.02] *= 8
    H = C * 1.01
    elig = np.ones((n, m), dtype=bool)

    def all_signals(c, v, h, cal_, elg):
        Cf, Vf, Hf = frame(c, cal_), frame(v, cal_), frame(h, cal_)
        return [sc.sig_h2(Vf, Cf, 3), sc.sig_h2(Vf, Cf, 5),
                sc.top_fraction_monthly(sc.sig_h3_score(Cf), elg, cal_),
                sc.top_fraction_monthly(sc.sig_h4a_score(Cf, Hf), elg, cal_), sc.sig_h4b(Cf), sc.sig_h8(Cf)]

    full = all_signals(C, V, H, cal, elig)
    T = int(sc.month_end_indices(cal)[-3])          # 取某個真正的月底
    part = all_signals(C[:T + 1], V[:T + 1], H[:T + 1], cal[:T + 1], elig[:T + 1])
    for k, (a, b) in enumerate(zip(full, part)):
        expect(np.array_equal(a[:T + 1], b), f"第 {k} 個訊號在切掉未來資料後改變")
    expect(sum(int(x.sum()) for x in full) > 0, "測試資料應該有訊號")


# ---------------------------------------------------------------- 事件、評估與判斷
def t14_extract_events():
    """資格、買得到、冷卻（只對實際進場的事件計）、期間界線、同日基準的超額報酬。"""
    n, m = 40, 12
    sig = np.zeros((n, m), dtype=bool); elig = np.ones((n, m), dtype=bool); ok = np.ones((n, m), dtype=bool)
    ret = np.full((n, m), 0.02); bench = np.full(n, 0.005)
    sig[[5, 7, 10, 20], 0] = True             # h=3：5 進場、7 忽略(2<4)、10 進場(5≥4)、20 進場
    sig[[5, 7], 1] = True; ok[5, 1] = False   # 第 5 天買不到 → 沒進場，所以第 7 天仍可進場
    sig[12, 2] = True; elig[12, 2] = False    # 不符合資格
    sig[30, 3] = True                         # 在期間之外（hi=25）
    sig[3, 4] = True                          # 在期間之外（lo=4）
    sig[15, 5] = True; ret[15, 5] = np.nan    # 沒有報酬
    sig[16, 6] = True; bench[16] = np.nan     # 沒有同日基準
    sig[[6, 9, 10], 7] = True                 # 邊界：6 進場；9 距 6 為 3 日（= h）忽略；10 距 6 為 4 日（= h+1）保留
    ts, js, r, ex = sc.extract_events(sig, elig, ok, ret, bench, 3, 4, 25)
    got = sorted(zip(ts.tolist(), js.tolist()))
    expect(got == [(5, 0), (6, 7), (7, 1), (10, 0), (10, 7), (20, 0)], f"{got}")
    expect(np.allclose(r, 0.02) and np.allclose(ex, 0.015), "報酬或超額不符")


def t15_evaluate_and_year_stats():
    """3 年、每月 6 個事件、超額約 +1%：事件數、月數、平均、勝率、正超額年數、去掉最佳年皆與手算一致。"""
    cal = pd.bdate_range("2010-01-01", "2012-12-31")
    first = pd.Series(np.arange(len(cal)), index=cal).groupby([cal.year, cal.month]).first().to_numpy()
    ts = np.repeat(first, 6)
    rng = np.random.default_rng(3)
    ex = 0.01 + rng.normal(0, 0.01, len(ts))
    r = ex + 0.004
    res = sc.evaluate(sc.SPECS[0], 10, ts, r, ex, cal)
    expect(res["n"] == 216 and res["months"] == 36, f"{res['n']} {res['months']}")
    cal2 = pd.bdate_range("2010-01-01", "2013-03-31")      # 另加一個只有 4 個事件的月份：要被略過
    ts2 = np.concatenate([ts, np.full(4, int(np.flatnonzero((cal2.year == 2013) & (cal2.month == 2))[0]))])
    r2, ex2 = np.concatenate([r, np.full(4, 0.9)]), np.concatenate([ex, np.full(4, 0.9)])
    res2 = sc.evaluate(sc.SPECS[0], 10, ts2, r2, ex2, cal2)
    expect(res2["months"] == 36 and res2["n"] == 220, f"事件不足 5 個的月份應略過：{res2['months']} 個月")
    expect(close(res["mean_ex"], ex.mean()) and close(res["mean_ret"], r.mean()), "平均不符")
    expect(close(res["net"], r.mean() - sc.cost_rt(0.0025)), "扣成本後報酬不符")
    expect(close(res["win_net"], float((r - sc.cost_rt(0.0025) > 0).mean())), "扣成本後勝率不符")
    expect(res["pos_years"] == 3 and res["t"] > 5, f"{res['pos_years']} {res['t']}")
    years = cal.year.to_numpy()[ts]
    best = max(set(years), key=lambda y: ex[years == y].sum())
    expect(res["best_year"] == best and close(res["excl_best"], ex[years != best].mean()), "去掉最佳年不符")


def t16_verdict_rules():
    """事前約定第 6 節：每一條都能單獨使結論變成放棄，原因文字要對得上。"""
    good = {"n": 500, "min_events": 200, "mean_ex": 0.01, "net": 0.005, "pos_years": 8, "excl_best": 0.004}
    ok, why = sc.verdict(good, True)
    expect(ok and why == "", f"{ok} {why}")
    cases = [({"n": 100}, True, "機會太少"), ({}, False, "沒有明顯贏過隨機"), ({"mean_ex": -0.01}, True, "表現不如隨機"),
             ({"net": -0.001}, True, "扣掉成本"), ({"pos_years": 5}, True, "5 個年度"), ({"excl_best": -0.001}, True, "最好的一年")]
    for ov, rej, text in cases:
        ok, why = sc.verdict({**good, **ov}, rej)
        expect(not ok and text in why, f"{ov} → {ok} {why}")
    ok, _ = sc.verdict({**good, "n": 90, "min_events": 80}, True)
    expect(ok, "H5 的事件門檻應為 80")
    expect(sc.verdict({**good, "n": 200}, True)[0], "剛好 200 個事件應通過")
    expect(sc.verdict({**good, "pos_years": 6}, True)[0], "剛好 6 個正超額年度應通過")


def t17_randomization_test():
    """隨機化檢定：極端值的 p 值公式、隨機結果以 0 為中心、注入 +2% 超額報酬能被偵測、
    用額外抽出的隨機結果當『假的真實結果』時整套流程的誤判率不超標；沒有事件的檢定不會當掉。"""
    rng = np.random.default_rng(11)
    n, m = 1500, 60
    cal = pd.bdate_range("2010-01-01", periods=n)
    ret = rng.normal(0.0, 0.05, (n, m)) + 0.05                  # 大盤整體 +5%：必須扣掉同日基準
    elig = np.arange(m)[None, :] % 2 == np.arange(n)[:, None] % 2  # 每天只有一半股票符合資格
    ret = np.where(elig, ret, 1.0)                              # 不符合資格的股票報酬極端高：隨機股票不可從這裡挑
    ok = np.ones((n, m), dtype=bool)
    bench = sc.benchmark(ret, elig & ok)
    pool = sc.make_pool(ret, elig, ok, bench)
    items = [sc.prepare_test(np.sort(rng.choice(np.arange(300, n - 100), 400, replace=False)), 5, pool, cal)
             for _ in range(8)]
    # 1) 隨機結果以 0 為中心
    nulls = sc.null_stats(items[0], np.random.default_rng(1), 400)
    expect(abs(nulls.mean()) < 4 * nulls.std(ddof=1) / math.sqrt(len(nulls)), f"隨機結果平均 {nulls.mean():.5f} 偏離 0")
    # 2) p 值公式：真實值比所有隨機結果都大 → 1/(n+1)；比所有都小 → 1
    pv, _, _ = sc.randomization_pvalues(items[:2], [1.0, -1.0], np.random.default_rng(2), n_null=300, n_check=20)
    expect(close(pv[0], 1 / 301) and pv[1] == 1.0, f"{pv}")
    # 3) 注入 +2%：p 值達到最小
    it = items[0]
    ret_, bench_, start_, cnt_, cols_ = pool
    js = cols_[start_[it["ts"]] + (rng.random(len(it["ts"])) * cnt_[it["ts"]]).astype(int)]
    r = ret[it["ts"], js] + 0.02
    s_obs = sc.stat_mean(r - bench[it["ts"]], it["inv"], it["cnt"])
    pv, _, _ = sc.randomization_pvalues([it], [s_obs], np.random.default_rng(3), n_null=300, n_check=20)
    expect(close(pv[0], 1 / 301), f"注入 +2% 後 p ＝ {pv[0]}")
    # 4) 假的真實結果（純雜訊）走完整流程，誤判率不超標
    obs = [float(sc.null_stats(i, np.random.default_rng(100 + k), 1)[0]) for k, i in enumerate(items)]
    pv, any_rej, raw = sc.randomization_pvalues(items, obs, np.random.default_rng(4), n_null=300, n_check=150)
    expect(any_rej <= 0.25, f"純雜訊下至少一個通過 BH 的比例 {any_rej:.2f} 過高（理論 ≤ 0.10）")
    expect(abs(raw.mean() - 0.05) < 0.05, f"p<0.05 比例 {raw.mean():.3f}（理論 0.05）")
    expect(bool(sc.bh_reject(pv).sum() <= 2), f"純雜訊的觀察值不應大量通過：{pv}")
    # 5) 沒有事件的檢定
    empty = sc.prepare_test(np.array([], dtype=int), 5, pool, cal)
    pv, _, _ = sc.randomization_pvalues([empty, items[1]], [float("nan"), 0.0], np.random.default_rng(5), n_null=50, n_check=10)
    expect(pv[0] == 1.0, "沒有事件應回 p ＝ 1")


def t21_monthly_statistic():
    """統計量＝各月平均超額報酬的平均（事件少於 5 個的月份略過），不是事件加權平均。"""
    vals = np.array([1.0] * 5 + [0.0] * 20 + [100.0] * 3)
    inv = np.array([0] * 5 + [1] * 20 + [2] * 3)
    cnt = np.array([5, 20, 3])
    expect(close(sc.stat_mean(vals, inv, cnt), 0.5), f"{sc.stat_mean(vals, inv, cnt)}（月平均 1 與 0 的平均 0.5；第 3 個月只有 3 個事件，略過）")
    expect(math.isnan(sc.stat_mean(vals[:3], np.array([0, 0, 0]), np.array([3]))), "沒有任何合格月份應為 NaN")


def t18_score_definitions():
    """H3 分數＝(t−21 日收盤 ÷ t−252 日收盤 − 1)；H4a 分數＝收盤 ÷ 過去 250 日最高價（含當日，至少 200 日）。"""
    cal = pd.bdate_range("2018-01-01", periods=300)
    C = frame((100 + np.arange(300, dtype=float))[:, None], cal)
    H = C * 1.02
    s3 = sc.sig_h3_score(C)
    expect(close(s3[280, 0], C.iloc[259, 0] / C.iloc[28, 0] - 1), f"H3 {s3[280, 0]}")
    expect(np.isnan(s3[251, 0]) and not np.isnan(s3[252, 0]), "H3 需要 252 日歷史")
    s4 = sc.sig_h4a_score(C, H)
    expect(close(s4[280, 0], C.iloc[280, 0] / H.iloc[280, 0]), "H4a 在上升趨勢中應等於收盤÷當日最高")
    expect(np.isnan(s4[150, 0]) and not np.isnan(s4[250, 0]), "H4a 至少 200 日歷史")
    C2 = C.copy(); C2.iloc[220, 0] = 400.0; H2 = C2 * 1.02
    expect(close(sc.sig_h4a_score(C2, H2)[260, 0], C2.iloc[260, 0] / (400.0 * 1.02)), "H4a 應以過去 250 日內最高價為分母")


def t19_conformity_with_preregistration():
    """程式裡的常數與 33 個檢定必須與事前約定一致（成本、資格門檻、檢定清單、持有期、滑價）。"""
    expect(close(sc.FEE_RT, 0.003855) and close(sc.cost_rt(0.0025), 0.008855) and close(sc.cost_rt(0.005), 0.013855), "成本門檻")
    expect((sc.LIQ_MIN, sc.PRICE_CAP, sc.LOCK, sc.BH_Q) == (1e8, 600.0, 0.095, 0.10), "資格或 BH 常數")
    expect((sc.MIN_MONTH_EVENTS, sc.MIN_YEAR_EVENTS, sc.MIN_POS_YEARS) == (5, 10, 6), "月與年的最低事件數、正超額年數")
    expect(sc.DEV == ("2007-01-01", "2016-12-31") and sc.N_TESTS == 33, "期間或檢定數")
    spec = {s.sid: (s.horizons, s.slip, s.elig, s.min_events) for s in sc.SPECS}
    expected = {"H2a": ((5, 10, 20, 40, 60), 0.0025, "std", 200), "H2b": ((5, 10, 20, 40, 60), 0.0025, "std", 200),
                "H3": ((20, 40, 60), 0.0025, "std", 200), "H4a": ((20, 40, 60), 0.0025, "std", 200),
                "H4b": ((5, 10, 20, 40, 60), 0.0025, "std", 200), "H5": ((3, 5, 10), 0.005, "none", 80),
                "H6a": ((5, 10, 20), 0.0025, "no_attn", 200), "H6b": ((5, 10, 20), 0.0025, "no_attn", 200),
                "H8": ((3, 5, 10), 0.0025, "std", 200)}
    expect(spec == expected, f"檢定清單與事前約定不一致：{spec}")
    expect(set(h for s in sc.SPECS for h in s.horizons) <= set(sc.ALL_H), "持有期必須在已計算報酬的清單內")


def t20_eligibility():
    """進場資格：成交金額剛好 1 億通過、少一點不通過；原始價 600 通過、601 不通過；三種資格對注意與處置股的處理。"""
    cal = pd.bdate_range("2020-01-01", periods=30)
    M = frame(np.column_stack([np.full(30, 1e8), np.full(30, 0.99e8), np.full(30, 2e8), np.full(30, 2e8)]), cal)
    RC = frame(np.column_stack([np.full(30, 600.0), np.full(30, 600.0), np.full(30, 601.0), np.full(30, 100.0)]), cal)
    C = frame(np.full((30, 4), 100.0), cal)
    attn = np.zeros((30, 4), dtype=bool); disp = np.zeros((30, 4), dtype=bool)
    base, el = sc.build_eligibility(M, RC, C, attn, disp)
    expect(base[25].tolist() == [True, False, False, True], f"{base[25]}")
    attn[25, 0] = True; disp[25, 3] = True
    base, el = sc.build_eligibility(M, RC, C, attn, disp)
    expect(el["std"][25].tolist() == [False, False, False, False], f"std {el['std'][25]}")
    expect(el["no_attn"][25].tolist() == [True, False, False, False], f"no_attn {el['no_attn'][25]}")
    expect(el["none"][25].tolist() == [True, False, False, True], f"none {el['none'][25]}")


TESTS = [t01_newey_west, t02_benjamini_hochberg, t03_p_value, t04_forward_returns, t05_entry_ok, t06_benchmark,
         t07_h2_volume_spike, t08_monthly_top_decile, t09_new_high_250, t10_h8_reversal, t11_h6_first_attention,
         t12_h5_disposition_tail, t13_price_signals_no_lookahead, t14_extract_events, t15_evaluate_and_year_stats,
         t16_verdict_rules, t17_randomization_test, t21_monthly_statistic, t18_score_definitions,
         t19_conformity_with_preregistration, t20_eligibility]


def main():
    lines, passed = [], 0
    for fn in TESTS:
        title = (fn.__doc__ or fn.__name__).strip().splitlines()[0]
        try:
            fn()
            passed += 1
            lines.append(f"[通過] {fn.__name__}：{title}")
        except Exception as e:  # noqa: BLE001  要回報所有失敗
            lines.append(f"[失敗] {fn.__name__}：{title}\n        {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                lines.append("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    text = "\n".join([f"# 訊號掃描測試報告：{passed}／{len(TESTS)} 通過", *lines])
    out = ROOT / "data" / "experiments" / "scan_tests_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
