"""訊號掃描 v4（心法 K1／K2）與排序測試（rank_test）的測試（依 docs/訊號掃描事前約定 v4.md）。

全部用人工構造資料、手算預期值；不讀真實 market.db。入口腳本一律走 tests/sandbox.py。
用法：python tests/test_signal_scan_v4.py        報告：data/experiments/signal_scan_v4_tests_report.txt
"""
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "backtest"))
sys.path.insert(0, str(ROOT / "tests"))
import account  # noqa: E402
import rank_test as rk  # noqa: E402
import signal_scan as s1  # noqa: E402
import signal_scan_v4 as v4  # noqa: E402
import test_account_engine as eng  # noqa: E402
from sandbox import run_sandboxed  # noqa: E402

expect = eng.expect
N = 100
T = 80            # K1 訊號日
KEY_T = 85        # K2 訊號日（關鍵日 ＝ T）


def base_series():
    """單一股票。第 0～79 天：收盤交替 100／110（前 60 日區間 ＝ 10%）、量 1000；第 80 天：關鍵日（K1 成立）。
    回傳 dict(O, H, L, C, V)，皆為 (N, 1) 的陣列。"""
    C = np.where(np.arange(N) % 2 == 0, 100.0, 110.0)
    O, V = C.copy(), np.full(N, 1000.0)
    H, L = C + 1.0, C - 1.0
    C[T], O[T], V[T], H[T], L[T] = 125.0, 118.0, 2500.0, 126.0, 117.0     # 創新高、量 2.5 倍、紅 K 實體 (125−118)÷118 ≈ 5.93%
    return {"O": O[:, None], "H": H[:, None], "L": L[:, None], "C": C[:, None], "V": V[:, None]}


def k1(a):
    return v4.k1_signal(a["O"], a["C"], a["V"])


def on(sig):
    return sorted(np.flatnonzero(sig[:, 0]).tolist())


def t01_k1_base_case():
    """基準案例：第 80 天同時滿足 4 個條件 → 只有這一天有訊號（前 60 日區間 10% ≤ 20%、收盤 125 > 前 60 日最高 110、量 2500 ＝ 前 20 日均量 1000 的 2.5 倍、實體 5.93%）。"""
    expect(on(k1(base_series())) == [T], on(k1(base_series())))


def t02_k1_each_condition_boundary():
    """每個條件各自破壞一次只讓訊號消失：壓縮 20% 剛好成立／20.1% 不成立；收盤等於前高不成立；量 2499 不成立；實體 4.9% 不成立。"""
    a = base_series()
    a["C"][:T:2] = 100.0
    a["C"][1:T:2] = 120.0                                 # 區間 ＝ 120÷100−1 ＝ 20%（前 60 日內）
    a["O"][:T] = a["C"][:T]
    a["C"][T], a["O"][T] = 125.0, 118.0
    expect(on(k1(a)) == [T], "壓縮恰為 20% 應成立")
    b = {k: v.copy() for k, v in a.items()}
    b["C"][T - 2] = 120.1                                   # 前 60 日內最高 120.1 → 區間 20.1%
    expect(on(k1(b)) == [], "壓縮 20.1% 不應成立")
    c = {k: v.copy() for k, v in base_series().items()}
    c["C"][T] = 110.0
    c["O"][T] = 100.0
    expect(on(k1(c)) == [], "收盤等於前 60 日最高（110）不是新高")
    d = {k: v.copy() for k, v in base_series().items()}
    d["V"][T] = 2499.0
    expect(on(k1(d)) == [], "量 2499 ＜ 2.5 倍")
    e = {k: v.copy() for k, v in base_series().items()}
    e["O"][T] = 125.0 / 1.049
    expect(on(k1(e)) == [], "實體 4.9% ＜ 5%")
    f = {k: v.copy() for k, v in base_series().items()}
    f["O"][T] = 125.0 / 1.05
    expect(on(k1(f)) == [T], "實體恰為 5% 成立")


def t03_k1_windows_exclude_today_and_need_history():
    """前 60 日與前 20 日都不含今天：今天的 125 若被算進區間，區間會是 25% 而不成立（基準案例成立即證明沒有算進去）；歷史不足 60 日時不產生訊號。"""
    a = base_series()
    expect(on(k1(a)) == [T], "含今天會讓壓縮失敗，基準案例卻成立，表示不含今天")
    # 歷史不足：只有 50 天資料的股票，即使最後一天條件齊備也不成立
    short = {k: v[: 51].copy() for k, v in base_series().items()}
    for k, v in (("C", 125.0), ("O", 118.0), ("V", 2500.0), ("H", 126.0), ("L", 117.0)):
        short[k][50] = v
    expect(on(k1(short)) == [], "不足 60 日不應有訊號")


def k2(a):
    return v4.k2_signal(k1(a), a["H"], a["L"], a["C"], a["V"])


def k2_case():
    """K2 基準：關鍵日 80（C 125、L 117、V 2500）；第 81～85 天收盤 125、124、123、122、121；第 85 天量 875、H 122、L 119。"""
    a = base_series()
    a["C"][81:86, 0] = [125.0, 124.0, 123.0, 122.0, 121.0]
    a["O"][81:86, 0] = a["C"][81:86, 0]
    a["H"][81:86, 0] = a["C"][81:86, 0] + 0.5
    a["L"][81:86, 0] = a["C"][81:86, 0] - 0.5
    a["V"][81:86, 0] = 900.0
    a["V"][KEY_T, 0], a["H"][KEY_T, 0], a["L"][KEY_T, 0] = 875.0, 122.0, 119.0
    return a


def t04_k2_base_case():
    """基準案例：第 85 天（關鍵日 80 的 5 天後）：3 日收盤 123、122、121 都 ≥ 關鍵日低點 117 且不上漲；量 875 ＝ 2500×35%；振幅 3÷122 ≈ 2.46% ＜ 3%；5 日均線 123、收盤 121 差 1.63% ≤ 2% → 成立。"""
    expect(on(k2(k2_case())) == [KEY_T], on(k2(k2_case())))


def t05_k2_each_condition_alone():
    """逐一破壞：量 876 不成立；收盤跌破關鍵日低點不成立；某日收盤上漲不成立；振幅 3.7÷122 超過 3% 不成立；偏離 5 日線 ＞ 2% 不成立。"""
    def mod(**kw):
        a = k2_case()
        for key, (idx, val) in kw.items():
            a[key][idx, 0] = val
        return a
    expect(on(k2(mod(V=(KEY_T, 876.0)))) == [], "量 876 ＞ 875")
    expect(on(k2(mod(V=(KEY_T, 875.0)))) == [KEY_T], "量恰 875 成立")
    expect(on(k2(mod(L=(T, 121.5)))) == [], "關鍵日低點 121.5 ＞ 收盤 121")
    expect(on(k2(mod(C=(KEY_T, 122.5), H=(KEY_T, 123.0), L=(KEY_T, 120.0)))) == [], "第 85 天收盤 122.5 ＞ 前一日 122（上漲）")
    expect(on(k2(mod(H=(KEY_T, 122.7), L=(KEY_T, 119.0)))) == [], "振幅 3.7÷122 ≈ 3.03% 不成立")
    expect(on(k2(mod(H=(KEY_T, 122.6), L=(KEY_T, 119.0)))) == [KEY_T], "振幅 3.6÷122 ≈ 2.95% 成立")
    expect(on(k2(mod(C=(81, 140.0)))) == [], "5 日均線 126、收盤 121 差 3.97% ＞ 2%")


def t05b_k1_window_is_full_60_days_for_min():
    """前 60 日的最低價要看滿 60 天：視窗最早那段（第 20～29 天）若有 85，區間 ＝ 110÷85−1 ≈ 29% ＞ 20% → 不成立；放在視窗之外（第 19 天）則不影響。"""
    a = base_series()
    a["C"][22, 0] = 85.0
    expect(on(k1(a)) == [], "第 22 天（視窗內，較早段）的低點必須被看到")
    b = base_series()
    b["C"][19, 0] = 85.0
    expect(on(k1(b)) == [T], "第 19 天已在前 60 日之外")


def t05c_k2_extra_boundaries():
    """K2 補邊界：第 3 日（t−2 相對 t−3）收盤上漲不成立；偏離 5 日線 2.5% 不成立；振幅分母是『前一日收盤』（3.65÷122 ＝ 2.99% 成立，若用今日收盤 121 則 3.02% 不成立）。"""
    a = k2_case()
    a["C"][82, 0] = 122.5                      # t−3 ＝ 122.5 ＜ t−2 ＝ 123：t−2 當天收盤上漲
    expect(on(k2(a)) == [], "t−2 相對 t−3 上漲不應成立")
    b = k2_case()
    b["C"][81, 0] = 130.5                      # 5 日均線 ＝ 124.1，收盤 121 偏離 2.5%
    expect(on(k2(b)) == [], "偏離 2.5% ＞ 2%")
    c = k2_case()
    c["H"][KEY_T, 0], c["L"][KEY_T, 0] = 122.65, 119.0
    expect(on(k2(c)) == [KEY_T], "振幅 3.65÷前一日收盤 122 ＝ 2.99% 應成立")


def t06_k2_key_day_fixed_at_five_days_and_must_be_k1():
    """關鍵日固定為 5 天前：訊號只出現在第 85 天，不會出現在第 84、86 天；關鍵日不滿足 K1（量不足）時整個 K2 消失。"""
    s = k2(k2_case())
    expect(not s[84, 0] and not s[86, 0], "只在關鍵日後第 5 天")
    a = k2_case()
    a["V"][T, 0] = 2000.0
    expect(on(k1(a)) == [] and on(k2(a)) == [], "關鍵日不是 K1 則無 K2")


def t07_no_lookahead():
    """切掉某日之後的資料重算，該日以前（含該日）的 K1、K2 訊號與全資料一致。"""
    a = k2_case()
    full1, full2 = k1(a), k2(a)
    for cut in (T, 83, KEY_T, 90):
        part = {k: v[: cut + 1].copy() for k, v in a.items()}
        p1 = v4.k1_signal(part["O"], part["C"], part["V"])
        p2 = v4.k2_signal(p1, part["H"], part["L"], part["C"], part["V"])
        expect((p1 == full1[: cut + 1]).all() and (p2 == full2[: cut + 1]).all(), f"切在 {cut} 不一致")


def t08_config():
    """K1、K2 各持有 10／20／40／60 日 ＝ 8 個檢定；標準資格、單邊滑價 0.25%、事件門檻 200；期間為開發期。"""
    specs, n, period, min_years = v4.scan_config()
    expect(n == 8 and [s.sid for s in specs] == ["K1", "K2"], (n, [s.sid for s in specs]))
    expect(all(s.horizons == (10, 20, 40, 60) and s.elig == "std" and s.slip == 0.0025 and s.min_events == 200 for s in specs), "規格")
    expect(period == s1.holdout.PERIODS["dev"] and min_years == 6, (period, min_years))


# ---------------------------------------------------------------- 排序
def t09_rank_scores_hand_computed():
    """R1 ＝ −(近 20 日最高價 ÷ 近 20 日最低價，含今天)；R2 ＝ 近 5 日報酬 − 近 20 日報酬。人工序列手算。"""
    n = 30
    C = np.linspace(100, 129, n)[:, None]          # 100, 101, …, 129
    H, L = C + 2.0, C - 2.0
    sc = rk.rank_scores(H, L, C)
    t = 25                                          # 近 20 日 ＝ 第 6～25 天：最高 H ＝ 125+2 ＝ 127，最低 L ＝ 106−2 ＝ 104
    expect(abs(sc["R1"][t, 0] - (-127 / 104)) < 1e-12, sc["R1"][t, 0])
    r5, r20 = 125 / 120 - 1, 125 / 105 - 1          # C[25]＝125，C[20]＝120，C[5]＝105
    expect(abs(sc["R2"][t, 0] - (r5 - r20)) < 1e-12, sc["R2"][t, 0])
    expect(np.isnan(sc["R1"][18, 0]) and np.isnan(sc["R2"][18, 0]), "歷史不足 20 日應為 NaN")
    # 較緊實者分數較高：兩檔股票同日，A 區間 10%、B 區間 30%
    Hh = np.array([[110.0, 130.0]] * 20); Ll = np.array([[100.0, 100.0]] * 20); Cc = np.array([[105.0, 115.0]] * 20)
    s2 = rk.rank_scores(Hh, Ll, Cc)["R1"][-1]
    expect(s2[0] > s2[1], f"{s2}")


def t10_improves_boundary():
    """『有改善』＝ 期末淨值高於 20 個隨機種子期末淨值由小到大的第 18 名（隨機結果的前 10%）；恰等於第 18 名不算。"""
    rnd = np.arange(1, 21) * 1000.0
    expect(not rk.improves(18000.0, rnd) and rk.improves(18000.5, rnd), "第 18 名邊界")
    expect(not rk.improves(5000.0, rnd) and rk.improves(25000.0, rnd), "遠低／遠高")
    expect(rk.improves(19000.0, rnd[::-1]), "與順序無關")


def t11_ranked_selection_is_deterministic_top_scores():
    """6 檔同日訊號、3 個空位：帶入分數排序後買進的是分數最高的 3 檔，且不受種子影響；分數 NaN 的排在最後。"""
    a = eng.mk(n_st=6)
    m = account.Market(**a)
    shape = a["O"].shape
    entry = np.zeros(shape, dtype=bool)
    entry[3, :] = True
    elig = np.ones(shape, dtype=bool)
    score = np.tile(np.array([1.0, 5.0, np.nan, 4.0, 2.0, 9.0]), (shape[0], 1))
    r = rk.run_ranked(m, entry, score, elig, __import__("account_sim_v2").make_cfg2(5, "S2"), 1, shape[0])
    expect(sorted(t["stock"] for t in r.trades) == [1, 3, 5], f"{sorted(t['stock'] for t in r.trades)}")
    r2 = rk.run_ranked(m, entry, score, elig, __import__("account_sim_v2").make_cfg2(5, "S2"), 1, shape[0])
    expect(np.array_equal(r.equity, r2.equity), "確定性")


def t12_entry_scripts_dev_only_in_sandbox():
    """兩個入口腳本都沒有期間選項：--period val／final、--val-confirm、--final-confirm 被拒絕且不寫東西；沙盒空資料庫直接跑失敗，不留 trials／報告／帳本，真實帳本與 trials.csv 位元組不變。"""
    import holdout
    real, trials = holdout.LEDGER, holdout.LEDGER.parent / "trials.csv"
    before = (real.read_bytes(), trials.read_bytes())
    for script in ("signal_scan_v4.py", "rank_test.py"):
        for extra in (["--period", "val"], ["--period", "final"], ["--val-confirm"], ["--final-confirm"]):
            r, files = run_sandboxed(script, *extra)
            expect(r.returncode != 0 and files == [], f"{script} {extra}: rc={r.returncode} files={files}")
        r, files = run_sandboxed(script)
        expect(r.returncode != 0 and "no such table" in r.stderr, f"{script}: {r.stderr[-200:]}")
        expect(not any(f.endswith(("trials.csv", "scan4_dev_report.txt", "rank_dev_report.txt", "holdout_access.log")) for f in files), f"{script}: {files}")
    expect((real.read_bytes(), trials.read_bytes()) == before, "真實帳本或 trials.csv 被改動了")


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
    text = "\n".join([f"# 訊號掃描 v4 與排序測試報告：{passed}／{len(TESTS)} 通過", *lines])
    out = ROOT / "data" / "experiments" / "signal_scan_v4_tests_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
