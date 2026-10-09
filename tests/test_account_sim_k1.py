"""帳戶模擬 K1 執行器（src/backtest/account_sim_k1.py，依 docs/帳戶模擬事前約定 v5（K1）.md）的測試。

涵蓋：4 個組合與偏好順序、7 條通過條件與三關門檻（含滑價壓力測試邊界）、K1 進場訊號接線、組合執行流程（持有期、對照、敏感度）、
三關的前後順序鎖（結果檔不存在／未通過／組合不符／已執行過一律拒絕）、入口腳本在沙盒中的防護。

用法：python tests/test_account_sim_k1.py        報告：data/experiments/account_sim_k1_tests_report.txt
"""
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "backtest"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "tests"))
import account  # noqa: E402
import account_sim as sim  # noqa: E402
import account_sim_k1 as k1  # noqa: E402
import account_sim_v2 as v2  # noqa: E402
import test_account_engine as eng  # noqa: E402
import test_signal_scan_v4 as t4  # noqa: E402
from sandbox import run_sandboxed  # noqa: E402

expect = eng.expect
GOOD_YEARS = [0.1, 0.05, -0.02, 0.03, 0.04, 0.02, -0.01, 0.06, 0.03, 0.02]
VAL_YEARS = [0.2, 0.05, -0.02, 0.03, 0.04]
CTRL, STRESS_OK = np.full(20, 20000.0), np.full(20, 21000.0)


def verdicts(finals, years, mdd, ctrl=CTRL, stress=STRESS_OK, stage="dev", n_days=None):
    return [ok for _, ok in k1.judge_k1(finals, ctrl, years, mdd, stress_finals=stress, stage=stage, n_days=n_days)]


def t01_combos_and_preference():
    """4 個組合：持有 10／20 天 × S2／S4；偏好順序 S2 ＞ S4，同一結構下 h＝20 ＞ h＝10；沒有通過者回傳 None。"""
    expect([c.sid for c in k1.COMBOS] == ["K1-10-S2", "K1-10-S4", "K1-20-S2", "K1-20-S4"], [c.sid for c in k1.COMBOS])
    expect([c.h for c in k1.COMBOS] == [10, 10, 20, 20], "持有期")
    expect(k1.pick_preferred(["K1-10-S2", "K1-20-S2"]) == "K1-20-S2", "S2 內 h＝20 優先")
    expect(k1.pick_preferred(["K1-20-S4", "K1-10-S2"]) == "K1-10-S2", "S2 優先於 S4，即使 h 較短")
    expect(k1.pick_preferred(["K1-10-S4", "K1-20-S4"]) == "K1-20-S4", "S4 內 h＝20 優先")
    expect(k1.pick_preferred(["K1-10-S4", "K1-10-S2", "K1-20-S4", "K1-20-S2"]) == "K1-20-S2", "全部通過")
    expect(k1.pick_preferred([]) is None, "無通過者")


def t02_judge_dev_each_condition_alone():
    """開發期 7 條：基準全成立；每次只破壞一條，只有該條為否（期末 32,578、15 次賺錢、高於對照、6 年為正、回檔 40%、去掉最佳年、滑價 0.5% 中位數 20,000）。"""
    good = np.full(20, 40000.0)
    expect(verdicts(good, GOOD_YEARS, 0.3) == [True] * 7, f"{verdicts(good, GOOD_YEARS, 0.3)}")
    expect(verdicts(np.full(20, 30000.0), GOOD_YEARS, 0.3, ctrl=np.full(20, 20000.0)) == [False] + [True] * 6, "條件1")
    expect(verdicts(np.array([19000.0] * 6 + [40000.0] * 14), GOOD_YEARS, 0.3)[1:2] == [False], "條件2")
    expect(verdicts(good, GOOD_YEARS, 0.3, ctrl=np.full(20, 40000.0))[2] is False, "條件3")
    expect(verdicts(good, [0.1, 0.05, 0.03, 0.04, 0.02, -0.01, -0.01, -0.01, -0.01, -0.01], 0.3)[3] is False, "條件4")
    expect(verdicts(good, GOOD_YEARS, 0.45)[4] is False, "條件5")
    expect(verdicts(good, [2.0, 0.01, 0.01, 0.01, 0.01, 0.01, -0.3, -0.3, -0.3, -0.3], 0.3)[5] is False, "條件6")
    expect(verdicts(good, GOOD_YEARS, 0.3, stress=np.full(20, 15000.0))[6] is False, "條件7")


def t03_judge_boundaries():
    """邊界：期末中位數剛好 32,578.x 成立、少 1 元不成立；15 次賺錢成立；回檔 40% 成立、40.01% 不成立；滑價 0.5% 中位數剛好 20,000 成立、19,999 不成立；對照相等不成立。"""
    need = k1.MIN_FINAL_DEV
    expect(abs(need - 20000 * 1.05 ** 10) < 1e-9 and round(need) == 32578, need)
    expect(verdicts(np.full(20, need), GOOD_YEARS, 0.3)[0] and not verdicts(np.full(20, need - 1), GOOD_YEARS, 0.3)[0], "期末邊界")
    expect(verdicts(np.array([19000.0] * 5 + [40000.0] * 15), GOOD_YEARS, 0.3)[1], "15 次")
    good = np.full(20, 40000.0)
    expect(verdicts(good, GOOD_YEARS, 0.40)[4] and not verdicts(good, GOOD_YEARS, 0.4001)[4], "回檔邊界")
    expect(verdicts(good, GOOD_YEARS, 0.3, stress=np.full(20, 20000.0))[6] and not verdicts(good, GOOD_YEARS, 0.3, stress=np.full(20, 19999.0))[6], "壓力測試邊界")
    expect(not verdicts(good, GOOD_YEARS, 0.3, ctrl=np.full(20, 40000.0))[2], "對照相等不成立")
    mixed = np.array([10000.0] * 10 + [30000.0] * 10)      # 中位數取均值 20,000
    expect(verdicts(good, GOOD_YEARS, 0.3, stress=mixed)[6], "壓力測試用中位數（均值 20,000 邊界）")


def t04_judge_val_and_final_thresholds():
    """驗證期：期末 ≥ 25,526、5 年中 ≥3 年為正；最後測試期：期末 ≥ 20000×1.05^(N÷245)、≥3 年為正；兩者都含滑價壓力測試與回檔 40%。"""
    expect(round(k1.MIN_FINAL_VAL) == 25526, k1.MIN_FINAL_VAL)
    v = lambda f, y=VAL_YEARS, **kw: verdicts(f, y, 0.3, stage="val", **kw)  # noqa: E731
    expect(v(np.full(20, k1.MIN_FINAL_VAL))[0] and not v(np.full(20, k1.MIN_FINAL_VAL - 1))[0], "驗證期期末邊界")
    expect(v(np.full(20, 30000.0), [0.1, 0.1, 0.1, -0.1, -0.1])[3] and not v(np.full(20, 30000.0), [0.1, 0.1, -0.1, -0.1, -0.1])[3], "3 年為正")
    expect(not v(np.full(20, 30000.0), stress=np.full(20, 19999.0))[6], "驗證期壓力測試")
    for n in (980, 1170):
        need = 20000 * 1.05 ** (n / 245)
        f = lambda x, n=n: verdicts(np.full(20, x), VAL_YEARS, 0.3, stage="final", n_days=n)  # noqa: E731
        expect(f(need)[0] and not f(need - 1)[0], f"最後測試期 N={n} 邊界")
    expect(verdicts(np.full(20, 30000.0), VAL_YEARS, 0.4001, stage="final", n_days=1170)[4] is False, "最後測試期回檔 40%")


def t05_k1_entry_wiring():
    """進場訊號接線：由還原後寬表（O、C、V）算出 K1；基準案例只有第 80 天有訊號，且另一檔沒有訊號的股票不受影響。"""
    a = t4.base_series()
    idx = pd.RangeIndex(t4.N)
    d = {k: pd.DataFrame(np.hstack([a[k], np.full_like(a[k], 100.0)]), index=idx, columns=["1111", "2222"]) for k in ("O", "C", "V")}
    sig = k1.k1_entry(d)
    expect(np.flatnonzero(sig[:, 0]).tolist() == [t4.T] and not sig[:, 1].any(), f"{np.flatnonzero(sig[:, 0])}")


def synth(n_st=3, n_days=40):
    a = eng.mk(n_st=n_st, n_days=n_days)
    return account.Market(**a), a["O"].shape


def t06_run_combo_wiring():
    """組合執行：持有期 ＝ 組合的 h（10 天組合的交易 10 天後賣、20 天組合 20 天後賣）；對照依（h、結構）快取；敏感度各用對應的滑價／停損；滑價 0.5% 結果可取出做壓力測試。"""
    a = eng.mk(n_st=3, n_days=40)
    eng.set_all(a, 6, 1, 80)             # 進場後 1 號股票跌 20%：單檔停損 15% 當天收盤標記、隔天以 80 賣出
    eng.set_all(a, 9, 1, 60)             # 之後繼續跌到 60：不設停損者續抱到持有期滿，會賣在 60，結果與照規則不同；0 號股票平盤，用來檢查持有期
    m, shape = account.Market(**a), a["O"].shape
    entry = np.zeros(shape, dtype=bool)
    entry[3, :2] = True
    elig = np.ones(shape, dtype=bool)
    ctx = {"mkt": m, "std": elig, "entries": {"K1": entry}}
    controls, seeds = {}, range(1, 4)
    c10, c20 = k1.COMBOS[0], k1.COMBOS[2]
    r10, ctrl10, extra = k1.run_combo(ctx, c10, controls, 1, shape[0], seeds)
    r20, ctrl20, _ = k1.run_combo(ctx, c20, controls, 1, shape[0], seeds)
    hold = lambda res: {t["exit_idx"] - t["entry_idx"] for t in res[0].trades if t["stock"] == 0}  # noqa: E731
    expect(hold(r10) == {10}, f"10 天組合：{hold(r10)}")
    expect(hold(r20) == {20}, f"20 天組合：{hold(r20)}")
    expect(ctrl10 is not ctrl20 and set(controls) == {(10, "S2"), (20, "S2")}, f"對照依（h、結構）快取：{set(controls)}")
    _, again, _ = k1.run_combo(ctx, c10, controls, 1, shape[0], seeds)
    expect(again is ctrl10, "同一（h、結構）重用對照")
    c10b = k1.COMBOS[1]
    _, ctrl10b, _ = k1.run_combo(ctx, c10b, controls, 1, shape[0], seeds)
    expect(ctrl10b is not ctrl10 and (10, "S4") in controls, "S4 有自己的對照")
    expect(list(extra) == ["滑價 0%", "滑價 0.5%", "不設單檔停損"], list(extra))
    for key, kw in (("滑價 0%", {"slip": 0.0}), ("滑價 0.5%", {"slip": 0.005}), ("不設單檔停損", {"stop_pct": 1.0})):
        want = sim.finals_of(sim.run_seeds(m, entry, elig, v2.make_cfg2(10, "S2", **kw), 1, shape[0], seeds))
        expect(np.array_equal(sim.finals_of(extra[key]), want), key)
    expect(not np.array_equal(sim.finals_of(extra["滑價 0%"]), sim.finals_of(extra["滑價 0.5%"])), "兩種滑價結果要不同")
    expect(not np.array_equal(sim.finals_of(extra["不設單檔停損"]), sim.finals_of(r10)), "不設單檔停損要和照規則不同")
    want_ctrl = sim.finals_of(sim.run_control(m, elig, v2.make_cfg2(10, "S2"), 1, shape[0], seeds))
    expect(np.array_equal(sim.finals_of(ctrl10), want_ctrl), "對照 ＝ 全部合格股票隨機進場（對應持有期）")


def t07_summarize_has_seven_conditions():
    """整理一個組合：標題含編號；條件 7 條（含滑價 0.5% 壓力測試）；回傳 passed／stat。"""
    m, shape = synth(6)
    entry = np.zeros(shape, dtype=bool)
    entry[3, :] = True
    elig = np.ones(shape, dtype=bool)
    ctx = {"mkt": m, "std": elig, "entries": {"K1": entry}}
    combo = k1.COMBOS[1]
    res, ctrl, extra = k1.run_combo(ctx, combo, {}, 1, shape[0], range(1, 21))
    cal = pd.date_range("2020-12-01", periods=shape[0], freq="D")
    lines, passed, stat = k1.summarize_combo(combo, res, ctrl, extra, cal, 1, stage="dev")
    text = "\n".join(lines)
    expect("K1-10-S4" in lines[0] and text.count("[✓]") + text.count("[✗]") == 7 and "滑價 0.5% 時" in text and "≤ 40%" in text, text)
    expect(isinstance(passed, bool) and abs(stat["final"] - np.median(sim.finals_of(res))) < 1e-9, "stat")
    stress_line = [ln for ln in lines if "滑價 0.5% 時" in ln][0]
    expect(f"{np.median(sim.finals_of(extra['滑價 0.5%'])):,.0f}" in stress_line, f"壓力測試應使用滑價 0.5% 的結果：{stress_line}")
    expect(f"{np.median(sim.finals_of(extra['滑價 0%'])):,.0f}" not in stress_line, "不應使用滑價 0% 的結果")


def t08_gate_dev_to_val_to_final():
    """三關順序鎖：驗證期需有開發期結果檔且組合 ＝ 依偏好選出者、且尚未跑過；最後測試期需驗證期結果檔通過且組合相同、且尚未跑過；否則一律拒絕。"""
    def refuses(stage, combo, d):
        try:
            k1.check_gate(stage, combo, Path(d))
        except SystemExit as e:
            return str(e)
        return None

    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        expect(refuses("dev", None, d) is None, "開發期不需前置")
        expect("開發期" in (refuses("val", "K1-20-S2", d) or ""), "沒有開發期結果檔")
        k1.save_result("k1_dev_result.json", {"passed": [], "chosen": None}, dd)
        expect("沒有" in (refuses("val", "K1-20-S2", d) or "") or "未" in (refuses("val", "K1-20-S2", d) or ""), "開發期沒有通過者")
        k1.save_result("k1_dev_result.json", {"passed": ["K1-20-S2", "K1-10-S2"], "chosen": "K1-20-S2"}, dd)
        expect(refuses("val", "K1-20-S2", d) is None, "選出的組合可進驗證期")
        expect("K1-20-S2" in (refuses("val", "K1-10-S2", d) or ""), "不是選出的組合不能進驗證期")
        expect("驗證期" in (refuses("final", "K1-20-S2", d) or ""), "沒有驗證期結果檔不能進最後測試期")
        k1.save_result("k1_val_result.json", {"combo": "K1-20-S2", "passed": False}, dd)
        expect("驗證期" in (refuses("final", "K1-20-S2", d) or "") and "已執行" in (refuses("val", "K1-20-S2", d) or ""), "驗證期未通過／已跑過")
        k1.save_result("k1_val_result.json", {"combo": "K1-20-S2", "passed": True}, dd)
        expect(refuses("final", "K1-20-S2", d) is None, "驗證期通過後可進最後測試期")
        expect(refuses("final", "K1-10-S2", d) is not None, "組合不符不能進最後測試期")
        k1.save_result("k1_final_result.json", {"combo": "K1-20-S2", "passed": False}, dd)
        expect("已執行" in (refuses("final", "K1-20-S2", d) or ""), "最後測試期只能跑一次")


def t08b_result_contents():
    """結果檔內容：開發期寫出通過者與依偏好順序選出者（不是第一個通過者）；驗證期／最後測試期寫出組合與是否通過（布林）。"""
    r = k1.dev_result(["K1-10-S4", "K1-10-S2", "K1-20-S4"])
    expect(r == {"passed": ["K1-10-S4", "K1-10-S2", "K1-20-S4"], "chosen": "K1-10-S2"}, r)
    expect(k1.dev_result([]) == {"passed": [], "chosen": None}, "無通過者")
    expect(k1.stage_result("K1-20-S2", True) == {"combo": "K1-20-S2", "passed": True} and k1.stage_result("K1-20-S2", 0)["passed"] is False, "階段結果")


def t09_result_files_roundtrip():
    """結果檔：寫入後讀回相同；檔案不存在時讀回 None。"""
    with tempfile.TemporaryDirectory() as d:
        expect(k1.load_result("x.json", Path(d)) is None, "不存在回傳 None")
        k1.save_result("x.json", {"a": [1, 2], "b": None}, Path(d))
        expect(k1.load_result("x.json", Path(d)) == {"a": [1, 2], "b": None}, "讀回")
        expect(json.loads((Path(d) / "x.json").read_text(encoding="utf-8")) == {"a": [1, 2], "b": None}, "檔案內容是 JSON")


def t10_stage_config():
    """各關設定：開發期 4 個組合、標記 acct5-dev；驗證期標記 acct5-val、報告 acct5_val_report.txt；最後測試期標記 acct5-final、報告 acct5_final_report.txt；期間依序為開發、驗證、最後測試。"""
    dev, val, fin = k1.stage_config("dev"), k1.stage_config("val"), k1.stage_config("final")
    expect(dev["tag"] == "acct5-dev" and dev["report"] == "acct5_dev_report.txt" and dev["period"] == ("2007-01-01", "2016-12-31"), f"{dev}")
    expect(val["tag"] == "acct5-val" and val["report"] == "acct5_val_report.txt" and val["period"] == ("2017-01-01", "2021-12-31"), f"{val}")
    expect(fin["tag"] == "acct5-final" and fin["report"] == "acct5_final_report.txt" and fin["period"][0] == "2022-01-01", f"{fin}")


def t11_entry_script_guards_in_sandbox():
    """入口腳本：驗證期缺 --combo 被拒絕；有確認旗標但沒有開發期結果檔、或最後測試期沒有驗證期結果檔，都在寫入帳本之前被拒絕（沙盒內沒有任何檔案）；未知組合被參數檢查拒絕；預設開發期在空資料庫失敗且不留紀錄；真實帳本與 trials.csv 位元組不變。"""
    import holdout
    real, trials = holdout.LEDGER, holdout.LEDGER.parent / "trials.csv"
    before = (real.read_bytes(), trials.read_bytes())
    r, files = run_sandboxed("account_sim_k1.py", "--period", "val", "--val-confirm")
    expect(r.returncode != 0 and files == [] and "--combo" in r.stderr, f"缺 --combo：rc={r.returncode} files={files} {r.stderr[-150:]}")
    r, files = run_sandboxed("account_sim_k1.py", "--period", "val", "--val-confirm", "--combo", "K1-20-S2")
    expect(r.returncode != 0 and files == [] and "開發期" in r.stderr, f"無開發期結果：rc={r.returncode} files={files} {r.stderr[-150:]}")
    r, files = run_sandboxed("account_sim_k1.py", "--period", "final", "--final-confirm", "--combo", "K1-20-S2")
    expect(r.returncode != 0 and files == [] and "驗證期" in r.stderr, f"無驗證期結果：rc={r.returncode} files={files} {r.stderr[-150:]}")
    r, files = run_sandboxed("account_sim_k1.py", "--period", "val", "--val-confirm", "--combo", "K1-99-S9")
    expect(r.returncode != 0 and files == [] and "invalid choice" in r.stderr, f"未知組合：{r.stderr[-150:]}")
    r, files = run_sandboxed("account_sim_k1.py")
    expect(r.returncode != 0 and "no such table" in r.stderr, r.stderr[-200:])
    expect(not any(f.endswith(("trials.csv", "acct5_dev_report.txt", "holdout_access.log", "k1_dev_result.json")) for f in files), files)
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
    text = "\n".join([f"# 帳戶模擬 K1 測試報告：{passed}／{len(TESTS)} 通過", *lines])
    out = ROOT / "data" / "experiments" / "account_sim_k1_tests_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
