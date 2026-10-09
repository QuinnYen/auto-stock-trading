"""帳戶模擬 v2 執行器（src/backtest/account_sim_v2.py，依 docs/帳戶模擬事前約定 v2.md）的測試。

涵蓋：結構設定（持股數、每檔比例、停機線 20%）、6 次試驗的清單與偏好順序、通過條件與邊界、0050 買進持有參考線、
規則 D 的進場訊號接線、引擎在 3 檔／6 檔與 20%／10% 停機線下的行為（人工構造情境）、入口腳本只能跑開發期（沙盒）。

用法：python tests/test_account_sim_v2.py        報告：data/experiments/account_sim_v2_tests_report.txt
"""
import sqlite3
import sys
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
import account_sim_v2 as v2  # noqa: E402
import test_account_engine as eng  # noqa: E402
import test_signal_scan_v3 as t3  # noqa: E402
from sandbox import run_sandboxed  # noqa: E402

expect = eng.expect


def t01_make_cfg2_structures():
    """S2 ＝ 3 檔、每檔 1/3；S4 ＝ 6 檔、每檔 1/6；兩者停機線 20%、單檔收盤停損 15%、持有天數 ＝ time_stop、固定比例、滑價 0.25%。"""
    for name, n in (("S2", 3), ("S4", 6)):
        c = v2.make_cfg2(60, name)
        expect(c.max_pos == n and abs(c.pos_cap - 1 / n) < 1e-12, f"{name}: {c}")
        expect(c.stop_line == 0.20 and c.restart_after == 20 and c.stop_pct == 0.15 and c.time_stop == 60, f"{name}: {c}")
        expect(not c.risk_sized and c.stop_close and c.ma_exit is None and not c.exit_on_attention and c.tp_pct is None, f"{name}: {c}")
        expect(c.slip == 0.0025 and c.min_shares == 10, f"{name}: {c}")
    expect(v2.make_cfg2(60, "S2", slip=0.005).slip == 0.005 and v2.make_cfg2(60, "S2", stop_pct=1.0).stop_pct == 1.0, "敏感度參數")


def t02_combos_are_the_six_trials():
    """6 次試驗：規則 A、B、D（皆持有 60 天）× 結構 S2、S4；沒有 10% 停機線的結構，也沒有規則 C。"""
    ids = [c.sid for c in v2.COMBOS]
    expect(ids == ["A-S2", "A-S4", "B-S2", "B-S4", "D-S2", "D-S4"], ids)
    expect(set(v2.STRUCTS) == {"S2", "S4"} and all(r.h == 60 for r in v2.RULES2) and [r.sid for r in v2.RULES2] == ["A", "B", "D"], "清單")
    expect(all(s[1] == 0.20 for s in v2.STRUCTS.values()), "停機線都是 20%")


def t03_thresholds():
    """年化 5% 的十年門檻 ＝ 20,000 × 1.05^10 ＝ 32,578（取整）；最大回檔上限 25%。"""
    expect(abs(v2.MIN_FINAL_DEV - 20000 * 1.05 ** 10) < 1e-9 and round(v2.MIN_FINAL_DEV) == 32578, v2.MIN_FINAL_DEV)
    expect(v2.MAX_MDD == 0.25 and v2.MIN_WIN_RUNS == 15 and v2.MIN_POS_YEARS == 6, "門檻")


GOOD_YEARS = [0.1, 0.05, -0.02, 0.03, 0.04, 0.02, -0.01, 0.06, 0.03, 0.02]


def verdicts(finals, ctrl, years, mdd):
    return [ok for _, ok in v2.judge2(finals, ctrl, years, mdd)]


def t04_judge2_each_condition_alone():
    """基準情況 6 條件全成立；每次只破壞一個條件，只有該條件為否。"""
    finals, ctrl = np.full(20, 40000.0), np.full(20, 30000.0)
    expect(verdicts(finals, ctrl, GOOD_YEARS, 0.2) == [True] * 6, f"{verdicts(finals, ctrl, GOOD_YEARS, 0.2)}")
    expect(verdicts(np.full(20, 30000.0), np.full(20, 20000.0), GOOD_YEARS, 0.2) == [False, True, True, True, True, True], "條件1")
    f2 = np.array([19000.0] * 6 + [40000.0] * 14)          # 中位數仍高、但只有 14 次賺錢
    expect(verdicts(f2, ctrl, GOOD_YEARS, 0.2) == [True, False, True, True, True, True], f"{verdicts(f2, ctrl, GOOD_YEARS, 0.2)}")
    expect(verdicts(finals, np.full(20, 41000.0), GOOD_YEARS, 0.2) == [True, True, False, True, True, True], "條件3")
    y4 = [0.1, 0.05, 0.03, 0.04, 0.02, -0.01, -0.01, -0.01, -0.01, -0.01]
    expect(verdicts(finals, ctrl, y4, 0.2) == [True, True, True, False, True, True], "條件4")
    expect(verdicts(finals, ctrl, GOOD_YEARS, 0.30) == [True, True, True, True, False, True], "條件5")
    y6 = [2.0, 0.01, 0.01, 0.01, 0.01, 0.01, -0.3, -0.3, -0.3, -0.3]
    expect(verdicts(finals, ctrl, y6, 0.2) == [True, True, True, True, True, False], f"{verdicts(finals, ctrl, y6, 0.2)}")


def t05_judge2_boundaries():
    """邊界：中位數剛好等於門檻成立、少 1 元不成立；剛好 15 次賺錢成立；剛好 6 年為正成立；回檔剛好 25% 成立、25.01% 不成立。"""
    ctrl = np.full(20, 10000.0)
    expect(verdicts(np.full(20, v2.MIN_FINAL_DEV), ctrl, GOOD_YEARS, 0.2)[0], "剛好等於門檻應成立")
    expect(not verdicts(np.full(20, v2.MIN_FINAL_DEV - 1), ctrl, GOOD_YEARS, 0.2)[0], "少 1 元不應成立")
    f15 = np.array([19000.0] * 5 + [40000.0] * 15)
    expect(verdicts(f15, ctrl, GOOD_YEARS, 0.2)[1], "15 次應成立")
    y6 = [0.1, 0.05, 0.03, 0.04, 0.02, 0.01, -0.01, -0.01, -0.01, -0.01]
    expect(verdicts(np.full(20, 40000.0), ctrl, y6, 0.2)[3], "剛好 6 年為正應成立")
    expect(verdicts(np.full(20, 40000.0), ctrl, GOOD_YEARS, 0.25)[4], "回檔剛好 25% 應成立")
    expect(not verdicts(np.full(20, 40000.0), ctrl, GOOD_YEARS, 0.2501)[4], "回檔 25.01% 不應成立")
    y = [3.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expect(not verdicts(np.full(20, 40000.0), ctrl, y, 0.2)[5], "其餘累計 ＝ 0 不應成立")


def t05b_control_tie_is_not_a_pass():
    """中位數剛好等於隨機進場對照中位數 → 條件 3 不成立（需『高於』）。"""
    expect(not verdicts(np.full(20, 40000.0), np.full(20, 40000.0), GOOD_YEARS, 0.2)[2], "相等不應成立")
    expect(verdicts(np.full(20, 40000.0), np.full(20, 39999.0), GOOD_YEARS, 0.2)[2], "高 1 元應成立")


def t06_pick_preferred():
    """多個組合通過時：S2（3 檔）優先於 S4；同一結構下 A ＞ B ＞ D；沒有通過者回傳 None。"""
    expect(v2.pick_preferred(["D-S4", "B-S4", "A-S4"]) == "A-S4", "同為 S4 取 A")
    expect(v2.pick_preferred(["A-S4", "D-S2"]) == "D-S2", "S2 優先於 S4，即使規則較後")
    expect(v2.pick_preferred(["B-S2", "A-S4", "D-S2"]) == "B-S2", "S2 內取 B ＞ D")
    expect(v2.pick_preferred(["A-S2", "B-S2", "D-S2", "A-S4"]) == "A-S2", "全部通過取 A-S2")
    expect(v2.pick_preferred([]) is None, "無通過者")


def t07_buy_hold_reference():
    """0050 買進持有參考線：100→110→99→121，2 萬元期末 24,200，最大回檔 (110−99)/110 ＝ 10%。"""
    final, mdd = v2.buy_hold([100, 110, 99, 121], 20000)
    expect(abs(final - 24200) < 1e-9 and abs(mdd - 0.10) < 1e-12, f"{final} {mdd}")


def t08_rule_d_entry_is_h1a_from_revenue_table():
    """規則 D 的進場訊號 ＝ H1a（營收年增創高）：從 month_revenue 資料表讀取，訊號日為次月 10 日當天或之後第一個交易日。"""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE month_revenue (stock_id TEXT, date TEXT, country TEXT, revenue REAL, revenue_month INT, revenue_year INT, create_time TEXT)")
    con.executemany("INSERT INTO month_revenue VALUES (?,?,?,?,?,?,?)",
                    [(sid, "x", "Taiwan", rev, m, y, "") for sid, y, m, rev in t3.rev_rows("1111", t3.base_series())])
    cal = t3.CAL_R
    cols = ["1111", "2222"]
    D = v2.d_entry(con, cal, cols, (len(cal), 2))
    got = {(str(cal[i].date()), cols[j]) for i, j in zip(*np.nonzero(D))}
    expect(got == {("2020-08-10", "1111")}, f"{got}")
    # 另一檔只有 H1a、沒有 H1b：2018～2020 全 1000，只有 2020-07 為 1200（+20%、上月年增 0% 不是負）
    flat = {(y, m): 1000 for y in (2018, 2019, 2020) for m in range(1, 13)}
    flat[(2020, 7)] = 1200
    con.executemany("INSERT INTO month_revenue VALUES (?,?,?,?,?,?,?)",
                    [(sid, "x", "Taiwan", rev, m, y, "") for sid, y, m, rev in t3.rev_rows("2222", flat)])
    D = v2.d_entry(con, cal, cols, (len(cal), 2))
    got = {(str(cal[i].date()), cols[j]) for i, j in zip(*np.nonzero(D))}
    expect(got == {("2020-08-10", "1111"), ("2020-08-10", "2222")}, f"只有 H1a 的那檔也要有訊號：{got}")


def synth(n_st):
    a = eng.mk(n_st=n_st)
    return a, account.Market(**a), a["O"].shape


def t09_s4_holds_six_with_half_size():
    """6 檔同日訊號：S2 只買 3 檔、每檔 66 股（20000÷3÷100.25 取整）；S4 買 6 檔、每檔 33 股（20000÷6÷100.25 取整）。"""
    a, mkt, shape = synth(6)
    entry = np.zeros(shape, dtype=bool)
    entry[3, :] = True
    elig = np.ones(shape, dtype=bool)
    rank = np.tile(-np.arange(6, dtype="float64"), (shape[0], 1))
    r2 = account.simulate(mkt, entry, rank, elig, v2.make_cfg2(5, "S2"), 1, shape[0])
    r4 = account.simulate(mkt, entry, rank, elig, v2.make_cfg2(5, "S4"), 1, shape[0])
    expect(sorted(t["stock"] for t in r2.trades) == [0, 1, 2] and {t["shares"] for t in r2.trades} == {66}, f"S2 {[(t['stock'], t['shares']) for t in r2.trades]}")
    expect(sorted(t["stock"] for t in r4.trades) == [0, 1, 2, 3, 4, 5] and {t["shares"] for t in r4.trades} == {33}, f"S4 {[(t['stock'], t['shares']) for t in r4.trades]}")


def t10_stop_line_20_does_not_trigger_where_10_does():
    """3 檔滿倉後同時下跌 15%（淨值約 −15%）：20% 停機線不觸發；v1 的 10% 停機線在同一情境觸發。"""
    a, mkt, shape = synth(3)
    for j in range(3):
        eng.set_all(a, 6, j, 85)
    mkt = account.Market(**a)
    entry = np.zeros(shape, dtype=bool)
    entry[3, :] = True
    elig = np.ones(shape, dtype=bool)
    rank = np.tile(-np.arange(3, dtype="float64"), (shape[0], 1))
    r20 = account.simulate(mkt, entry, rank, elig, v2.make_cfg2(5, "S2"), 1, shape[0])
    r10 = account.simulate(mkt, entry, rank, elig, sim.make_cfg(5), 1, shape[0])
    expect(r20.triggers == [] and len(r10.triggers) >= 1, f"20%：{r20.triggers}；10%：{r10.triggers}")
    expect(r20.equity.min() > 20000 * 0.8, f"20% 線下的最低淨值 {r20.equity.min():.0f} 不應低於 16,000")


def t11_summarize_combo_uses_v2_rules():
    """整理一個組合：標題含組合編號；6 個通過條件用 v2 門檻（含 25% 回檔、32,578 元）；回傳 passed／stat。"""
    a, mkt, shape = synth(6)
    entry = np.zeros(shape, dtype=bool)
    entry[3, :] = True
    elig = np.ones(shape, dtype=bool)
    cfg = v2.make_cfg2(5, "S4")
    res = sim.run_seeds(mkt, entry, elig, cfg, 1, shape[0], range(1, 21))
    ctrl = sim.run_control(mkt, elig, cfg, 1, shape[0], range(1, 21))
    cal = pd.date_range("2020-12-01", periods=shape[0], freq="D")
    combo = v2.COMBOS[1]
    lines, passed, stat = v2.summarize_combo(combo, res, ctrl, {}, cal, 1)
    text = "\n".join(lines)
    expect("A-S4" in lines[0], lines[0])
    expect("32,578" in text and "≤ 25%" in text and text.count("[✓]") + text.count("[✗]") == 6, text)
    expect(isinstance(passed, bool) and abs(stat["final"] - np.median(sim.finals_of(res))) < 1e-9, "stat")


def t11b_run_combo_wiring():
    """run_combo：用組合對應的進場訊號與結構設定；對照依結構快取重複使用；三個敏感度情境各自使用對應的滑價／停損設定。"""
    a, mkt, shape = synth(6)
    for j in range(3, 6):
        eng.set_all(a, 8, j, 130)        # D 會買的 3、4、5 號股票進場後上漲；A 買的 0、1、2 號平盤
    eng.set_all(a, 6, 0, 80)             # A 會買的 0 號股票跌 20%：單檔停損 15% 會賣出，不設停損則續抱
    elig = np.ones(shape, dtype=bool)
    ea, ed = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    ea[3, :3] = True
    ed[5, 3:] = True
    ctx = {"mkt": mkt, "std": elig, "entries": {"A": ea, "D": ed}}
    controls, seeds = {}, range(1, 6)
    res_a, ctrl_a, extra = v2.run_combo(ctx, v2.COMBOS[0], controls, 1, shape[0], seeds)       # A-S2
    expect(np.array_equal(sim.finals_of(res_a), sim.finals_of(sim.run_seeds(mkt, ea, elig, v2.make_cfg2(60, "S2"), 1, shape[0], seeds))), "A-S2 照規則")
    expect(not np.array_equal(sim.finals_of(res_a), sim.finals_of(sim.run_seeds(mkt, ed, elig, v2.make_cfg2(60, "S2"), 1, shape[0], seeds))), "D 的訊號和 A 不同")
    res_d, ctrl_d, _ = v2.run_combo(ctx, v2.COMBOS[4], controls, 1, shape[0], seeds)           # D-S2
    expect(ctrl_d is ctrl_a, "同一結構的對照要重複使用同一份")
    expect(np.array_equal(sim.finals_of(res_d), sim.finals_of(sim.run_seeds(mkt, ed, elig, v2.make_cfg2(60, "S2"), 1, shape[0], seeds))), "D-S2 照規則")
    expect(np.array_equal(sim.finals_of(ctrl_a), sim.finals_of(sim.run_control(mkt, elig, v2.make_cfg2(60, "S2"), 1, shape[0], seeds))), "對照 ＝ 全部合格股票隨機進場")
    res_4, ctrl_4, _ = v2.run_combo(ctx, v2.COMBOS[1], controls, 1, shape[0], seeds)           # A-S4
    expect(ctrl_4 is not ctrl_a and set(controls) == {"S2", "S4"}, "S4 要有自己的對照")
    want4 = sim.finals_of(sim.run_seeds(mkt, ea, elig, v2.make_cfg2(60, "S4"), 1, shape[0], seeds))
    expect(np.array_equal(sim.finals_of(res_4), want4) and not np.array_equal(want4, sim.finals_of(res_a)), "A-S4 照 S4 設定")
    expect(np.array_equal(sim.finals_of(ctrl_4), sim.finals_of(sim.run_control(mkt, elig, v2.make_cfg2(60, "S4"), 1, shape[0], seeds))), "S4 對照")
    expect(list(extra) == ["滑價 0%", "滑價 0.5%", "不設單檔停損"], list(extra))
    for key, kw in (("滑價 0%", {"slip": 0.0}), ("滑價 0.5%", {"slip": 0.005}), ("不設單檔停損", {"stop_pct": 1.0})):
        want = sim.finals_of(sim.run_seeds(mkt, ea, elig, v2.make_cfg2(60, "S2", **kw), 1, shape[0], seeds))
        expect(np.array_equal(sim.finals_of(extra[key]), want), key)
    expect(not np.array_equal(sim.finals_of(extra["滑價 0%"]), sim.finals_of(extra["滑價 0.5%"])), "兩種滑價結果要不同")
    expect(not np.array_equal(sim.finals_of(extra["不設單檔停損"]), sim.finals_of(res_a)), "不設單檔停損要和照規則不同")


def t12_entry_script_default_is_dev_in_sandbox():
    """入口腳本預設跑開發期：沙盒（空資料庫）中直接跑會因缺資料表失敗，不留 trials 或報告，也不寫帳本；不認得的參數被拒絕。"""
    r, files = run_sandboxed("account_sim_v2.py")
    expect(r.returncode != 0 and "no such table" in r.stderr, r.stderr[-200:])
    expect(not any(f.endswith(("trials.csv", "acct2_dev_report.txt", "holdout_access.log")) for f in files), files)
    r, files = run_sandboxed("account_sim_v2.py", "--bogus")
    expect(r.returncode != 0 and files == [], f"rc={r.returncode} files={files}")


def verdicts_val(finals, ctrl, years, mdd):
    return [ok for _, ok in v2.judge_val(finals, ctrl, years, mdd)]


VAL_YEARS = [0.2, 0.05, -0.02, 0.03, 0.04]


def t13_judge_val_thresholds_and_boundaries():
    """驗證期（事前約定 v3 第 5 節）：中位數 ≥ 25,526（20000×1.05^5）；≥15 次賺錢；高於對照；5 年中 ≥3 年為正；回檔 ≤ 40%；去掉最佳年仍為正。"""
    expect(abs(v2.MIN_FINAL_VAL - 20000 * 1.05 ** 5) < 1e-9 and round(v2.MIN_FINAL_VAL) == 25526, v2.MIN_FINAL_VAL)
    expect(v2.MIN_POS_YEARS_VAL == 3 and v2.MAX_MDD_VAL == 0.40, "門檻常數")
    ctrl, good = np.full(20, 20000.0), np.full(20, 30000.0)
    expect(verdicts_val(good, ctrl, VAL_YEARS, 0.30) == [True] * 6, f"{verdicts_val(good, ctrl, VAL_YEARS, 0.30)}")
    expect(verdicts_val(np.full(20, v2.MIN_FINAL_VAL), ctrl, VAL_YEARS, 0.3)[0], "剛好等於門檻成立")
    expect(not verdicts_val(np.full(20, v2.MIN_FINAL_VAL - 1), ctrl, VAL_YEARS, 0.3)[0], "少 1 元不成立")
    expect(verdicts_val(np.array([19000.0] * 5 + [30000.0] * 15), ctrl, VAL_YEARS, 0.3)[1], "15 次成立")
    expect(not verdicts_val(np.array([19000.0] * 6 + [30000.0] * 14), ctrl, VAL_YEARS, 0.3)[1], "14 次不成立")
    expect(not verdicts_val(good, np.full(20, 30000.0), VAL_YEARS, 0.3)[2], "對照相等不成立")
    expect(verdicts_val(good, ctrl, [0.1, 0.1, 0.1, -0.1, -0.1], 0.3)[3], "剛好 3 年為正成立")
    expect(not verdicts_val(good, ctrl, [0.1, 0.1, -0.1, -0.1, -0.1], 0.3)[3], "2 年為正不成立")
    expect(verdicts_val(good, ctrl, VAL_YEARS, 0.40)[4] and not verdicts_val(good, ctrl, VAL_YEARS, 0.4001)[4], "回檔 40% 邊界")
    expect(not verdicts_val(good, ctrl, [2.0, 0.01, -0.3, -0.3, -0.3], 0.3)[5], "去掉最佳年後累計為負不成立")
    expect(verdicts(good, ctrl, VAL_YEARS, 0.30) != verdicts_val(good, ctrl, VAL_YEARS, 0.30), "開發期判定（25% 回檔、6 年為正）與驗證期不同")


def t14_mode_config():
    """開發期維持 v2 的 6 個組合與 acct2-dev；驗證期只有 A-S2、標記 acct3-val、報告 acct3_val_report.txt、期間為 2017～2021。"""
    dev, val = v2.mode_config("dev"), v2.mode_config("val")
    expect([c.sid for c in dev["combos"]] == [c.sid for c in v2.COMBOS] and dev["tag"] == "acct2-dev" and dev["report"] == "acct2_dev_report.txt", f"{dev}")
    expect(dev["period"] == ("2007-01-01", "2016-12-31") and dev["judge"] is v2.judge2, "開發期設定")
    expect([c.sid for c in val["combos"]] == ["A-S2"] and val["tag"] == "acct3-val" and val["report"] == "acct3_val_report.txt", f"{val}")
    expect(val["period"] == ("2017-01-01", "2021-12-31") and val["judge"] is v2.judge_val, "驗證期設定")


def t15_val_entry_guards_in_sandbox():
    """入口腳本：驗證期沒有 --val-confirm 被防護拒絕（訊息提到 --val-confirm）且不寫任何東西；不認得的期間被參數檢查拒絕；有 --val-confirm 時才進入帳本（沙盒內），空資料庫失敗，不寫 trials 或報告。"""
    r, files = run_sandboxed("account_sim_v2.py", "--period", "val")
    expect(r.returncode != 0 and files == [] and "--val-confirm" in r.stderr, f"無旗標：rc={r.returncode} files={files} err={r.stderr[-200:]}")
    r, files = run_sandboxed("account_sim_v2.py", "--period", "bogus", "--val-confirm")
    expect(r.returncode != 0 and files == [] and "invalid choice" in r.stderr, f"bogus：rc={r.returncode} files={files} err={r.stderr[-200:]}")
    r, files = run_sandboxed("account_sim_v2.py", "--period", "val", "--val-confirm")
    expect(r.returncode != 0 and "no such table" in r.stderr, f"空資料庫應因缺資料表失敗：{r.stderr[-200:]}")
    expect(any(f.endswith("holdout_access.log") for f in files), f"確認後應寫入（沙盒內的）帳本：{files}")
    expect(not any(f.endswith(("trials.csv", "acct3_val_report.txt", "acct2_dev_report.txt")) for f in files), files)


def t16_report_year_label_follows_period_length():
    """報告第二行的年數依模擬長度計算：約 1,225 個交易日寫『5 年』（驗證期），約 2,450 個交易日仍寫『10 年』（開發期）。"""
    for n_days, label in ((1225, "做 5 年"), (2450, "做 10 年")):
        a = eng.mk(n_st=6, n_days=n_days)
        mkt, shape = account.Market(**a), a["O"].shape
        entry = np.zeros(shape, dtype=bool)
        entry[3, :] = True
        elig = np.ones(shape, dtype=bool)
        cfg = v2.make_cfg2(5, "S2")
        res = sim.run_seeds(mkt, entry, elig, cfg, 1, shape[0], range(1, 4))
        ctrl = sim.run_control(mkt, elig, cfg, 1, shape[0], range(1, 4))
        cal = pd.date_range("2017-01-01", periods=shape[0], freq="D")
        lines, _, _ = sim.summarize_rule(sim.Rule("T", "測試", 5), res, ctrl, {}, cal, 1)
        expect(label in lines[1], f"{n_days}: {lines[1]}")


def t17_judge_final_scales_with_days():
    """最後測試期（事前約定 v4 第 4 節）：期末門檻 ＝ 20000×1.05^(N÷245)；至少 3 個日曆年度為正；回檔 ≤ 40%；其餘同驗證期。"""
    ctrl, years = np.full(20, 20000.0), [0.2, 0.05, -0.02, 0.03, 0.04]
    for n in (980, 1170):
        need = 20000 * 1.05 ** (n / 245)
        ok = [x for _, x in v2.judge_final(np.full(20, need), ctrl, years, 0.3, n_days=n)]
        low = [x for _, x in v2.judge_final(np.full(20, need - 1), ctrl, years, 0.3, n_days=n)]
        expect(ok == [True] * 6, f"N={n} 剛好等於門檻應全成立：{ok}")
        expect(low[0] is False and all(low[1:]), f"N={n} 少 1 元只有第 1 條不成立：{low}")
    text = v2.judge_final(np.full(20, 40000.0), ctrl, years, 0.3, n_days=1170)[0][0]
    expect(f"{20000 * 1.05 ** (1170 / 245):,.0f}" in text and "25,526" not in text, f"門檻文字應是依 N 換算的數字：{text}")
    g = lambda y, m=0.3: [x for _, x in v2.judge_final(np.full(20, 40000.0), ctrl, y, m, n_days=1170)]  # noqa: E731
    expect(g([0.1, 0.1, 0.1, -0.1, -0.1])[3] and not g([0.1, 0.1, -0.1, -0.1, -0.1])[3], "3 個年度為正的邊界")
    expect(g(years, 0.40)[4] and not g(years, 0.4001)[4], "回檔 40% 邊界")
    expect(not g([2.0, 0.01, -0.3, -0.3, -0.3])[5], "去掉最佳年後累計為負")


def t18_mode_config_final():
    """最後測試期：只有 A-S2、標記 acct4-final、報告 acct4_final_report.txt、期間從 2022-01-01 起；驗證期與開發期設定不變。"""
    f = v2.mode_config("final")
    expect([c.sid for c in f["combos"]] == ["A-S2"] and f["tag"] == "acct4-final" and f["report"] == "acct4_final_report.txt", f"{f}")
    expect(f["period"][0] == "2022-01-01" and f["judge"] is v2.judge_final, "最後測試期設定")
    expect(v2.mode_config("val")["tag"] == "acct3-val" and v2.mode_config("dev")["tag"] == "acct2-dev", "其他期間設定不應變")


def t18b_bind_judge():
    """bind_judge：最後測試期把交易日數綁進判定（門檻隨 N 變）；驗證期與開發期原樣回傳判定函式。"""
    ctrl, years = np.full(20, 20000.0), [0.2, 0.05, -0.02, 0.03, 0.04]
    f = v2.bind_judge("final", v2.judge_final, 1000)
    expect(f(np.full(20, 20000 * 1.05 ** (1000 / 245)), ctrl, years, 0.3)[0][1], "N＝1000 的門檻剛好成立")
    expect(not f(np.full(20, 20000 * 1.05 ** (1000 / 245) - 1), ctrl, years, 0.3)[0][1], "少 1 元不成立")
    expect(v2.bind_judge("val", v2.judge_val, 1000) is v2.judge_val and v2.bind_judge("dev", v2.judge2, 1000) is v2.judge2, "非最後測試期原樣回傳")


def t19_final_entry_guards_in_sandbox():
    """入口腳本：最後測試期沒有 --final-confirm 被拒絕（只有 --val-confirm 也不行，訊息提到 --final-confirm）且不寫東西；有 --final-confirm 時帳本只寫在沙盒內，真實帳本與 trials.csv 位元組不變。"""
    import holdout
    real, trials = holdout.LEDGER, holdout.LEDGER.parent / "trials.csv"
    snap = lambda: (real.read_bytes(), trials.read_bytes())  # noqa: E731
    before = snap()
    for extra in ([], ["--val-confirm"]):
        r, files = run_sandboxed("account_sim_v2.py", "--period", "final", *extra)
        expect(r.returncode != 0 and files == [] and "--final-confirm" in r.stderr, f"{extra}: rc={r.returncode} files={files} err={r.stderr[-200:]}")
    r, files = run_sandboxed("account_sim_v2.py", "--period", "final", "--final-confirm")
    expect(r.returncode != 0 and "no such table" in r.stderr, f"空資料庫應因缺資料表失敗：{r.stderr[-200:]}")
    expect(any(f.endswith("holdout_access.log") for f in files), f"確認後應寫入沙盒內的帳本：{files}")
    expect(not any(f.endswith(("trials.csv", "acct4_final_report.txt")) for f in files), files)
    expect(snap() == before, "真實帳本或 trials.csv 被改動了")


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
    text = "\n".join([f"# 帳戶模擬 v2 測試報告：{passed}／{len(TESTS)} 通過", *lines])
    out = ROOT / "data" / "experiments" / "account_sim_v2_tests_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
