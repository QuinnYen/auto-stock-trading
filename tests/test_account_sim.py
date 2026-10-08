"""帳戶模擬執行器（src/backtest/account_sim.py）的測試：績效整理、通過條件邊界、種子排序、小型合成資料的整合。

用法：python tests/test_account_sim.py        報告：data/experiments/account_sim_tests_report.txt
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "backtest"))
sys.path.insert(0, str(ROOT / "tests"))
import account  # noqa: E402
import account_sim as sim  # noqa: E402
import test_account_engine as eng  # noqa: E402

expect = eng.expect


def t01_max_drawdown():
    """[100,120,90,110]：最高 120 → 最低 90，回檔 25%；單調上升為 0。"""
    expect(abs(sim.max_drawdown([100, 120, 90, 110]) - 0.25) < 1e-12, "回檔應為 25%")
    expect(sim.max_drawdown([1, 2, 3]) == 0.0, "單調上升應為 0")


def t02_year_table():
    """兩年：2020 底 22000（本金 20000 → +10%、+2000 元）；2021 底 20000（相對 22000 → −9.09%、−2000 元）。"""
    dates = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-12-30", "2021-12-31"])
    yt = sim.year_table([21000, 22000, 20900, 20000], dates)
    expect(list(yt.index) == [2020, 2021], f"{list(yt.index)}")
    expect(abs(yt.ret[2020] - 0.10) < 1e-12 and abs(yt.pnl[2020] - 2000) < 1e-9, f"{yt.loc[2020].to_dict()}")
    expect(abs(yt.ret[2021] - (20000 / 22000 - 1)) < 1e-12 and abs(yt.pnl[2021] + 2000) < 1e-9, f"{yt.loc[2021].to_dict()}")


def t03_median_index():
    """20 次取由小到大第 10 名；5 次取第 3 名；回傳的是原順序中的位置。"""
    vals = np.array([float(v) for v in [50, 3, 17, 9, 1, 20, 11, 8, 15, 6, 19, 2, 14, 7, 4, 18, 12, 10, 16, 13]])
    i = sim.median_index(vals)
    expect(vals[i] == 11.0 and i == 6, f"第 10 名應為 11（位置 6），得到 {vals[i]}（位置 {i}）")
    expect(sim.median_index(np.array([5.0, 1, 4, 2, 3])) == 4, "5 次應取第 3 名（值 3，位置 4）")


GOOD_YEARS = [0.1, 0.05, -0.02, 0.03, 0.04, 0.02, -0.01, 0.06, 0.03, 0.02]


def verdicts(finals, ctrl, years, mdd):
    return [ok for _, ok in sim.judge(finals, ctrl, years, mdd)]


def t04_judge_all_pass_and_each_failure():
    """基準情況 6 條件全成立；每次只破壞一個條件，只有該條件為否。"""
    finals = np.full(20, 22000.0)
    ctrl = np.full(20, 21000.0)
    expect(verdicts(finals, ctrl, GOOD_YEARS, 0.2) == [True] * 6, f"{verdicts(finals, ctrl, GOOD_YEARS, 0.2)}")
    expect(not verdicts(np.full(20, 19000.0), np.full(20, 18000.0), GOOD_YEARS, 0.2)[0], "條件1")
    f2 = np.array([19000.0] * 6 + [22000.0] * 14)
    expect(verdicts(f2, ctrl, GOOD_YEARS, 0.2) == [True, False, True, True, True, True], f"{verdicts(f2, ctrl, GOOD_YEARS, 0.2)}")
    expect(verdicts(finals, np.full(20, 23000.0), GOOD_YEARS, 0.2) == [True, True, False, True, True, True], "條件3")
    y4 = [0.1, 0.05, 0.03, 0.04, 0.02, -0.01, -0.01, -0.01, -0.01, -0.01]
    expect(verdicts(finals, ctrl, y4, 0.2)[3] is False, "條件4")
    expect(verdicts(finals, ctrl, GOOD_YEARS, 0.35) == [True, True, True, True, False, True], "條件5")
    y6 = [2.0, 0.01, 0.01, 0.01, 0.01, 0.01, -0.3, -0.3, -0.3, -0.3]
    expect(verdicts(finals, ctrl, y6, 0.2) == [True, True, True, True, True, False], f"{verdicts(finals, ctrl, y6, 0.2)}")


def t05_judge_boundaries():
    """邊界：剛好 15 次賺錢成立；中位數剛好等於本金不成立；剛好 6 年為正成立；回檔剛好 30% 成立；其餘年度累計剛好 0 不成立。"""
    ctrl = np.full(20, 10000.0)
    f15 = np.array([19000.0] * 5 + [22000.0] * 15)
    expect(verdicts(f15, ctrl, GOOD_YEARS, 0.2)[1], "15 次應成立")
    expect(not verdicts(np.full(20, 20000.0), ctrl, GOOD_YEARS, 0.2)[0], "中位數 ＝ 本金不應成立")
    y6pos = [0.1, 0.05, 0.03, 0.04, 0.02, 0.01, -0.01, -0.01, -0.01, -0.01]
    expect(verdicts(np.full(20, 22000.0), ctrl, y6pos, 0.2)[3], "剛好 6 年為正應成立")
    expect(verdicts(np.full(20, 22000.0), ctrl, GOOD_YEARS, 0.30)[4], "回檔剛好 30% 應成立")
    y = [3.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expect(not verdicts(np.full(20, 22000.0), ctrl, y, 0.2)[5], "其餘累計 ＝ 0 不應成立")


def t06_make_cfg():
    """設定與事前約定一致：固定比例、收盤停損 15%、1/3 資金、停機線 10%、滑價 0.25%、持有天數 ＝ time_stop。"""
    c = sim.make_cfg(60)
    expect(not c.risk_sized and c.stop_close and c.stop_pct == 0.15 and abs(c.pos_cap - 1 / 3) < 1e-12, f"{c}")
    expect(c.slip == 0.0025 and c.stop_line == 0.10 and c.time_stop == 60 and c.max_pos == 3, f"{c}")
    expect(c.ma_exit is None and not c.exit_on_attention and c.tp_pct is None, f"{c}")


def t07_seeded_rank():
    """同種子相同、不同種子不同、形狀正確。"""
    a, b, c = sim.seeded_rank((5, 4), 3), sim.seeded_rank((5, 4), 3), sim.seeded_rank((5, 4), 4)
    expect(a.shape == (5, 4) and np.array_equal(a, b) and not np.array_equal(a, c), "種子行為不符")


def synth(n_st=6):
    a = eng.mk(n_st=n_st)
    return account.Market(**a), a["O"].shape


def t08_seeded_selection_follows_rank():
    """6 檔同日（第 3 天）訊號、3 個空位：每個種子買進的是該種子排名前 3 的股票。"""
    mkt, shape = synth()
    entry = np.zeros(shape, dtype=bool)
    entry[3, :] = True
    elig = np.ones(shape, dtype=bool)
    res = sim.run_seeds(mkt, entry, elig, sim.make_cfg(5), 1, shape[0], [1, 2, 3])
    for s, r in zip([1, 2, 3], res):
        want = sorted(np.argsort(-sim.seeded_rank(shape, s)[3])[:3].tolist())
        got = sorted(t["stock"] for t in r.trades)
        expect(got == want, f"種子 {s}：{got} vs {want}")
    picks = {tuple(sorted(t["stock"] for t in r.trades)) for r in res}
    expect(len(picks) > 1, "三個種子選到完全相同的股票，種子沒有作用")


def t09_control_buys_random_each_day():
    """隨機進場對照（entry ＝ 全部合格）：第一個進場日（訊號日 1 → 第 2 天）買進該種子排名前 3 的股票。"""
    mkt, shape = synth()
    elig = np.ones(shape, dtype=bool)
    r = sim.run_control(mkt, elig, sim.make_cfg(5), 1, shape[0], [5])[0]
    first = sorted(t["stock"] for t in r.trades if t["entry_idx"] == 2)
    want = sorted(np.argsort(-sim.seeded_rank(shape, 5)[1])[:3].tolist())
    expect(first == want, f"{first} vs {want}")


def t10_summarize_rule_consistent():
    """合成資料跑 20 個種子後整理：中位數取第 10 名、期末金額與逐年損益相符、報告含通過條件。"""
    mkt, shape = synth()
    for j in range(shape[1]):
        for key in ("O", "H", "L", "C", "RO"):
            getattr(mkt, key)[8:, j] = 100 + 3 * j
    entry = np.zeros(shape, dtype=bool)
    entry[3, :] = True
    elig = np.ones(shape, dtype=bool)
    cfg = sim.make_cfg(5)
    res = sim.run_seeds(mkt, entry, elig, cfg, 1, shape[0], range(1, 21))
    ctrl = sim.run_control(mkt, elig, cfg, 1, shape[0], range(1, 21))
    cal = pd.date_range("2020-12-01", periods=shape[0], freq="D")
    rule = sim.Rule("T", "測試規則", 5)
    lines, passed, stat = sim.summarize_rule(rule, res, ctrl, {}, cal, 1)
    finals = sim.finals_of(res)
    expect(len(set(np.round(finals, 6))) > 1, "合成資料沒有造成不同期末淨值")
    expect(abs(stat["final"] - np.median(finals)) < 1e-9, f"{stat['final']} vs {np.median(finals)}")
    text = "\n".join(lines)
    expect("通過條件" in text and "結論" in text and "逐年賺賠" in text, "報告缺少段落")
    mi = sim.median_index(finals)
    yt = sim.year_table(res[mi].equity, cal[1:1 + len(res[mi].equity)])
    expect(f"{yt.pnl.iloc[0]:+,.0f}元" in text, f"逐年金額 {yt.pnl.iloc[0]:+,.0f} 不在報告中")
    expect(isinstance(passed, bool), "passed 應為布林")


TESTS = [t01_max_drawdown, t02_year_table, t03_median_index, t04_judge_all_pass_and_each_failure, t05_judge_boundaries,
         t06_make_cfg, t07_seeded_rank, t08_seeded_selection_follows_rank, t09_control_buys_random_each_day,
         t10_summarize_rule_consistent]


def main():
    lines, passed = [], 0
    for fn in TESTS:
        title = (fn.__doc__ or fn.__name__).strip().splitlines()[0]
        try:
            fn()
            passed += 1
            lines.append(f"[通過] {fn.__name__}：{title}")
        except Exception as e:  # noqa: BLE001  測試要回報所有失敗
            lines.append(f"[失敗] {fn.__name__}：{title}\n        {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                lines.append("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    text = "\n".join([f"# 帳戶模擬執行器測試報告：{passed}／{len(TESTS)} 通過", *lines])
    out = ROOT / "data" / "experiments" / "account_sim_tests_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
