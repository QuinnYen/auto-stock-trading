"""資金×持股數探索（src/backtest/explore_capital.py）的測試：設定、績效整理、資金縮放性質、組合清單、報告、入口腳本只跑開發期（沙盒）。

用法：python tests/test_explore_capital.py        報告：data/experiments/explore_capital_tests_report.txt
"""
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "backtest"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "tests"))
import account  # noqa: E402
import explore_capital as ex  # noqa: E402
import test_account_engine as eng  # noqa: E402
from sandbox import run_sandboxed  # noqa: E402

expect = eng.expect


def t01_make_cfg_x():
    """設定：固定比例（1/N）、最多 N 檔、拿掉帳戶停機線（永不觸發）、單檔停損 7%、跌破 10 日均線賣、持有最多 10 天、注意股出場、不設停利。"""
    for n in (3, 10, 30):
        c = ex.make_cfg_x(n, 0.0025)
        expect(c.max_pos == n and abs(c.pos_cap - 1 / n) < 1e-12 and not c.risk_sized, f"{n}: {c}")
        expect(c.stop_line >= 1.0 and c.stop_pct == 0.07 and c.ma_exit == 10 and c.time_stop == 10, f"{n}: {c}")
        expect(c.tp_pct is None and c.exit_on_attention and c.slip == 0.0025 and not c.stop_close, f"{n}: {c}")
    expect(ex.make_cfg_x(3, 0.0).slip == 0.0, "滑價參數")


def t02_grid():
    """組合：4 種資金 × 3 種持股數 × 2 個版本 ＝ 24 個設定；每個設定跑滑價 0%、0.25% 兩次 ＝ 48 次模擬。"""
    expect(ex.CAPITALS == (20_000, 200_000, 2_000_000, 20_000_000) and ex.SLOTS == (3, 10, 30), f"{ex.CAPITALS} {ex.SLOTS}")
    expect(ex.VERSIONS == ("breakout", "pullback") and ex.SLIPS_X == (0.0, 0.0025), "版本與滑價")
    cells = ex.grid()
    expect(len(cells) == 24 and len(set(cells)) == 24, len(cells))
    expect(len(cells) * len(ex.SLIPS_X) == 48, "模擬次數")


def fake_result(capital, n=245, final_mult=1.21, dip=0.9):
    eq = np.full(n, capital, dtype="float64")
    eq[n // 3:] = capital * 1.1
    eq[n // 2:] = capital * 1.1 * dip      # 先漲 10%，再跌 10%（回檔 10%）
    eq[-1] = capital * final_mult
    trades = [{"net": 100.0, "basis": 1000.0}, {"net": -50.0, "basis": 500.0}, {"net": 30.0, "basis": 600.0}]
    return account.Result(eq, np.full(n, capital * 0.2), np.array([2] * (n // 2) + [4] * (n - n // 2)), trades, [], 321.0, capital=capital)


def t03_summarize_x_hand_computed():
    """績效整理（資金 20 萬、245 個交易日 ＝ 1 年）：期末倍數 1.21、年化 21%、最大回檔 (1.1−0.99)/1.1 ＝ 10%、3 筆交易勝率 2/3、平均報酬 (10%−10%+5%)/3、平均持股 3 檔、成本 321。"""
    st = ex.summarize_x(fake_result(200_000), 200_000)
    expect(abs(st["final"] - 242_000) < 1e-6 and abs(st["mult"] - 1.21) < 1e-12 and abs(st["cagr"] - 0.21) < 1e-9, st)
    expect(abs(st["mdd"] - 0.10) < 1e-12, st["mdd"])
    expect(st["trades"] == 3 and abs(st["win"] - 2 / 3) < 1e-12 and abs(st["avg_ret"] - (0.10 - 0.10 + 0.05) / 3) < 1e-12, st)
    expect(abs(st["avg_net"] - 80 / 3) < 1e-9 and abs(st["avg_held"] - np.mean([2] * 122 + [4] * 123)) < 1e-9 and st["cost"] == 321.0, st)


def t03b_cagr_uses_years():
    """年化 ＝ 期末倍數^(1÷年數)−1：490 個交易日（2 年）倍數 1.21 → 年化 10%；122 個交易日（約半年）倍數 1.1 → 年化約 21%。"""
    a = ex.summarize_x(fake_result(20_000, n=490), 20_000)
    expect(abs(a["cagr"] - 0.10) < 1e-9, a["cagr"])
    b = ex.summarize_x(fake_result(20_000, n=122, final_mult=1.1), 20_000)
    expect(abs(b["cagr"] - (1.1 ** (245 / 122) - 1)) < 1e-9, b["cagr"])


def t04_summarize_x_without_trades():
    """沒有任何交易：不報錯，勝率、平均報酬為 0，年化為 0。"""
    r = account.Result(np.full(245, 20000.0), np.full(245, 20000.0), np.zeros(245, dtype=int), [], [], 0.0, capital=20000.0)
    st = ex.summarize_x(r, 20000)
    expect(st["trades"] == 0 and st["win"] == 0.0 and st["avg_ret"] == 0.0 and abs(st["cagr"]) < 1e-12 and st["mdd"] == 0.0, st)


def run_flat(capital, n_slots, slip=0.0025):
    a = eng.mk(n_st=4, n_days=40)
    m = account.Market(**a)
    entry = np.zeros(a["O"].shape, dtype=bool)
    entry[3, :] = True
    elig = np.ones_like(entry)
    rank = np.tile(-np.arange(4, dtype="float64"), (40, 1))
    return account.simulate(m, entry, rank, elig, ex.make_cfg_x(n_slots, slip), 1, 40, capital)


def t05_capital_scaling():
    """平盤、同一組訊號：資金放大 100 倍，手續費與稅合計約放大 100 倍、期末倍數幾乎相同；資金小時整股取整造成的差異較大（2 萬元 3 檔每檔只買 66 股）。"""
    small, big = run_flat(20_000, 3), run_flat(2_000_000, 3)
    ratio = big.cost_total / small.cost_total
    expect(95 < ratio < 105, f"成本比 {ratio}")
    ms, mb = small.equity[-1] / 20_000, big.equity[-1] / 2_000_000
    expect(abs(mb - ms) < 0.003 and mb < 1.0, f"期末倍數 {ms} vs {mb}")
    expect(small.equity[0] == 20_000 and big.equity[0] == 2_000_000, "起始淨值 ＝ 本金")
    expect(small.n_held.max() == 3 and run_flat(2_000_000, 10).n_held.max() == 4, "持股數上限依設定（4 檔訊號、上限 10 時全買）")


def t06_report_lists_every_capital_and_marks_exploration():
    """報告：列出每種資金與持股數、含 0050 參考與隨機對照、聲明這是開發期探索而非通過／不通過判定。"""
    rows = []
    for cap in ex.CAPITALS:
        for n in ex.SLOTS:
            for ver in ex.VERSIONS:
                st = ex.summarize_x(fake_result(cap), cap)
                rows.append({"cap": cap, "n": n, "ver": ver, "s0": st, "s25": st})
    control = {(c, n): {"cagr_med": 0.01, "cagr_lo": -0.05, "cagr_hi": 0.06, "mdd_med": 0.5} for c in ex.CAPITALS for n in ex.SLOTS}
    text = ex.build_report(rows, control, bh=(33673.0, 0.558), period=("2007-01-01", "2016-12-30"))
    for cap in ex.CAPITALS:
        expect(f"{cap:,}" in text, f"缺少資金 {cap:,}")
    expect("探索" in text and "不是通過" in text and "0050" in text and "隨機" in text, "聲明或參考線缺少")
    expect("## 突破型" in text and "## 回檔型" in text, "兩個版本各有一張表")
    for cap in ex.CAPITALS:
        expect(text.count(f"| {cap:,} 元 |") == 6, f"資金 {cap:,} 應有 6 列（2 版本 × 3 持股數）：{text.count(f'| {cap:,} 元 |')}")
    expect(text.count("| 3 檔 |") == 8 and text.count("| 10 檔 |") == 8 and text.count("| 30 檔 |") == 8, "持股數各 8 列")


def t07_entry_script_dev_only_in_sandbox():
    """入口腳本沒有期間選項：--period val／final、--val-confirm 都被拒絕且不寫東西；沙盒（空資料庫）直接跑會失敗，不留 trials 或報告，也不寫帳本。"""
    for extra in (["--period", "val"], ["--period", "final"], ["--val-confirm"], ["--final-confirm"]):
        r, files = run_sandboxed("explore_capital.py", *extra)
        expect(r.returncode != 0 and files == [], f"{extra}: rc={r.returncode} files={files}")
    r, files = run_sandboxed("explore_capital.py")
    expect(r.returncode != 0 and "no such table" in r.stderr, r.stderr[-200:])
    expect(not any(f.endswith(("trials.csv", "explore_capital_dev_report.txt", "holdout_access.log")) for f in files), files)


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
    text = "\n".join([f"# 資金×持股數探索測試報告：{passed}／{len(TESTS)} 通過", *lines])
    out = ROOT / "data" / "experiments" / "explore_capital_tests_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
