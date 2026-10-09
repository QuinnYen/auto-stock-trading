"""驗證集防護（src/backtest/holdout.py）的測試。

用法：python tests/test_holdout.py        報告：data/experiments/holdout_tests_report.txt
"""
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "backtest"))
sys.path.insert(0, str(ROOT / "tests"))
import holdout  # noqa: E402
from sandbox import run_sandboxed  # noqa: E402


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def blocked(period, **kw):
    with tempfile.TemporaryDirectory() as d:
        ledger = Path(d) / "x.log"
        try:
            holdout.require_access(period, "t", ledger=ledger, **kw)
        except SystemExit:
            return True, ledger.exists()
        return False, ledger.exists()


def t01_dev_passes_without_ledger():
    """開發期不需確認，也不寫帳本。"""
    stopped, wrote = blocked("dev")
    expect(not stopped and not wrote, f"dev 應直接放行且不寫帳本：{stopped}, {wrote}")


def t02_val_requires_val_confirm():
    """驗證期未確認就中止；只給 final 確認也不行。"""
    for kw in ({}, {"final_confirm": True}):
        stopped, wrote = blocked("val", **kw)
        expect(stopped and not wrote, f"val 應中止且不寫帳本：{kw}")


def t03_final_requires_final_confirm():
    """最後測試期未確認就中止；只給 val 確認也不行。"""
    for kw in ({}, {"val_confirm": True}):
        stopped, wrote = blocked("final", **kw)
        expect(stopped and not wrote, f"final 應中止且不寫帳本：{kw}")


def t04_confirmed_access_is_logged():
    """確認後放行，帳本多一行（時間、腳本、期間），再存取會附加而非覆蓋。"""
    with tempfile.TemporaryDirectory() as d:
        ledger = Path(d) / "sub" / "x.log"
        holdout.require_access("val", "scriptA", val_confirm=True, ledger=ledger)
        holdout.require_access("final", "scriptB", final_confirm=True, ledger=ledger)
        rows = [r.split("\t") for r in ledger.read_text(encoding="utf-8").splitlines()]
        expect(len(rows) == 2, f"應有 2 行：{rows}")
        expect(rows[0][1:] == ["scriptA", "val"] and rows[1][1:] == ["scriptB", "final"], f"內容不符：{rows}")


def t05_unknown_period_stops():
    """未知期間名稱中止。"""
    stopped, _ = blocked("test")
    expect(stopped, "未知期間應中止")


def t06_periods_do_not_overlap():
    """三個期間首尾相接、不重疊。"""
    d, v, f = (holdout.PERIODS[k] for k in ("dev", "val", "final"))
    expect(d[1] < v[0] and v[1] < f[0], "期間不應重疊")
    expect(d[0] == "2007-01-01" and d[1] == "2016-12-31" and v == ("2017-01-01", "2021-12-31") and f[0] == "2022-01-01",
           "期間日期與事前約定不符")


def t07_entry_points_are_wired():
    """兩個有期間選項的入口腳本實際被擋下：不帶確認旗標跑 val／final 會立刻中止（在沙盒中執行：即使防護失效也只會碰到空資料庫）。"""
    for script in ("strong_stocks.py", "account_sim.py"):
        for period in ("val", "final"):
            r, written = run_sandboxed(script, "--period", period)
            expect(r.returncode != 0 and "--" in (r.stderr + r.stdout), f"{script} --period {period} 應被擋下：{r.returncode}")
            expect("載入資料" not in r.stdout, f"{script} --period {period} 不應開始載入資料")
            expect(not [w for w in written if "holdout_access" in w], f"{script} --period {period} 被擋下的嘗試不應寫入帳本")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t") and k[1:3].isdigit() and callable(v)]


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
    text = "\n".join([f"# 驗證集防護測試報告：{passed}／{len(TESTS)} 通過", *lines])
    out = ROOT / "data" / "experiments" / "holdout_tests_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
