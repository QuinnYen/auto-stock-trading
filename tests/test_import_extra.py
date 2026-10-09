"""補下載資料（股利政策、月營收、融資融券）匯入 SQLite 的測試。

在暫存資料庫上執行完整的 init_db.main()，不動 data/market.db。檢查：每張表列數與檔案一致、欄位逐格與原始檔相同、
重複執行結果相同、每檔資料先刪後寫（不會累加）。下載未完成時，只檢查目前已存在的檔案。

用法：python tests/test_import_extra.py        報告：data/experiments/import_extra_tests_report.txt
"""
import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import init_db  # noqa: E402

# (資料夾, 資料表, 資料表欄位 → 原始檔欄位)；只列需要逐格比對的欄位
CHECKS = {
    "dividend_policy": {
        "year": "year", "cash_earnings_distribution": "CashEarningsDistribution",
        "cash_statutory_surplus": "CashStatutorySurplus", "cash_ex_dividend_trading_date": "CashExDividendTradingDate",
        "stock_earnings_distribution": "StockEarningsDistribution", "announcement_date": "AnnouncementDate",
        "announcement_time": "AnnouncementTime", "cash_increase_subscription_price": "CashIncreaseSubscriptionpRrice",
        "participate_distribution_of_total_shares": "ParticipateDistributionOfTotalShares",
        "total_employee_cash_dividend": "TotalEmployeeCashDividend", "date": "date",
    },
    "month_revenue": {"revenue": "revenue", "revenue_month": "revenue_month", "revenue_year": "revenue_year",
                      "create_time": "create_time", "date": "date"},
    "margin": {"margin_today_balance": "MarginPurchaseTodayBalance", "short_today_balance": "ShortSaleTodayBalance",
               "margin_buy": "MarginPurchaseBuy", "short_sell": "ShortSaleSell", "date": "date"},
}

_state = {}


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def files(table):
    """只取匯入開始前就已存在的檔案（下載進行中時，之後才出現的檔案不一定被匯入，不列入檢查）。"""
    return sorted(p for p in (init_db.RAW / table).glob(f"{table}_*.json") if p.stat().st_mtime <= _state["cutoff"])


def db_count(con, table):
    ids = [p.stem[len(table) + 1:] for p in files(table)]
    return sum(con.execute(f"SELECT COUNT(*) FROM {table} WHERE stock_id IN ({','.join('?' * len(chunk))})", chunk).fetchone()[0]
               for chunk in (ids[i:i + 500] for i in range(0, len(ids), 500)))


def setup():
    _state["cutoff"] = time.time()
    tmp = tempfile.TemporaryDirectory()
    init_db.DB = Path(tmp.name) / "t.db"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        init_db.main()
    _state["tmp"], _state["out"] = tmp, buf.getvalue()
    _state["db"] = init_db.DB


def t01_row_counts_match_files():
    """三張表的資料庫列數 ＝ 所有檔案列數合計；匯入輸出沒有『不一致』。"""
    con = sqlite3.connect(_state["db"])
    for table in CHECKS:
        n_files = sum(len(json.loads(p.read_text(encoding="utf-8"))) for p in files(table))
        n_db = db_count(con, table)
        expect(n_files == n_db, f"{table}：檔案 {n_files} ≠ 資料庫 {n_db}")
    bad = [ln for ln in _state["out"].splitlines() if "不一致" in ln and ln.split()[0] in CHECKS]
    expect(not bad, f"匯入輸出出現不一致：{bad}")
    con.close()


def t02_cells_match_raw():
    """每張表抽樣：前、中、後各檔的每一列，指定欄位逐格與原始檔相同。"""
    con = sqlite3.connect(_state["db"])
    con.row_factory = sqlite3.Row
    for table, cols in CHECKS.items():
        fs = files(table)
        if not fs:
            continue
        for p in (fs[0], fs[len(fs) // 2], fs[-1]):
            sid = p.stem[len(table) + 1:]
            raw = json.loads(p.read_text(encoding="utf-8"))
            db = con.execute(f"SELECT * FROM {table} WHERE stock_id = ? ORDER BY date" + (
                ", announcement_date, year" if table == "dividend_policy" else ""), (sid,)).fetchall()
            expect(len(raw) == len(db), f"{table} {sid}：列數 {len(raw)} ≠ {len(db)}")
            raw.sort(key=lambda r: ((r["date"], r["AnnouncementDate"], r["year"]) if table == "dividend_policy" else (r["date"],)))
            for r, d in zip(raw, db):
                for dc, rc in cols.items():
                    expect(d[dc] == r[rc], f"{table} {sid} {r['date']} {dc}：資料庫 {d[dc]!r} ≠ 原始 {r[rc]!r}")
    con.close()


def t03_rerun_is_idempotent():
    """第二次執行匯入：各表列數仍等於檔案列數合計（逐檔先刪後寫，不累加；下載進行中檔案會增加，故以當下檔案為準）。"""
    _state["cutoff"] = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        init_db.main()
    con = sqlite3.connect(_state["db"])
    for table in CHECKS:
        n_files = sum(len(json.loads(p.read_text(encoding="utf-8"))) for p in files(table))
        n_db = db_count(con, table)
        expect(n_files == n_db, f"重複匯入後 {table}：檔案 {n_files} ≠ 資料庫 {n_db}")
    con.close()


def t04_stock_id_mismatch_stops():
    """檔名代號與內容不符時中止（拿一份股利檔改名放進暫存資料夾）。"""
    fs = files("dividend_policy")
    src = next(p for p in fs if json.loads(p.read_text(encoding="utf-8")))
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / "dividend_policy"
        folder.mkdir()
        (folder / "dividend_policy_9999.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        old = init_db.RAW
        init_db.RAW = Path(d)
        try:
            con = sqlite3.connect(":memory:")
            con.executescript(init_db.SCHEMA)
            try:
                init_db.import_per_stock(con, "dividend_policy", "dividend_policy", "dividend_policy", ["stock_id"], ["stock_id"])
            except RuntimeError:
                return
            raise AssertionError("代號不符應丟出例外")
        finally:
            init_db.RAW = old


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t") and k[1:3].isdigit() and callable(v)]


def main():
    lines, passed = [], 0
    setup()
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
    n = {t: len(files(t)) for t in CHECKS}
    text = "\n".join([f"# 補下載資料匯入測試報告：{passed}／{len(TESTS)} 通過（測試時檔案數：{n}）", *lines])
    out = ROOT / "data" / "experiments" / "import_extra_tests_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    _state["tmp"].cleanup()
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
