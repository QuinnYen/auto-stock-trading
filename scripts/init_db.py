"""建立 SQLite 資料庫 data/market.db，並把 data/raw 與 data/shadow 的現有檔案匯入。

原始價格不做還原（還原價在讀取時用事件表推算）；零成交日（close = 0）也照原樣存入。
可重複執行：有主鍵的表用 INSERT OR REPLACE，快照類表先清空再寫入。
任何不一致（代號與檔名不符、檔案缺欄位）都直接丟出例外。
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ANN = ROOT / "data" / "shadow" / "announcements"
ALERTS = RAW / "alerts"
DB = ROOT / "data" / "market.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS stock_price (
    stock_id TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, spread REAL,
    volume INTEGER, money INTEGER, turnover REAL,
    PRIMARY KEY (stock_id, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS dividend_result (
    stock_id TEXT NOT NULL, date TEXT NOT NULL,
    before_price REAL, after_price REAL, dividend REAL, dividend_type TEXT,
    max_price REAL, min_price REAL, open_price REAL, reference_price REAL,
    PRIMARY KEY (stock_id, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS capital_reduction (
    stock_id TEXT NOT NULL, date TEXT NOT NULL,
    last_close REAL, post_reduction_ref_price REAL, limit_up REAL, limit_down REAL,
    opening_ref_price REAL, exright_ref_price REAL, reason TEXT,
    PRIMARY KEY (stock_id, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS split_price (
    stock_id TEXT NOT NULL, date TEXT NOT NULL, type TEXT,
    before_price REAL, after_price REAL, max_price REAL, min_price REAL, open_price REAL,
    PRIMARY KEY (stock_id, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS par_value_change (
    stock_id TEXT NOT NULL, date TEXT NOT NULL, stock_name TEXT,
    before_close REAL, after_ref_close REAL, after_ref_max REAL, after_ref_min REAL, after_ref_open REAL,
    PRIMARY KEY (stock_id, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS trading_date (date TEXT PRIMARY KEY) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS stock_info (
    stock_id TEXT NOT NULL, stock_name TEXT, type TEXT, industry_category TEXT, date TEXT
);
CREATE INDEX IF NOT EXISTS idx_stock_info_id ON stock_info (stock_id);

CREATE TABLE IF NOT EXISTS delisting (date TEXT, stock_id TEXT, stock_name TEXT);
CREATE INDEX IF NOT EXISTS idx_delisting_id ON delisting (stock_id);

CREATE TABLE IF NOT EXISTS institutional (
    stock_id TEXT NOT NULL, date TEXT NOT NULL, name TEXT NOT NULL, buy INTEGER, sell INTEGER,
    PRIMARY KEY (stock_id, date, name)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS margin (
    stock_id TEXT NOT NULL, date TEXT NOT NULL,
    margin_buy INTEGER, margin_cash_repayment INTEGER, margin_limit INTEGER, margin_sell INTEGER,
    margin_today_balance INTEGER, margin_yesterday_balance INTEGER, note TEXT, offset_loan_and_short INTEGER,
    short_buy INTEGER, short_cash_repayment INTEGER, short_limit INTEGER, short_sell INTEGER,
    short_today_balance INTEGER, short_yesterday_balance INTEGER,
    PRIMARY KEY (stock_id, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS shareholding (
    stock_id TEXT NOT NULL, date TEXT NOT NULL, stock_name TEXT, international_code TEXT,
    foreign_remaining_shares INTEGER, foreign_shares INTEGER, foreign_remain_ratio REAL,
    foreign_shares_ratio REAL, foreign_upper_limit_ratio REAL, chinese_upper_limit_ratio REAL,
    shares_issued INTEGER, recently_declare_date TEXT, note TEXT,
    PRIMARY KEY (stock_id, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS securities_lending (
    stock_id TEXT NOT NULL, date TEXT NOT NULL, transaction_type TEXT, volume INTEGER, fee_rate REAL,
    close REAL, original_return_date TEXT, original_lending_period INTEGER
);
CREATE INDEX IF NOT EXISTS idx_securities_lending ON securities_lending (stock_id, date);

CREATE TABLE IF NOT EXISTS per (
    stock_id TEXT NOT NULL, date TEXT NOT NULL, dividend_yield REAL, per REAL, pbr REAL,
    PRIMARY KEY (stock_id, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS month_revenue (
    stock_id TEXT NOT NULL, date TEXT NOT NULL, country TEXT, revenue INTEGER,
    revenue_month INTEGER, revenue_year INTEGER, create_time TEXT,
    PRIMARY KEY (stock_id, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS attention (
    stock_id TEXT NOT NULL, stock_name TEXT, date TEXT NOT NULL, cumulative INTEGER, info TEXT,
    close REAL, per REAL
);
CREATE INDEX IF NOT EXISTS idx_attention ON attention (stock_id, date);

CREATE TABLE IF NOT EXISTS disposition (
    stock_id TEXT NOT NULL, stock_name TEXT, announce_date TEXT, cumulative INTEGER, condition TEXT,
    period_start TEXT NOT NULL, period_end TEXT NOT NULL, measure TEXT, content TEXT
);
CREATE INDEX IF NOT EXISTS idx_disposition ON disposition (stock_id, period_start);

CREATE TABLE IF NOT EXISTS announcement (
    report_date TEXT NOT NULL, seq INTEGER NOT NULL,
    speak_date TEXT, speak_time TEXT, stock_id TEXT, stock_name TEXT,
    subject TEXT, clause TEXT, fact_date TEXT, content TEXT,
    event_type TEXT, direction TEXT, is_routine INTEGER, summary TEXT,
    model TEXT, prompt_version TEXT, input_tokens INTEGER, output_tokens INTEGER,
    PRIMARY KEY (report_date, seq)
) WITHOUT ROWID;
"""


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def import_per_stock(con, folder: str, prefix: str, table: str, columns: list[str], keys: list[str]) -> int:
    """data/raw/<folder>/<prefix>_<代號>.json，每列的 stock_id 必須與檔名一致。"""
    sql = f"INSERT OR REPLACE INTO {table} VALUES ({','.join('?' * len(columns))})"
    total = 0
    for path in sorted((RAW / folder).glob(f"{prefix}_*.json")):
        stock_id = path.stem[len(prefix) + 1:]
        rows = load(path)
        for r in rows:
            if r["stock_id"] != stock_id:
                raise RuntimeError(f"{path.name} 內有不同代號的資料：{r['stock_id']}")
        con.execute(f"DELETE FROM {table} WHERE stock_id = ?", (stock_id,))
        con.executemany(sql, [tuple(r[k] for k in keys) for r in rows])
        total += len(rows)
    return total


def import_meta(con, name: str, table: str, keys: list[str], clear: bool) -> int:
    rows = load(RAW / "meta" / f"{name}.json")
    if clear:
        con.execute(f"DELETE FROM {table}")
    sql = f"INSERT OR REPLACE INTO {table} VALUES ({','.join('?' * len(keys))})"
    con.executemany(sql, [tuple(r[k] for k in keys) for r in rows])
    return len(rows)


def roc_date(text: str) -> str:
    """'114.01.06' 或 '113/12/25' -> '2025-01-06' / '2024-12-25'"""
    y, m, d = re.split(r"[./]", text.strip())
    return f"{int(y) + 1911}-{int(m):02d}-{int(d):02d}"


def to_float(text):
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def import_alerts(con) -> tuple[int, int]:
    """注意股、處置股公告（整批清空後重寫；檔案小）。"""
    con.execute("DELETE FROM attention")
    con.execute("DELETE FROM disposition")
    n_att = n_dis = 0
    for path in sorted(ALERTS.glob("notice_*.json")):
        rows = [(r[1], r[2], roc_date(r[5]), int(r[3]), strip_tags(r[4]), to_float(r[6]), to_float(r[7]))
                for r in load(path)["data"]]
        con.executemany("INSERT INTO attention VALUES (?,?,?,?,?,?,?)", rows)
        n_att += len(rows)
    for path in sorted(ALERTS.glob("punish_*.json")):
        rows = []
        for r in load(path)["data"]:
            start, end = (roc_date(x) for x in r[6].split("～"))
            rows.append((r[2], r[3], roc_date(r[1]), int(r[4]), r[5], start, end, r[7], strip_tags(r[8])))
        con.executemany("INSERT INTO disposition VALUES (?,?,?,?,?,?,?,?,?)", rows)
        n_dis += len(rows)
    return n_att, n_dis


def import_announcements(con) -> int:
    total = 0
    for path in sorted(ANN.glob("*.json")):
        d = load(path)
        rows = []
        for seq, item in enumerate(d["items"]):
            a, ai, u = item["announcement"], item["ai"], item["usage"]
            rows.append((
                d["date"], seq, a["發言日期"], a["發言時間"], a["公司代號"], a["公司名稱"],
                a["主旨"], a["符合條款"], a["事實發生日"], a["說明"],
                ai["event_type"], ai["direction"], int(ai["is_routine"]), ai["summary"],
                d["model"], d["prompt_version"], u["input_tokens"], u["output_tokens"],
            ))
        con.executemany(f"INSERT OR REPLACE INTO announcement VALUES ({','.join('?' * 18)})", rows)
        total += len(rows)
    return total


def main() -> None:
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    with con:  # 單一交易，失敗整批復原
        counts = {
            "stock_price": import_per_stock(
                con, "price", "price", "stock_price",
                ["stock_id", "date", "open", "high", "low", "close", "spread", "volume", "money", "turnover"],
                ["stock_id", "date", "open", "max", "min", "close", "spread",
                 "Trading_Volume", "Trading_money", "Trading_turnover"]),
            "dividend_result": import_per_stock(
                con, "dividend_result", "dividend_result", "dividend_result",
                ["stock_id", "date", "before_price", "after_price", "dividend", "dividend_type",
                 "max_price", "min_price", "open_price", "reference_price"],
                ["stock_id", "date", "before_price", "after_price", "stock_and_cache_dividend",
                 "stock_or_cache_dividend", "max_price", "min_price", "open_price", "reference_price"]),
            "capital_reduction": import_per_stock(
                con, "capital_reduction", "capital_reduction", "capital_reduction",
                ["stock_id", "date", "last_close", "post_reduction_ref_price", "limit_up", "limit_down",
                 "opening_ref_price", "exright_ref_price", "reason"],
                ["stock_id", "date", "ClosingPriceonTheLastTradingDay", "PostReductionReferencePrice",
                 "LimitUp", "LimitDown", "OpeningReferencePrice", "ExrightReferencePrice",
                 "ReasonforCapitalReduction"]),
            "institutional": import_per_stock(
                con, "institutional", "institutional", "institutional",
                ["stock_id", "date", "name", "buy", "sell"], ["stock_id", "date", "name", "buy", "sell"]),
            "margin": import_per_stock(
                con, "margin", "margin", "margin",
                ["stock_id", "date", "margin_buy", "margin_cash_repayment", "margin_limit", "margin_sell",
                 "margin_today_balance", "margin_yesterday_balance", "note", "offset_loan_and_short",
                 "short_buy", "short_cash_repayment", "short_limit", "short_sell",
                 "short_today_balance", "short_yesterday_balance"],
                ["stock_id", "date", "MarginPurchaseBuy", "MarginPurchaseCashRepayment", "MarginPurchaseLimit",
                 "MarginPurchaseSell", "MarginPurchaseTodayBalance", "MarginPurchaseYesterdayBalance", "Note",
                 "OffsetLoanAndShort", "ShortSaleBuy", "ShortSaleCashRepayment", "ShortSaleLimit",
                 "ShortSaleSell", "ShortSaleTodayBalance", "ShortSaleYesterdayBalance"]),
            "shareholding": import_per_stock(
                con, "shareholding", "shareholding", "shareholding",
                ["stock_id", "date", "stock_name", "international_code", "foreign_remaining_shares",
                 "foreign_shares", "foreign_remain_ratio", "foreign_shares_ratio", "foreign_upper_limit_ratio",
                 "chinese_upper_limit_ratio", "shares_issued", "recently_declare_date", "note"],
                ["stock_id", "date", "stock_name", "InternationalCode", "ForeignInvestmentRemainingShares",
                 "ForeignInvestmentShares", "ForeignInvestmentRemainRatio", "ForeignInvestmentSharesRatio",
                 "ForeignInvestmentUpperLimitRatio", "ChineseInvestmentUpperLimitRatio", "NumberOfSharesIssued",
                 "RecentlyDeclareDate", "note"]),
            "securities_lending": import_per_stock(
                con, "securities_lending", "securities_lending", "securities_lending",
                ["stock_id", "date", "transaction_type", "volume", "fee_rate", "close",
                 "original_return_date", "original_lending_period"],
                ["stock_id", "date", "transaction_type", "volume", "fee_rate", "close",
                 "original_return_date", "original_lending_period"]),
            "per": import_per_stock(
                con, "per", "per", "per",
                ["stock_id", "date", "dividend_yield", "per", "pbr"],
                ["stock_id", "date", "dividend_yield", "PER", "PBR"]),
            "month_revenue": import_per_stock(
                con, "month_revenue", "month_revenue", "month_revenue",
                ["stock_id", "date", "country", "revenue", "revenue_month", "revenue_year", "create_time"],
                ["stock_id", "date", "country", "revenue", "revenue_month", "revenue_year", "create_time"]),
            "split_price": import_meta(
                con, "TaiwanStockSplitPrice", "split_price",
                ["stock_id", "date", "type", "before_price", "after_price", "max_price", "min_price",
                 "open_price"], clear=False),
            "par_value_change": import_meta(
                con, "TaiwanStockParValueChange", "par_value_change",
                ["stock_id", "date", "stock_name", "before_close", "after_ref_close", "after_ref_max",
                 "after_ref_min", "after_ref_open"], clear=False),
            "trading_date": import_meta(con, "TaiwanStockTradingDate", "trading_date", ["date"], clear=False),
            "stock_info": import_meta(
                con, "TaiwanStockInfo", "stock_info",
                ["stock_id", "stock_name", "type", "industry_category", "date"], clear=True),
            "delisting": import_meta(
                con, "TaiwanStockDelisting", "delisting", ["date", "stock_id", "stock_name"], clear=True),
            "announcement": import_announcements(con),
        }
        counts["attention"], counts["disposition"] = import_alerts(con)

    print("資料表        檔案列數  資料庫列數")
    for table, n_files in counts.items():
        n_db = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        flag = "" if n_db == n_files else "  <-- 不一致"
        print(f"{table:<18}{n_files:>8}{n_db:>10}{flag}")
    con.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
