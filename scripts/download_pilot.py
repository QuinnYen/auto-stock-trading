"""小範圍試跑：挑流動性最高的 200 檔，下載籌碼與基本面資料集，存成 data/raw/<資料集>/*.json。

選股依據：已下載的價格資料中，2010 年以來平均日成交金額最高、且至少有 1000 個交易日的 200 檔
（此為試跑用的粗略標準，含已下市股；選股清單存在 data/pilot_stocks.json）。
可續傳：已存在的檔案跳過。任何請求失敗都記錄後直接停止，不略過、不重試。
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_all as da  # noqa: E402  共用 log、set_status、wait_out_maintenance

ROOT = da.ROOT
DB = ROOT / "data" / "market.db"
PILOT_LIST = ROOT / "data" / "pilot_stocks.json"
N_STOCKS = 200

DATASETS = [
    ("TaiwanStockInstitutionalInvestorsBuySell", "institutional"),
    ("TaiwanStockMarginPurchaseShortSale", "margin"),
    ("TaiwanStockShareholding", "shareholding"),
    ("TaiwanStockSecuritiesLending", "securities_lending"),
    ("TaiwanStockPER", "per"),
    ("TaiwanStockMonthRevenue", "month_revenue"),
]


def pick_stocks() -> list[str]:
    if PILOT_LIST.exists():
        return json.loads(PILOT_LIST.read_text(encoding="utf-8"))
    con = sqlite3.connect(DB)
    ids = [r[0] for r in con.execute(
        """SELECT stock_id FROM stock_price
           WHERE date >= '2010-01-01' AND close > 0 AND stock_id GLOB '[1-9][0-9][0-9][0-9]'
           GROUP BY stock_id HAVING COUNT(*) >= 1000
           ORDER BY AVG(money) DESC LIMIT ?""", (N_STOCKS,))]
    con.close()
    PILOT_LIST.write_text(json.dumps(ids), encoding="utf-8")
    return ids


def main() -> None:
    ids = pick_stocks()
    jobs = [(s, d, p) for s in ids for d, p in DATASETS
            if not (da.RAW / p / f"{p}_{s}.json").exists()]
    da.log(f"試跑啟動：{len(ids)} 檔，待下載 {len(jobs)} 個檔案")

    for i, (stock_id, dataset, prefix) in enumerate(jobs, 1):
        out = da.RAW / prefix / f"{prefix}_{stock_id}.json"
        da.wait_out_maintenance()
        try:
            rows = da.finmind.fetch(dataset, data_id=stock_id, start_date=da.START, end_date=da.END)
        except Exception as e:
            da.log(f"FAIL [{i}/{len(jobs)}] {dataset} {stock_id}: {e}")
            raise
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out)
        da.set_status(i, len(jobs), f"{stock_id} {prefix}", 0)

    da.log("試跑結束")


if __name__ == "__main__":
    main()
