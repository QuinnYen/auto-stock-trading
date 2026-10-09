"""補下載：全市場（資料庫中有價格的 4 位數普通股）的股利政策、月營收、融資融券，存成 data/raw/<資料集>/*.json。

供 H1（營收）、H9（融資券）、H10（股利）假設使用。名單取 market.db 中有成交價的 4 位數普通股（空價格檔不下載）。
可續傳：已存在的檔案跳過（試跑已下載的 200 檔月營收與融資券不重抓）。任何請求失敗直接停止，不略過、不重試。
"""
import sqlite3
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_all as da  # noqa: E402  共用 log、set_status、wait_out_maintenance

DB = da.ROOT / "data" / "market.db"

# 小的先下載，大的最後
DATASETS = [
    ("TaiwanStockDividend", "dividend_policy"),
    ("TaiwanStockMonthRevenue", "month_revenue"),
    ("TaiwanStockMarginPurchaseShortSale", "margin"),
]


def stock_ids() -> list[str]:
    con = sqlite3.connect(DB)
    ids = [r[0] for r in con.execute(
        """SELECT stock_id FROM stock_price
           WHERE close > 0 AND stock_id GLOB '[1-9][0-9][0-9][0-9]'
           GROUP BY stock_id ORDER BY stock_id""")]
    con.close()
    return ids


def main() -> None:
    da.finmind.on_event = da.log
    ids = stock_ids()
    jobs = [(s, d, p) for d, p in DATASETS for s in ids
            if not (da.RAW / p / f"{p}_{s}.json").exists()]
    da.log(f"補下載啟動：{len(ids)} 檔，待下載 {len(jobs)} 個檔案")

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

    da.log("補下載結束")


if __name__ == "__main__":
    main()
