"""下載驗證用的樣本股票，原樣存成 data/raw/<資料集>/*.json（尚未進 SQLite）。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))
import finmind  # noqa: E402

RAW = ROOT / "data" / "raw"
for sub in ("price", "dividend_result", "capital_reduction"):
    (RAW / sub).mkdir(parents=True, exist_ok=True)

SAMPLES = ["2330", "2327", "1204", "2384", "2456"]
START = "1994-10-01"
END = "2026-10-07"

# (資料集, 檔名前綴)；除權息與減資只抓有資料的區間，所以一律從 START 抓
DATASETS = [
    ("TaiwanStockPrice", "price"),
    ("TaiwanStockDividendResult", "dividend_result"),
    ("TaiwanStockCapitalReductionReferencePrice", "capital_reduction"),
]

for stock_id in SAMPLES:
    for dataset, prefix in DATASETS:
        out = RAW / prefix / f"{prefix}_{stock_id}.json"
        if out.exists():
            print("skip", out.name)
            continue
        rows = finmind.fetch(dataset, data_id=stock_id, start_date=START, end_date=END)
        out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"{dataset} {stock_id}: {len(rows)} rows")
