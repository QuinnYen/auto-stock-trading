"""下載全市場歷史資料（現有上市 + 全部已下市），原樣存成 data/raw/<資料集>/*.json。

可斷點續傳：已存在的檔案直接跳過；暫時性錯誤由 finmind.fetch 重試，仍失敗就停止（不寫檔），重跑即會補上。
"""
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))
import finmind  # noqa: E402

RAW = ROOT / "data" / "raw"
START = "2003-01-01"
END = "2026-10-07"

DATASETS = [
    ("TaiwanStockPrice", "price"),
    ("TaiwanStockDividendResult", "dividend_result"),
    ("TaiwanStockCapitalReductionReferencePrice", "capital_reduction"),
]

# 全市場一次取得、免費、不帶 data_id 的事件表：(資料集, 額外參數)
META_DATASETS = [
    ("TaiwanStockSplitPrice", {}),
    ("TaiwanStockParValueChange", {"start_date": "2020-01-01"}),
]

# 普通股的粗略篩法：4 位數且非 0 開頭（91xx 存託憑證會混入，之後再處理）
COMMON = re.compile(r"[1-9]\d{3}")


TAIPEI = timezone(timedelta(hours=8))
LOG = ROOT / "data" / "logs" / "download.log"        # 只記事件（啟動、失敗、維護、結束），附加寫入
STATUS = ROOT / "data" / "logs" / "download.status"  # 一行進度，每次覆蓋


def log(msg: str) -> None:
    line = f"{datetime.now(TAIPEI):%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_status(done: int, total: int, current: str, failed: int) -> None:
    now = datetime.now(TAIPEI)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(f"{now:%Y-%m-%d %H:%M:%S} 進度 {done}/{total} 目前 {current} 失敗 {failed}\n",
                      encoding="utf-8")


def wait_out_maintenance() -> None:
    """FinMind 每週日 00:00～03:00（台灣時間）維護，這段時間直接睡到 03:00 再繼續。"""
    now = datetime.now(TAIPEI)
    if now.weekday() == 6 and now.hour < 3:
        resume = now.replace(hour=3, minute=0, second=30, microsecond=0)
        log(f"FinMind 維護中，等到 {resume:%H:%M}")
        time.sleep((resume - now).total_seconds())


def universe() -> list[str]:
    info = json.loads((RAW / "meta" / "TaiwanStockInfo.json").read_text(encoding="utf-8"))
    delisted = json.loads((RAW / "meta" / "TaiwanStockDelisting.json").read_text(encoding="utf-8"))
    ids = {r["stock_id"] for r in info if r["type"] == "twse" and COMMON.fullmatch(r["stock_id"])}
    ids |= {r["stock_id"] for r in delisted if COMMON.fullmatch(r["stock_id"])}
    return sorted(ids)


def download_meta() -> None:
    for dataset, params in META_DATASETS:
        out = RAW / "meta" / f"{dataset}.json"
        if out.exists():
            continue
        wait_out_maintenance()
        rows = finmind.fetch(dataset, **params)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out)
        log(f"{dataset}: {len(rows)} rows")


def main() -> None:
    finmind.on_event = log
    download_meta()
    ids = universe()
    jobs = [(s, d, p) for s in ids for d, p in DATASETS if not (RAW / p / f"{p}_{s}.json").exists()]
    log(f"啟動：{len(ids)} 檔，待下載 {len(jobs)} 個檔案")

    for i, (stock_id, dataset, prefix) in enumerate(jobs, 1):
        out = RAW / prefix / f"{prefix}_{stock_id}.json"
        wait_out_maintenance()
        try:
            rows = finmind.fetch(dataset, data_id=stock_id, start_date=START, end_date=END)
        except Exception as e:  # finmind.fetch 已依錯誤類型重試過；仍失敗就停止（不寫檔，重跑會續傳）
            log(f"FAIL [{i}/{len(jobs)}] {dataset} {stock_id}: {e}")
            raise
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out)
        set_status(i, len(jobs), stock_id, 0)

    log("結束")


if __name__ == "__main__":
    main()
