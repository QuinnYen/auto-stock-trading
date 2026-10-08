"""下載證交所「注意有價證券」與「處置有價證券」公告歷史（上市，2005 年起），原樣存成 JSON。

來源：證交所 rwd API（免費，可用日期區間查詢）。每季一個請求，檔名含區間。
含今天的最後一季每次都重新下載（資料會增加）；其他季已存在就跳過。
驗證：stat 必須為 OK，且 data 筆數必須等於 total，否則直接丟出例外。
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_all as da  # noqa: E402  共用 log

OUT = da.RAW / "alerts"
URL = "https://www.twse.com.tw/rwd/zh/announcement/{kind}"
KINDS = ("notice", "punish")
FIRST_YEAR = 2005
INTERVAL = 6.0   # 秒；證交所網站對連續請求會限制，放慢


def quarters(today: date):
    for year in range(FIRST_YEAR, today.year + 1):
        for q in range(4):
            start = date(year, 3 * q + 1, 1)
            if start > today:
                return
            end = date(year + (q == 3), (3 * q + 3) % 12 + 1, 1) - date.resolution
            yield start, min(end, today)


def fetch(kind: str, start: date, end: date) -> dict:
    resp = requests.get(URL.format(kind=kind), headers={"User-Agent": "Mozilla/5.0"}, timeout=60,
                        params={"startDate": f"{start:%Y%m%d}", "endDate": f"{end:%Y%m%d}", "response": "json"})
    resp.raise_for_status()
    body = resp.json()
    if body.get("stat") != "OK":
        raise RuntimeError(f"{kind} {start}~{end} 回應異常：{body.get('stat')}")
    if len(body["data"]) != body["total"]:
        raise RuntimeError(f"{kind} {start}~{end} 筆數不符：data {len(body['data'])} / total {body['total']}")
    return body


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    today = date.today()
    jobs = []
    for kind in KINDS:
        for start, end in quarters(today):
            path = OUT / f"{kind}_{start:%Y%m%d}_{end:%Y%m%d}.json"
            if end == today:
                for old in OUT.glob(f"{kind}_{start:%Y%m%d}_*.json"):
                    old.unlink()          # 舊的未完成季
            elif path.exists():
                continue
            jobs.append((kind, start, end, path))
    da.log(f"公告歷史下載啟動：待下載 {len(jobs)} 個區間")
    for i, (kind, start, end, path) in enumerate(jobs, 1):
        body = fetch(kind, start, end)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fields": body["fields"], "total": body["total"], "data": body["data"]},
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        print(f"[{i}/{len(jobs)}] {path.name}: {body['total']} 筆", flush=True)
        time.sleep(INTERVAL)
    da.log("公告歷史下載結束")


if __name__ == "__main__":
    main()
