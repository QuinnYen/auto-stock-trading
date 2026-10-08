"""FinMind API 的最小封裝：讀 token、呼叫 v4/data、遵守流量限制。"""
import os
import time
from pathlib import Path

import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
MIN_INTERVAL = 6.5  # 秒；免費會員 600 次/小時 = 每 6 秒一次，留一點餘裕

_last_call = 0.0


def _load_token() -> str:
    token = os.environ.get("FINMIND_TOKEN")
    if token:
        return token
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "FINMIND_TOKEN":
            return value.strip()
    raise RuntimeError("找不到 FINMIND_TOKEN（環境變數或 .env）")


def fetch(dataset: str, **params) -> list[dict]:
    """呼叫 FinMind，回傳資料列。失敗時丟出例外（不吞錯誤）。"""
    global _last_call
    wait = MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()

    resp = requests.get(
        API_URL,
        params={"dataset": dataset, **params},
        headers={"Authorization": f"Bearer {_load_token()}"},
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != 200:
        raise RuntimeError(f"FinMind 回應錯誤 dataset={dataset}: {body.get('msg')}")
    return body["data"]
