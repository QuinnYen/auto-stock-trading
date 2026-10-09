"""FinMind API 的最小封裝：讀 token、呼叫 v4/data、遵守流量限制、依錯誤類型決定重試或停止。

重試規則（FinMind 會暫時封鎖短時間內大量 4xx 的 IP，所以只對暫時性錯誤重試，且次數有限）：
  5xx／逾時／連線中斷：最多重試 3 次（30、120、300 秒）；403 ip banned：睡 31 分鐘後只試一次；
  402 超額：睡到下個整點後只試一次；其他 4xx、格式錯誤：立刻丟出例外；每次執行累計重試超過上限也丟出例外。
"""
import os
import time
from datetime import datetime
from pathlib import Path

import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
MIN_INTERVAL = 6.5  # 秒；免費會員 600 次/小時 = 每 6 秒一次，留一點餘裕

RETRY_DELAYS = (30, 120, 300)   # 5xx／網路錯誤的重試間隔（秒），長度即最多重試次數
MAX_RETRIES_PER_RUN = 10        # 每次執行（每個行程）累計重試上限，避免持續故障時累積大量失敗請求
BAN_WAIT = 31 * 60              # 403 ip banned：封鎖 30 分鐘後自動解除

_last_call = 0.0
_retries_used = 0
_get = requests.get             # 測試時可替換
_sleep = time.sleep
on_event = print                # 重試與等待事件的輸出；下載腳本會換成寫入日誌的函式


def _load_token() -> str:
    token = os.environ.get("FINMIND_TOKEN")
    if token:
        return token
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "FINMIND_TOKEN":
            return value.strip()
    raise RuntimeError("找不到 FINMIND_TOKEN（環境變數或 .env）")


def _secs_to_next_hour(now: datetime) -> int:
    """到下個整點的秒數，再加 30 秒緩衝。"""
    return 3600 - (now.minute * 60 + now.second) + 30


def _request(dataset: str, params: dict) -> tuple[int, str, list | None]:
    """送出一次請求，回傳 (狀態碼, 訊息, 資料)；網路層錯誤視為 599。內容不是 JSON 時直接丟出例外。"""
    try:
        resp = _get(API_URL, params={"dataset": dataset, **params},
                    headers={"Authorization": f"Bearer {_load_token()}"}, timeout=60)
    except (requests.Timeout, requests.ConnectionError) as e:
        return 599, f"{type(e).__name__}: {e}", None
    try:
        body = resp.json()
    except ValueError:
        if resp.status_code >= 400:
            return resp.status_code, f"HTTP {resp.status_code}（內容不是 JSON）", None
        raise RuntimeError(f"FinMind 回應不是 JSON dataset={dataset}")
    code = resp.status_code if resp.status_code >= 400 else body.get("status")
    if code == 200:
        return 200, "", body["data"]
    return code, str(body.get("msg")), None


def fetch(dataset: str, **params) -> list[dict]:
    """呼叫 FinMind，回傳資料列。暫時性錯誤依上述規則重試；其他錯誤丟出例外（不吞錯誤）。"""
    global _last_call, _retries_used
    retries, banned, over_quota = 0, False, False
    while True:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            _sleep(wait)
        _last_call = time.monotonic()

        code, msg, data = _request(dataset, params)
        if code == 200:
            return data
        err = RuntimeError(f"FinMind 錯誤 {code} dataset={dataset} {params.get('data_id', '')}: {msg}")
        if code == 403 and not banned:
            banned, delay = True, BAN_WAIT
        elif code == 402 and not over_quota:
            over_quota, delay = True, _secs_to_next_hour(datetime.now())
        elif (code == 599 or (isinstance(code, int) and code >= 500))                 and retries < len(RETRY_DELAYS) and _retries_used < MAX_RETRIES_PER_RUN:
            delay = RETRY_DELAYS[retries]
            retries += 1
            _retries_used += 1
        else:
            raise err
        on_event(f"RETRY 等待 {delay} 秒後重試：{err}")
        _sleep(delay)
