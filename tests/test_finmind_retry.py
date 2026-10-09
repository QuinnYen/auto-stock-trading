"""FinMind 錯誤處理與重試規則的測試（用假的伺服器回應與假的睡眠，不連網、不真的等待）。

規則：5xx／逾時／連線中斷 → 最多重試 3 次（30 秒、120 秒、300 秒）；403 ip banned → 睡 31 分鐘後只試一次；
402 超額 → 睡到下個整點後只試一次；其他 4xx 與格式錯誤 → 立刻停；每次執行累計重試超過上限 → 停。

用法：python tests/test_finmind_retry.py        報告：data/experiments/finmind_retry_tests_report.txt
"""
import sys
import traceback
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))
import finmind  # noqa: E402


class Resp:
    def __init__(self, status=200, body=None, bad_json=False):
        self.status_code, self._body, self._bad = status, body if body is not None else {"status": 200, "data": [{"a": 1}]}, bad_json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error for url: x", response=self)

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._body


OK = Resp()


def run(script):
    """script：依序回傳的 Resp 或要丟出的例外。回傳 (結果或例外, 睡眠秒數清單, 請求次數, 事件文字清單)。"""
    sleeps, events, calls = [], [], []
    items = list(script)

    def fake_get(*a, **k):
        calls.append(1)
        it = items.pop(0)
        if isinstance(it, Exception):
            raise it
        return it

    saved = (finmind._get, finmind._sleep, finmind.MIN_INTERVAL, finmind.on_event, finmind._load_token, finmind._retries_used)
    finmind._get, finmind._sleep, finmind.MIN_INTERVAL = fake_get, sleeps.append, 0
    finmind.on_event, finmind._load_token, finmind._retries_used = events.append, (lambda: "t"), 0
    try:
        try:
            out = finmind.fetch("X", data_id="1")
        except Exception as e:  # noqa: BLE001
            out = e
    finally:
        (finmind._get, finmind._sleep, finmind.MIN_INTERVAL, finmind.on_event, finmind._load_token, finmind._retries_used) = saved
    return out, sleeps, len(calls), events


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def t01_success_no_retry():
    """一次成功：回傳資料、不睡、不記事件。"""
    out, sleeps, n, ev = run([OK])
    expect(out == [{"a": 1}] and sleeps == [] and n == 1 and ev == [], f"{out} {sleeps} {n} {ev}")


def t02_5xx_retries_then_succeeds():
    """500 兩次後成功：睡 30、120 秒，共 3 次請求，每次重試都有事件。"""
    out, sleeps, n, ev = run([Resp(500), Resp(502), OK])
    expect(out == [{"a": 1}] and sleeps == [30, 120] and n == 3 and len(ev) == 2, f"{out} {sleeps} {n} {ev}")


def t03_5xx_gives_up_after_3_retries():
    """500 連續出現：共 4 次請求（1 次＋3 次重試），睡 30、120、300 秒，最後丟出例外。"""
    out, sleeps, n, _ = run([Resp(500)] * 4)
    expect(isinstance(out, Exception) and sleeps == [30, 120, 300] and n == 4, f"{out!r} {sleeps} {n}")


def t04_network_errors_are_retried():
    """逾時與連線中斷視同 5xx：重試。"""
    out, sleeps, n, _ = run([requests.Timeout("t"), requests.ConnectionError("c"), OK])
    expect(out == [{"a": 1}] and sleeps == [30, 120] and n == 3, f"{out} {sleeps} {n}")


def t05_other_4xx_stops_immediately():
    """400、401、404、422：不重試、不睡。"""
    for code in (400, 401, 404, 422):
        out, sleeps, n, _ = run([Resp(code), OK])
        expect(isinstance(out, Exception) and sleeps == [] and n == 1, f"{code}: {out!r} {sleeps} {n}")


def t06_403_waits_31_minutes_once():
    """403：睡 31 分鐘後再試一次，成功就繼續；再 403 就停（總共 2 次請求）。"""
    out, sleeps, n, _ = run([Resp(403, {"msg": "ip banned", "status": 403}), OK])
    expect(out == [{"a": 1}] and sleeps == [31 * 60] and n == 2, f"{out} {sleeps} {n}")
    out, sleeps, n, _ = run([Resp(403)] * 3)
    expect(isinstance(out, Exception) and sleeps == [31 * 60] and n == 2, f"{out!r} {sleeps} {n}")


def t07_402_waits_until_next_hour_once():
    """402：睡到下個整點（秒數在 0～3700 之間），只試一次；再 402 就停。"""
    out, sleeps, n, _ = run([Resp(402), OK])
    expect(out == [{"a": 1}] and len(sleeps) == 1 and 0 < sleeps[0] <= 3700 and n == 2, f"{out} {sleeps} {n}")
    out, sleeps, n, _ = run([Resp(402)] * 3)
    expect(isinstance(out, Exception) and len(sleeps) == 1 and n == 2, f"{out!r} {sleeps} {n}")


def t08_secs_to_next_hour():
    """到下個整點的秒數：10:15:30 → 2670＋30 緩衝；10:59:59 → 1＋30；整點 10:00:00 → 3600＋30。"""
    f = finmind._secs_to_next_hour
    expect(f(datetime(2026, 1, 1, 10, 15, 30)) == 2670 + 30, f(datetime(2026, 1, 1, 10, 15, 30)))
    expect(f(datetime(2026, 1, 1, 10, 59, 59)) == 1 + 30, f(datetime(2026, 1, 1, 10, 59, 59)))
    expect(f(datetime(2026, 1, 1, 10, 0, 0)) == 3600 + 30, f(datetime(2026, 1, 1, 10, 0, 0)))


def t09_body_status_error_stops():
    """HTTP 200 但內容 status 不是 200（例如 token 錯誤 400）：立刻停；內容不是 JSON 也立刻停。"""
    out, sleeps, n, _ = run([Resp(200, {"status": 400, "msg": "TokenIllegal"}), OK])
    expect(isinstance(out, Exception) and "TokenIllegal" in str(out) and sleeps == [] and n == 1, f"{out!r} {sleeps} {n}")
    out, sleeps, n, _ = run([Resp(200, bad_json=True), OK])
    expect(isinstance(out, Exception) and sleeps == [] and n == 1, f"{out!r} {sleeps} {n}")


def t10_body_status_5xx_and_403_follow_rules():
    """HTTP 200 但內容 status 為 500／403／402 時，與 HTTP 狀態碼同樣處理。"""
    out, sleeps, n, _ = run([Resp(200, {"status": 500, "msg": "x"}), OK])
    expect(out == [{"a": 1}] and sleeps == [30], f"500: {out} {sleeps}")
    out, sleeps, n, _ = run([Resp(200, {"status": 403, "msg": "ip banned"}), OK])
    expect(out == [{"a": 1}] and sleeps == [31 * 60], f"403: {out} {sleeps}")


def t11_run_cap_stops_when_retries_exhausted():
    """每次執行累計重試上限：已用 MAX_RETRIES_PER_RUN 次後，下一個 5xx 不再重試、直接停。"""
    saved = finmind.MAX_RETRIES_PER_RUN
    finmind.MAX_RETRIES_PER_RUN = 2
    try:
        out, sleeps, n, _ = run([Resp(500)] * 4)
    finally:
        finmind.MAX_RETRIES_PER_RUN = saved
    expect(isinstance(out, Exception) and sleeps == [30, 120] and n == 3, f"{out!r} {sleeps} {n}")


def t12_retries_are_throttled():
    """每次請求（含重試）都經過流量限制：MIN_INTERVAL 對每次請求都生效。"""
    sleeps, items = [], [Resp(500), OK]
    saved = (finmind._get, finmind._sleep, finmind.MIN_INTERVAL, finmind.on_event, finmind._load_token, finmind._retries_used, finmind._last_call)
    finmind._get, finmind._sleep, finmind.MIN_INTERVAL = (lambda *a, **k: items.pop(0)), sleeps.append, 6.5
    finmind.on_event, finmind._load_token, finmind._retries_used = (lambda m: None), (lambda: "t"), 0
    finmind._last_call = finmind.time.monotonic()
    try:
        finmind.fetch("X")
    finally:
        (finmind._get, finmind._sleep, finmind.MIN_INTERVAL, finmind.on_event, finmind._load_token, finmind._retries_used, finmind._last_call) = saved
    throttle = [s for s in sleeps if 0 < s < 7]
    expect(len(throttle) == 2, f"兩次請求都應先經流量限制：{sleeps}")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t") and k[1:3].isdigit() and callable(v)]


def main():
    lines, passed = [], 0
    for fn in TESTS:
        title = (fn.__doc__ or fn.__name__).strip().splitlines()[0]
        try:
            fn()
            passed += 1
            lines.append(f"[通過] {fn.__name__}：{title}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"[失敗] {fn.__name__}：{title}\n        {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                lines.append("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    text = "\n".join([f"# FinMind 重試規則測試報告：{passed}／{len(TESTS)} 通過", *lines])
    out = ROOT / "data" / "experiments" / "finmind_retry_tests_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
