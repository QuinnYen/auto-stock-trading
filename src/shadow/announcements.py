"""影子模式：抓上市公司每日重大訊息，交給 Claude 整理成固定格式後存檔。

只記錄，不下單、不影響任何交易邏輯。公告來源（證交所 OpenAPI）只回傳最近一個交易日，
無法回補歷史，所以每天都要跑，錯過的日子補不回來。

輸出：
  data/shadow/raw/<YYYY-MM-DD>.json            證交所原始公告
  data/shadow/announcements/<YYYY-MM-DD>.json  原始公告 + AI 整理結果（含模型與提示詞版本）
任何一步失敗都直接丟出例外，不吞錯誤、不略過。
"""
import json
import os
import sys
from pathlib import Path

import anthropic
import requests

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
RAW_DIR = ROOT / "data" / "shadow" / "raw"
OUT_DIR = ROOT / "data" / "shadow" / "announcements"

SOURCE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
MODEL = "claude-sonnet-5-5"
PROMPT_VERSION = "v2"
SUMMARY_PROMPT_LIMIT = 50  # 提示詞要求的字數；結構化輸出無法限制長度，由程式檢查
SUMMARY_MAX_LEN = 100      # 超過就視為異常並停止（v1 實測模型約超出要求字數 20～40%）
PRICE_IN, PRICE_OUT = 2.0, 10.0  # claude-sonnet-5-5，美元 / 百萬 token（快取不計）

EVENT_TYPES = [
    "財報營收", "股利配息", "增資減資", "併購重組", "重大契約訂單", "人事異動",
    "訴訟裁罰", "取得處分資產", "庫藏股", "股權異動", "更名面額", "其他",
]

SYSTEM = (
    "你是台股盤後公告整理員。每次收到一則上市公司重大訊息，只依公告文字本身判斷，"
    "不要使用公告以外的資訊，也不要推測公司未來股價。\n"
    "- event_type：從列舉中選最貼切的一個。\n"
    "- direction：對公司股東權益的直接影響，positive、negative、neutral 三選一；"
    "公告沒有足夠資訊判斷時用 neutral。\n"
    "- is_routine：只有純程序性、不含新的業績、股利或交易資訊的公告才為 true，包括：法說會或"
    "投資人活動邀請、更正或補正公告、催繳通知、股東會召開通知、媒體報導澄清、資金貸與或背書保證"
    "的例行申報。公告中含有營收、獲利、股利、處分或取得資產、增減資、人事異動等實質內容時為 false，"
    "即使公告是依規定定期發布。\n"
    f"- summary：繁體中文，一句話、不超過 {SUMMARY_PROMPT_LIMIT} 字，只陳述公告內容。"
)

SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "direction": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "is_routine": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["event_type", "direction", "is_routine", "summary"],
    "additionalProperties": False,
}


def _load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "ANTHROPIC_API_KEY":
            return value.strip()
    raise RuntimeError("找不到 ANTHROPIC_API_KEY（環境變數或 .env）")


def roc_to_iso(roc: str) -> str:
    """'1151006' -> '2026-10-06'"""
    roc = roc.strip()
    return f"{int(roc[:-4]) + 1911}-{roc[-4:-2]}-{roc[-2:]}"


def fetch_raw() -> tuple[str, list[dict]]:
    """抓最新一天的重大訊息，回傳 (出表日期 ISO, 公告列表)，並原樣存檔。"""
    resp = requests.get(SOURCE_URL, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise RuntimeError("證交所回傳 0 筆公告")
    dates = {r["出表日期"] for r in rows}
    if len(dates) != 1:
        raise RuntimeError(f"出表日期不一致：{sorted(dates)}")
    day = roc_to_iso(dates.pop())
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{day}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    # 來源欄位名稱有多餘空白（如 "主旨 "），原始檔保留原樣，往後使用的版本去掉空白
    return day, [{k.strip(): v for k, v in r.items()} for r in rows]


def extract(client: anthropic.Anthropic, row: dict) -> tuple[dict, dict]:
    text = (
        f"公司：{row['公司代號']} {row['公司名稱']}\n"
        f"主旨：{row['主旨']}\n"
        f"符合條款：{row['符合條款']}\n"
        f"事實發生日：{row['事實發生日']}\n"
        f"說明：{row['說明']}"
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": text}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"{row['公司代號']} 非正常結束 stop_reason={resp.stop_reason} "
                           f"details={resp.stop_details}")
    ai = json.loads(next(b.text for b in resp.content if b.type == "text"))
    if len(ai["summary"]) > SUMMARY_MAX_LEN:
        raise RuntimeError(f"{row['公司代號']} 摘要過長（{len(ai['summary'])} 字）：{ai['summary']}")
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return ai, usage


def main() -> None:
    day, rows = fetch_raw()
    out = OUT_DIR / f"{day}.json"
    if out.exists():
        print(f"{day} 已整理過，略過（{out}）")
        return

    client = anthropic.Anthropic(api_key=_load_api_key())
    items = []
    for i, row in enumerate(rows, 1):
        ai, usage = extract(client, row)
        items.append({"announcement": row, "ai": ai, "usage": usage})
        print(f"[{i}/{len(rows)}] {row['公司代號']} {ai['event_type']}", flush=True)

    tokens_in = sum(i["usage"]["input_tokens"] for i in items)
    tokens_out = sum(i["usage"]["output_tokens"] for i in items)
    cost = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1_000_000

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"date": day, "model": MODEL, "prompt_version": PROMPT_VERSION,
         "input_tokens": tokens_in, "output_tokens": tokens_out, "est_cost_usd": round(cost, 4),
         "items": items},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成：{len(items)} 則，輸入 {tokens_in} / 輸出 {tokens_out} token，約 ${cost:.4f} -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
