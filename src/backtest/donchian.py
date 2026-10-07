"""20 日高點突破（唐奇安）+ ATR 移動停損 的第一版回測。

規則（皆為第一版假設，見規格第 9-1 節）：
- 訊號：收盤價 > 前 20 日最高價，且近 20 日平均成交金額 >= LIQ_MIN
- 進場：隔天開盤價；同日多檔訊號依近 60 日漲幅由高到低
- 停損：進場價 - ATR_MULT × ATR14；之後隨最高收盤價上調（只升不降）
- 出場：盤中最低價 <= 停損價，以 min(開盤, 停損價) 成交
- 部位：單筆風險 2% 總資金 ÷ 停損距離，最多同時 3 檔，零股（1 股為單位）
- 下市後仍持有：DELIST_MODE="zero" 視為全損；"last" 以最後收盤價賣出
"""
import json
import re
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
COMMON = re.compile(r"[1-9]\d{3}")

START_CAPITAL = 20_000
BACKTEST_START = "2005-01-01"
BREAKOUT_N = 20
ATR_N = 14
ATR_MULT = 3.0
RISK_PCT = 0.02
MAX_POSITIONS = 3
LIQ_MIN = 50_000_000  # 近 20 日平均成交金額（元）；門檻值待定，先用保守假設
FEE_BASE = 0.001425
MIN_FEE = 1           # 零股每筆最低 1 元（使用者提供）
TAX_RATE = 0.003


def _load_json(name: str):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def load_stock(stock_id: str):
    """讀取並還原（除權息、減資）。檔案不齊或沒資料回傳 None。"""
    names = [f"price/price_{stock_id}.json", f"dividend_result/dividend_result_{stock_id}.json",
             f"capital_reduction/capital_reduction_{stock_id}.json"]
    if not all((RAW / n).exists() for n in names):
        return None
    price, div, cap = (_load_json(n) for n in names)
    rows = [r for r in price if r["close"] > 0 and r["open"] > 0]
    if len(rows) < BREAKOUT_N + ATR_N:
        return None

    events = [(e["date"], e["reference_price"] / e["before_price"])
              for e in div if e["before_price"] > 0 and e["reference_price"] > 0]
    events += [(e["date"], e["PostReductionReferencePrice"] / e["ClosingPriceonTheLastTradingDay"])
               for e in cap
               if e["ClosingPriceonTheLastTradingDay"] > 0 and e["PostReductionReferencePrice"] > 0]
    events.sort()

    # 事件日之前的價格都乘上該事件的調整係數
    factor, j, factors = 1.0, len(events) - 1, [1.0] * len(rows)
    for i in range(len(rows) - 1, -1, -1):
        while j >= 0 and events[j][0] > rows[i]["date"]:
            factor *= events[j][1]
            j -= 1
        factors[i] = factor

    s = {
        "date": [r["date"] for r in rows],
        "o": [r["open"] * f for r, f in zip(rows, factors)],
        "h": [r["max"] * f for r, f in zip(rows, factors)],
        "l": [r["min"] * f for r, f in zip(rows, factors)],
        "c": [r["close"] * f for r, f in zip(rows, factors)],
        "money": [r["Trading_money"] for r in rows],
    }
    n = len(rows)
    s["last"] = s["date"][-1]
    s["atr"] = [None] * n
    s["hh"] = [None] * n   # 前 N 日最高價（不含當天）
    s["liq"] = [None] * n
    tr = [s["h"][0] - s["l"][0]] + [
        max(s["h"][i] - s["l"][i], abs(s["h"][i] - s["c"][i - 1]), abs(s["l"][i] - s["c"][i - 1]))
        for i in range(1, n)]
    for i in range(n):
        if i >= ATR_N:
            s["atr"][i] = sum(tr[i - ATR_N + 1:i + 1]) / ATR_N
        if i >= BREAKOUT_N:
            s["hh"][i] = max(s["h"][i - BREAKOUT_N:i])
            s["liq"][i] = sum(s["money"][i - BREAKOUT_N + 1:i + 1]) / BREAKOUT_N
    return s


def universe() -> list[str]:
    ids = {p.stem.split("_")[-1] for p in (RAW / "price").glob("price_*.json")}
    return sorted(i for i in ids if COMMON.fullmatch(i))


def run(stocks: dict, fee_rate: float, delist_mode: str):
    dates = sorted({d for s in stocks.values() for d in s["date"] if d >= BACKTEST_START})
    signals: dict[str, list] = {}
    for sid, s in stocks.items():
        for i in range(60, len(s["date"])):
            if s["date"][i] < BACKTEST_START or s["hh"][i] is None or s["atr"][i] is None:
                continue
            if s["c"][i] > s["hh"][i] and s["liq"][i] >= LIQ_MIN:
                signals.setdefault(s["date"][i], []).append((s["c"][i] / s["c"][i - 60] - 1, sid))

    def fee(amount):
        return max(amount * fee_rate, MIN_FEE)

    cash = float(START_CAPITAL)
    held: dict[str, dict] = {}
    pending: list[str] = []
    trades, equity, total_cost = [], [], 0.0

    def sell(sid, pos, price):
        nonlocal cash, total_cost
        amount = pos["shares"] * price
        cost = fee(amount) + amount * TAX_RATE
        cash += amount - cost
        total_cost += cost
        trades.append((pos["cost_basis"], amount - cost))

    def mark(sid, pos, d):
        s = stocks[sid]
        k = bisect_right(s["date"], d) - 1
        return pos["shares"] * s["c"][k]

    for d in dates:
        # 1) 昨天的訊號，今天開盤進場
        eq_prev = cash + sum(mark(sid, p, d) for sid, p in held.items())
        for sid in pending:
            s = stocks[sid]
            i = bisect_left(s["date"], d)
            if sid in held or len(held) >= MAX_POSITIONS or i >= len(s["date"]) or s["date"][i] != d:
                continue
            atr_sig = s["atr"][i - 1]
            entry = s["o"][i]
            dist = ATR_MULT * atr_sig
            shares = int(min(RISK_PCT * eq_prev / dist, cash / (entry * (1 + fee_rate))))
            if shares < 1:
                continue
            amount = shares * entry
            while shares > 1 and amount + fee(amount) > cash:
                shares -= 1
                amount = shares * entry
            if amount + fee(amount) > cash:
                continue
            cost = fee(amount)
            cash -= amount + cost
            total_cost += cost
            held[sid] = {"shares": shares, "atr": atr_sig, "stop": entry - dist,
                         "maxc": entry, "cost_basis": amount + cost}
        pending = []

        # 2) 盤中停損 / 下市處理 / 3) 收盤上調停損
        for sid in list(held):
            s, pos = stocks[sid], held[sid]
            if d > s["last"]:
                if delist_mode == "last":
                    sell(sid, pos, s["c"][-1])
                else:
                    trades.append((pos["cost_basis"], 0.0))
                del held[sid]
                continue
            i = bisect_left(s["date"], d)
            if s["date"][i] != d:
                continue  # 停止買賣，無法成交
            if s["l"][i] <= pos["stop"]:
                sell(sid, pos, min(s["o"][i], pos["stop"]))
                del held[sid]
            else:
                pos["maxc"] = max(pos["maxc"], s["c"][i])
                pos["stop"] = max(pos["stop"], pos["maxc"] - ATR_MULT * pos["atr"])

        # 4) 收盤產生明天的訊號（依近 60 日漲幅排序）
        pending = [sid for _, sid in sorted(signals.get(d, []), reverse=True)]
        equity.append(cash + sum(mark(sid, p, d) for sid, p in held.items()))

    return dates, equity, trades, total_cost


def metrics(dates, equity, lo="", hi="9999"):
    idx = [i for i, d in enumerate(dates) if lo <= d < hi]
    e = [equity[i] for i in idx]
    years = len(e) / 245
    peak, mdd = e[0], 0.0
    for x in e:
        peak = max(peak, x)
        mdd = max(mdd, 1 - x / peak)
    total = e[-1] / e[0] - 1
    cagr = (e[-1] / e[0]) ** (1 / years) - 1 if years > 0 else 0.0
    return {"total": total, "cagr": cagr, "mdd": mdd, "calmar": cagr / mdd if mdd else 0.0}


def stop_line_hits(equity, pct=0.10):
    """從收盤淨值高點回落 pct 的次數（回到新高才重新計）。"""
    peak, hits, breached = equity[0], 0, False
    for x in equity:
        if x > peak:
            peak, breached = x, False
        elif not breached and x < peak * (1 - pct):
            hits, breached = hits + 1, True
    return hits


def benchmark_0050():
    price = [r for r in _load_json("price/price_0050.json") if r["close"] > 0]
    div = _load_json("dividend_result/dividend_result_0050.json")
    events = [(e["date"], e["reference_price"] / e["before_price"]) for e in div if e["before_price"] > 0]
    # 2025-06 的 4 拆 1 不在除權息表；單日跌幅 > 50% 視為拆分，用收盤比調整（僅用於 0050 基準）
    for a, b in zip(price, price[1:]):
        if b["close"] / a["close"] < 0.5:
            events.append((b["date"], b["close"] / a["close"]))
    events.sort()
    factor, j, out = 1.0, len(events) - 1, {}
    for r in reversed(price):
        while j >= 0 and events[j][0] > r["date"]:
            factor *= events[j][1]
            j -= 1
        out[r["date"]] = r["close"] * factor
    return out


def main():
    stocks = {}
    for sid in universe():
        s = load_stock(sid)
        if s:
            stocks[sid] = s
    print(f"載入 {len(stocks)} 檔（檔案齊全者）", flush=True)

    bench = benchmark_0050()
    # 3 折（月退佣，使用者提供）為主情境；不打折為保守對照
    for discount in (0.3, 1.0):
        fee_rate = FEE_BASE * discount
        for mode in ("zero", "last"):
            dates, equity, trades, cost = run(stocks, fee_rate, mode)
            wins = sum(1 for basis, got in trades if got > basis)
            print(f"\n=== 手續費 {discount * 10:g} 折、最低 {MIN_FEE} 元；下市持股 {'全損' if mode == 'zero' else '按最後價'} ===")
            print(f"交易 {len(trades)} 筆，勝率 {wins / max(len(trades), 1):.0%}，總成本 {cost:,.0f} 元，"
                  f"停機線觸發 {stop_line_hits(equity)} 次，期末淨值 {equity[-1]:,.0f}")
            for label, lo, hi in (("全期間", "", "9999"), ("樣本內 2005-2017", "", "2018"),
                                  ("樣本外 2018-", "2018", "9999")):
                m = metrics(dates, equity, lo, hi)
                print(f"  {label}: 總報酬 {m['total']:+.0%} 年化 {m['cagr']:+.1%} "
                      f"最大回撤 {m['mdd']:.0%} Calmar {m['calmar']:.2f}")

    bd = [d for d in sorted(bench) if d >= BACKTEST_START]
    be = [bench[d] for d in bd]
    print("\n=== 0050 買進持有（還原） ===")
    for label, lo, hi in (("全期間", "", "9999"), ("樣本內 2005-2017", "", "2018"),
                          ("樣本外 2018-", "2018", "9999")):
        m = metrics(bd, be, lo, hi)
        print(f"  {label}: 總報酬 {m['total']:+.0%} 年化 {m['cagr']:+.1%} "
              f"最大回撤 {m['mdd']:.0%} Calmar {m['calmar']:.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
