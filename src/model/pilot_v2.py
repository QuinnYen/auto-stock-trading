"""試驗 #2：把試驗 #1 的模型分數放進「2 萬元、2% 風險、最多 3 檔」的帳戶模擬。

與試驗 #1 的差別：不再是等權組合，而是逐日模擬真實帳戶。
- 模型分數：沿用試驗 #1 的特徵、標籤與每月重訓（參數不變），每天收盤後對可交易股票評分。
- 進場：收盤後分數排名前 TOPN 且未持有的股票，依分數高低，隔天開盤買進，直到持滿 MAX_POSITIONS 檔。
- 部位：單筆風險 2% 總資金 ÷ 停損距離（停損 = 進場價 − 3 × ATR14），另限制每檔不超過總資金 1/3、零股整數股。
- 出場：盤中最低價 <= 停損價（停損隨最高收盤價上調）以 min(開盤, 停損價) 成交；持有滿 HOLD 個交易日後開盤賣出；
  下市無法出場以全損計。
- 停機線：收盤淨值從歷史高點回落 10% 即全數賣出並停手；停手 RESTART_AFTER 個交易日後視為人工重啟，高點重設。
- 成本：手續費 0.1425% × 折扣、每筆最低 1 元，賣出證交稅 0.3%；滑價以每次買賣單邊不利價差 SLIPPAGE_LEVELS 做敏感度分析。
- 對照：同一引擎用隨機分數跑 N_CONTROL 次（規格新必過項 5：勝過同規則的隨機進場）。
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_v1 as v1  # noqa: E402

START_CAPITAL = 20_000.0
MAX_POSITIONS = 3
RISK_PCT = 0.02
POSITION_CAP = 1 / 3
ATR_N, ATR_MULT = 14, 3.0
TOPN = 20
HOLD = v1.HOLD
STOP_LINE = 0.10
RESTART_AFTER = 20          # 假設：停機後約一個月由人工重啟
TAX = 0.003
FEE_BASE, MIN_FEE = 0.001425, 1.0
SCENARIOS = {"3 折": 0.3, "不打折": 1.0}
SLIPPAGE_LEVELS = (0.0, 0.0025, 0.005, 0.01)   # 每次買賣單邊的不利價差（保守假設，用來涵蓋 9:10 零股成交價與滑價）
N_CONTROL = 30
OUT = v1.OUT


def matrices(con, ids, cal):
    adj = v1.load_adjusted(con, ids)
    adj = adj[adj.date.isin(set(cal.strftime("%Y-%m-%d")))]
    return {k: v1.wide(adj, k, cal) for k in ("o", "h", "l", "c", "money")}


def atr_matrix(m):
    c = m["c"]
    tr = pd.concat([m["h"] - m["l"], (m["h"] - c.shift(1)).abs(), (m["l"] - c.shift(1)).abs()]).groupby(level=0).max()
    return tr.reindex(c.index).rolling(ATR_N, min_periods=10).mean()


def model_scores(X, cal):
    """每月第一個交易日訓練一次，該月每天用同一個模型評分（訓練標籤出場日早於訓練日）。"""
    start = int(np.searchsorted(cal, pd.Timestamp(v1.TEST_START)))
    elig = X[X.ok & X.y.notna() & (X.pos % v1.TRAIN_STEP == 0) & (X.index.get_level_values(0) >= v1.TRAIN_START)]
    months = pd.Series(np.arange(start, len(cal)), index=cal[start:]).groupby([cal[start:].year, cal[start:].month])
    pred = []
    for (y, mo), idx in months:
        first, last = idx.iloc[0], idx.iloc[-1]
        model = v1.fit(elig[elig.pos <= first - (HOLD + 2)])
        rows = X[(X.pos >= first) & (X.pos <= last)]
        pred.append(pd.Series(model.predict(rows[v1.FEATURES]), index=rows.index))
        print(f"  評分 {y}-{mo:02d}", flush=True)
    return pd.concat(pred).unstack().reindex(cal)


def simulate(S, m, atr, liq_ok, discount, start_idx, slip=0.0):
    """逐日帳戶模擬。S：分數矩陣（日 × 股），其餘同形狀。回傳 (淨值序列, 交易清單, 停機日期)。"""
    O, H, L, C = (m[k].values for k in ("o", "h", "l", "c"))
    A, SC, OK = atr.values, S.values, liq_ok.values
    n_days, n_st = O.shape
    last_idx = np.array([np.flatnonzero(~np.isnan(O[:, j]))[-1] if (~np.isnan(O[:, j])).any() else -1
                         for j in range(n_st)])
    rate = FEE_BASE * discount

    def fee(amount):
        return max(amount * rate, MIN_FEE)

    cash, peak = START_CAPITAL, START_CAPITAL
    held: dict[int, dict] = {}
    last_close = np.full(n_st, np.nan)
    pending_buy: list[int] = []
    sell_all, halted_until = False, -1
    equity = np.full(n_days, np.nan)
    trades, triggers, cost_total = [], [], 0.0

    def close_position(j, d, price, reason):
        nonlocal cash, cost_total
        pos = held.pop(j)
        price = price * (1 - slip)
        amount = pos["shares"] * price
        cost = fee(amount) + amount * TAX if price > 0 else 0.0
        cash += amount - cost
        cost_total += cost
        trades.append({"stock": j, "entry_idx": pos["entry_idx"], "exit_idx": d, "shares": pos["shares"],
                       "entry": pos["entry"], "exit": price, "reason": reason,
                       "gross": pos["shares"] * (price - pos["entry"]), "net": amount - cost - pos["basis"]})

    for d in range(start_idx, n_days):
        # 1) 開盤：執行昨天收盤後決定的動作
        for j in list(held):
            if d > last_idx[j]:
                close_position(j, d, 0.0, "下市全損")
            elif np.isnan(O[d, j]):
                continue
            elif sell_all or held[j]["sell_at_open"]:
                close_position(j, d, O[d, j], "停機" if sell_all else "持有期滿")
        sell_all = False
        if d >= halted_until and pending_buy:
            equity_prev = equity[d - 1] if d > start_idx else START_CAPITAL
            for j in pending_buy:
                if len(held) >= MAX_POSITIONS:
                    break
                if j in held or np.isnan(O[d, j]) or np.isnan(A[d - 1, j]):
                    continue
                entry = O[d, j] * (1 + slip)
                dist = min(ATR_MULT * A[d - 1, j], entry * 0.99)
                budget = min(RISK_PCT * equity_prev / dist, equity_prev * POSITION_CAP / entry)
                shares = int(budget)
                while shares > 0 and shares * entry + fee(shares * entry) > cash:
                    shares -= 1
                if shares < 1:
                    continue
                cost = fee(shares * entry)
                cash -= shares * entry + cost
                cost_total += cost
                held[j] = {"shares": shares, "entry": entry, "entry_idx": d, "stop": entry - dist,
                           "atr": A[d - 1, j], "maxc": entry, "basis": shares * entry + cost,
                           "sell_at_open": False}
        pending_buy = []

        # 2) 盤中：停損
        for j in list(held):
            if np.isnan(L[d, j]):
                continue
            if L[d, j] <= held[j]["stop"]:
                close_position(j, d, min(O[d, j], held[j]["stop"]), "停損")

        # 3) 收盤：更新停損、標記淨值、判斷停機線與持有期滿、產生明日候選
        last_close = np.where(np.isnan(C[d]), last_close, C[d])
        for j, pos in held.items():
            if not np.isnan(C[d, j]):
                pos["maxc"] = max(pos["maxc"], C[d, j])
                pos["stop"] = max(pos["stop"], pos["maxc"] - ATR_MULT * pos["atr"])
            if d - pos["entry_idx"] >= HOLD - 1:
                pos["sell_at_open"] = True
        equity[d] = cash + sum(p["shares"] * last_close[j] for j, p in held.items())
        peak = equity[d] if d == halted_until else max(peak, equity[d])   # 人工重啟當天高點重設
        if d >= halted_until and equity[d] <= peak * (1 - STOP_LINE):
            sell_all, halted_until = True, d + RESTART_AFTER
            triggers.append(d)
        if d >= halted_until and not sell_all:
            ok = np.flatnonzero(OK[d] & ~np.isnan(SC[d]))
            order = ok[np.argsort(-SC[d, ok], kind="stable")][:TOPN]
            pending_buy = [int(j) for j in order if j not in held]
    return equity, trades, triggers, cost_total


def summarize(name, equity, trades, triggers, cost_total, cal, start_idx):
    eq = pd.Series(equity[start_idx:], index=cal[start_idx:])
    years = len(eq) / 245
    cagr = (eq.iloc[-1] / START_CAPITAL) ** (1 / years) - 1
    mdd = float((1 - eq / eq.cummax()).max())
    t = pd.DataFrame(trades)
    gross = t.gross.sum() if len(t) else 0.0
    yearly = eq.groupby(eq.index.year).last().pipe(lambda s: s / s.shift(1).fillna(START_CAPITAL) - 1)
    lines = [f"[{name}] 期末 {eq.iloc[-1]:,.0f} 元（{eq.iloc[-1] / START_CAPITAL - 1:+.0%}），年化 {cagr:+.1%}，"
             f"最大回撤 {mdd:.0%}，停機線觸發 {len(triggers)} 次（平均每年 {len(triggers) / years:.1f} 次）",
             f"    交易 {len(t)} 筆，勝率 {(t.net > 0).mean():.0%}，總成本 {cost_total:,.0f} 元，"
             f"毛利 {gross:,.0f} 元，成本占毛利 {cost_total / gross:.0%}" if gross > 0 else
             f"    交易 {len(t)} 筆，毛利 {gross:,.0f} 元（成本占比不適用）",
             "    各年報酬：" + "  ".join(f"{y}:{r:+.0%}" for y, r in yearly.items())]
    if len(t):
        lines.append("    出場原因：" + "、".join(f"{k} {v}" for k, v in t.reason.value_counts().items()))
    return lines, cagr, mdd, float(eq.iloc[-1])


def main():
    con = sqlite3.connect(v1.DB)
    ids = json.loads(v1.PILOT.read_text(encoding="utf-8"))
    print("建立特徵與標籤…", flush=True)
    X, bench, cal = v1.build(con, ids)
    m = matrices(con, ids, cal)
    atr = atr_matrix(m)
    liq = m["money"].rolling(20, min_periods=15).mean()
    liq_ok = liq >= v1.LIQ_MIN
    start_idx = int(np.searchsorted(cal, pd.Timestamp(v1.TEST_START)))
    print("逐月訓練與評分…", flush=True)
    S = model_scores(X, cal).reindex(columns=m["c"].columns)

    out = []
    for sname, disc in SCENARIOS.items():
        for slip in SLIPPAGE_LEVELS:
            label = f"{sname} / 單邊滑價 {slip:.2%}"
            print(f"模擬：{label}…", flush=True)
            eq, trades, trig, cost = simulate(S, m, atr, liq_ok, disc, start_idx, slip)
            lines, cagr, mdd, final = summarize(f"模型 / {label}", eq, trades, trig, cost, cal, start_idx)
            out += ["", *lines]
            rng = np.random.default_rng(0)
            ctrl = []
            for _ in range(N_CONTROL):
                R = pd.DataFrame(rng.random(S.shape), index=S.index, columns=S.columns)
                e, t_, tr_, c_ = simulate(R, m, atr, liq_ok, disc, start_idx, slip)
                ctrl.append((summarize("c", e, t_, tr_, c_, cal, start_idx)[1:], len(tr_)))
            finals = np.array([c[0][2] for c in ctrl])
            cagrs = np.array([c[0][0] for c in ctrl])
            mdds = np.array([c[0][1] for c in ctrl])
            trigs = np.array([c[1] for c in ctrl])
            out += [f"  隨機進場對照（{N_CONTROL} 次）：年化中位數 {np.median(cagrs):+.1%}"
                    f"（5%～95% 區間 {np.percentile(cagrs, 5):+.1%}～{np.percentile(cagrs, 95):+.1%}），"
                    f"最大回撤中位數 {np.median(mdds):.0%}，停機觸發中位數 {np.median(trigs):.0f} 次；"
                    f"模型年化勝過 {(cagr > cagrs).mean():.0%} 的對照"]
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(out)
    (OUT / "pilot_v2_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
