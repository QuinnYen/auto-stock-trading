"""帳戶模擬引擎：2 萬元、2% 風險反推部位、最多 3 檔、逐日撮合（規則型策略用）。

價格處理
- 訊號與損益使用還原價（O/H/L/C）；零股股數、價格上限與成交金額使用原始開盤價（RO）。
- 持倉價值 ＝ 股數 × scale × 還原價，scale ＝ 進場原始價 ÷ 進場還原價（等於股息再投入的總報酬）。
撮合假設（保守）
- 進場：訊號日收盤後決定，隔天開盤買；開盤已接近漲停（≥ 前收 +9.5%）視為買不到。
- 出場（stop_close=True 時改為：收盤跌破停損價才標記、隔天開盤賣出，無盤中停損）：停損價在盤中被碰到時以 min(開盤, 停損價) 成交（停損先於停利）；開盤接近跌停（≤ 前收 −9.5%）賣不掉，順延。
- 滑價：每次買賣單邊不利價差 slip；處置期間的賣出再加 disp_extra_slip。
- 下市：最後價格日之後仍持有，視為全損。
帳戶風控：收盤淨值自高點回落 stop_line 即全數賣出並停手 restart_after 個交易日（假設人工重啟，高點重設）。
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

START_CAPITAL = 20_000.0
FEE_BASE, MIN_FEE, TAX = 0.001425, 1.0, 0.003
LIMIT_LOCK = 0.095


@dataclass
class Cfg:
    stop_pct: float = 0.07          # 初始停損（相對進場價）
    ma_exit: int | None = 10        # 收盤跌破 N 日均線，隔天開盤賣出
    tp_pct: float | None = None     # 固定停利（相對進場價）；None ＝ 不設
    time_stop: int = 10             # 持有滿 N 個交易日，開盤賣出
    exit_on_attention: bool = True  # 持股被列為注意股，隔天開盤賣出
    risk_pct: float = 0.02
    risk_sized: bool = True         # False ＝ 固定比例部位：每檔買 pos_cap × 前日淨值（不用風險反推）
    stop_close: bool = False        # True ＝ 收盤（還原價）≤ 進場價 ×(1−stop_pct) 才標記，隔天開盤賣；不做盤中停損
    max_pos: int = 3
    pos_cap: float = 1 / 3
    min_shares: int = 10            # 價格上限：每檔預算 ÷ 股價 ≥ min_shares
    discount: float = 0.3
    slip: float = 0.0
    disp_extra_slip: float = 0.01   # 處置期間賣出的額外滑價（placeholder 假設）
    stop_line: float = 0.10
    restart_after: int = 20


@dataclass
class Market:
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    RO: np.ndarray
    ma_break: np.ndarray
    attn_exit: np.ndarray
    disp_exit: np.ndarray
    disp_in: np.ndarray
    last_idx: np.ndarray


@dataclass
class Result:
    equity: np.ndarray
    cash: np.ndarray
    n_held: np.ndarray
    trades: list
    triggers: list
    cost_total: float
    counters: Counter = field(default_factory=Counter)
    open_positions: list = field(default_factory=list)
    capital: float = START_CAPITAL


def simulate(m: Market, entry, rank, elig, cfg: Cfg, start_idx: int, end_idx: int, capital=START_CAPITAL) -> Result:
    """entry/elig 為布林矩陣（日 × 股），rank 為候選排序分數（越大越優先）。"""
    O, H, L, C, RO = m.O, m.H, m.L, m.C, m.RO
    rate = FEE_BASE * cfg.discount
    rank = np.where(np.isnan(rank), -np.inf, rank)

    def fee(amount):
        return max(amount * rate, MIN_FEE)

    n = end_idx - start_idx
    cash, peak = capital, capital
    held: dict[int, dict] = {}
    pending: list[int] = []
    sell_all, halted_until = False, -1
    equity, cash_arr, n_held = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=int)
    trades, triggers, cost_total = [], [], 0.0
    cnt: Counter = Counter()
    last_close = C[start_idx - 1].copy()

    def sell(j, d, adj_price, reason):
        nonlocal cash, cost_total
        pos = held.pop(j)
        slip = cfg.slip + (cfg.disp_extra_slip if m.disp_in[d, j] else 0.0)
        raw = adj_price * pos["scale"] * (1 - slip) if adj_price > 0 else 0.0
        amount = pos["shares"] * raw
        fee_sell = fee(amount) if amount > 0 else 0.0
        tax = amount * TAX
        cash += amount - fee_sell - tax
        cost_total += fee_sell + tax
        trades.append({"stock": j, "entry_idx": pos["entry_idx"], "exit_idx": d, "shares": pos["shares"],
                       "reason": reason, "entry_adj": pos["entry_adj"], "exit_adj": adj_price,
                       "entry_raw": pos["raw_entry"], "exit_raw": raw, "scale": pos["scale"],
                       "stop_adj": pos["stop"], "fee_buy": pos["fee_buy"], "fee_sell": fee_sell, "tax": tax,
                       "eq_at_entry": pos["eq_at_entry"], "basis": pos["basis"],
                       "gross": amount - pos["shares"] * pos["raw_entry"],
                       "net": amount - fee_sell - tax - pos["basis"]})

    for d in range(start_idx, end_idx):
        k = d - start_idx
        eq_prev = equity[k - 1] if k > 0 else capital

        # 1) 開盤：賣出已決定的部位，再買進昨天選出的候選
        for j in list(held):
            pos = held[j]
            if d > m.last_idx[j]:
                sell(j, d, 0.0, "下市全損")
                cnt["出場_下市全損"] += 1
            elif np.isnan(O[d, j]):
                continue
            elif sell_all or pos["flag"]:
                if O[d, j] <= C[d - 1, j] * (1 - LIMIT_LOCK):
                    cnt["出場順延_跌停鎖死"] += 1
                    continue
                sell(j, d, O[d, j], "停機" if sell_all else pos["flag"])
        sell_all = False
        if d >= halted_until:
            for j in pending:
                if len(held) >= cfg.max_pos:
                    cnt["候選未用_已持滿"] += 1
                    continue
                if j in held:
                    continue
                if np.isnan(O[d, j]) or np.isnan(RO[d, j]) or RO[d, j] <= 0 or np.isnan(C[d - 1, j]):
                    cnt["候選略過_無開盤價"] += 1
                    continue
                if O[d, j] >= C[d - 1, j] * (1 + LIMIT_LOCK):
                    cnt["候選略過_漲停買不到"] += 1
                    continue
                raw_fill = RO[d, j] * (1 + cfg.slip)
                budget = eq_prev * cfg.pos_cap
                if budget / raw_fill < cfg.min_shares:
                    cnt["候選略過_股價超過價格上限"] += 1
                    continue
                shares = int(budget / raw_fill) if not cfg.risk_sized else                     int(min(cfg.risk_pct * eq_prev / (cfg.stop_pct * raw_fill), budget / raw_fill))
                while shares > 0 and shares * raw_fill + fee(shares * raw_fill) > cash:
                    shares -= 1
                if shares < 1:
                    cnt["候選略過_股數不足或現金不足"] += 1
                    continue
                fee_buy = fee(shares * raw_fill)
                cash -= shares * raw_fill + fee_buy
                cost_total += fee_buy
                adj_fill = O[d, j] * (1 + cfg.slip)
                held[j] = {"shares": shares, "entry_idx": d, "scale": RO[d, j] / O[d, j], "raw_entry": raw_fill,
                           "entry_adj": adj_fill, "stop": adj_fill * (1 - cfg.stop_pct),
                           "tp": adj_fill * (1 + cfg.tp_pct) if cfg.tp_pct else None, "fee_buy": fee_buy,
                           "eq_at_entry": eq_prev, "basis": shares * raw_fill + fee_buy, "flag": None}
                cnt["進場成功"] += 1
        pending = []

        # 2) 盤中：停損優先於停利
        for j in list(held):
            pos = held[j]
            if np.isnan(L[d, j]) or np.isnan(O[d, j]):
                continue
            if O[d, j] <= C[d - 1, j] * (1 - LIMIT_LOCK):
                if L[d, j] <= pos["stop"]:
                    cnt["停損順延_跌停鎖死"] += 1
                continue
            if cfg.stop_close:
                continue
            if L[d, j] <= pos["stop"]:
                sell(j, d, min(O[d, j], pos["stop"]), "停損")
            elif pos["tp"] is not None and H[d, j] >= pos["tp"]:
                sell(j, d, max(O[d, j], pos["tp"]), "停利")

        # 3) 收盤：標記淨值、決定隔天要賣的部位、停機線、選出明天的候選
        last_close = np.where(np.isnan(C[d]), last_close, C[d])
        eq = cash + sum(p["shares"] * p["scale"] * last_close[j] for j, p in held.items())
        equity[k], cash_arr[k], n_held[k] = eq, cash, len(held)
        for j, pos in held.items():
            if pos["flag"]:
                continue
            if m.disp_exit[d, j]:
                pos["flag"] = "處置股"
            elif cfg.exit_on_attention and m.attn_exit[d, j]:
                pos["flag"] = "注意股"
            elif cfg.stop_close and C[d, j] <= pos["stop"]:
                pos["flag"] = "停損"
            elif cfg.ma_exit and m.ma_break[d, j]:
                pos["flag"] = "均線出場"
            elif d - pos["entry_idx"] >= cfg.time_stop - 1:
                pos["flag"] = "時間停損"
        peak = eq if d == halted_until else max(peak, eq)
        if d >= halted_until and eq <= peak * (1 - cfg.stop_line):
            sell_all, halted_until = True, d + cfg.restart_after
            triggers.append(d)
            cnt["停機觸發"] += 1
        if d >= halted_until and not sell_all:
            ok = np.flatnonzero(entry[d] & elig[d])
            cnt["符合進場條件的股票日"] += len(ok)
            if len(ok):
                order = ok[np.argsort(-rank[d, ok], kind="stable")][:20]
                pending = [int(j) for j in order if j not in held]
    return Result(equity, cash_arr, n_held, trades, triggers, cost_total, cnt,
                  [{"stock": j, **p} for j, p in held.items()], capital)
