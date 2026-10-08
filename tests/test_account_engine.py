"""帳戶模擬引擎（src/backtest/account.py）的人工構造情境測試。

每個情境用小型合成價格資料，預期結果以手算公式寫在測試裡（不呼叫引擎內部函式），涵蓋冒煙測試沒觸發到的路徑：
漲停買不到、跌停賣不掉（出場與停損）、下市全損、停損跳空、停利與停損優先順序、注意／處置／均線出場與優先順序、
價格上限、最低手續費、滑價、還原價 scale、停機線與重啟、持滿 3 檔排序、現金不足縮股、停牌日順延；
另外獨立重算強勢股訊號的 B3 漲幅排名、B4 勝過大盤與大盤濾網（strong_stocks.build）。

用法：python tests/test_account_engine.py        報告：data/experiments/engine_tests_report.txt
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "backtest"))
import account  # noqa: E402
import strong_stocks as ss  # noqa: E402

RATE = 0.001425 * 0.3
TAX = 0.003
N_DAYS = 30


def fee(amount, rate=RATE):
    return max(amount * rate, 1.0)


def mk(n_st=3, n_days=N_DAYS, price=100.0):
    """平坦價格（高低收開都 100），原始價等於還原價。"""
    full = lambda: np.full((n_days, n_st), price, dtype="float64")  # noqa: E731
    zeros = lambda: np.zeros((n_days, n_st), dtype=bool)  # noqa: E731
    return {"O": full(), "H": full(), "L": full(), "C": full(), "RO": full(), "ma_break": zeros(),
            "attn_exit": zeros(), "disp_exit": zeros(), "disp_in": zeros(),
            "last_idx": np.full(n_st, n_days - 1)}


def set_day(a, d, j, o=None, h=None, l=None, c=None, ro=None):
    for key, v in (("O", o), ("H", h), ("L", l), ("C", c), ("RO", ro)):
        if v is not None:
            a[key][d, j] = v


def set_all(a, d_from, j, price, raw=True):
    """從 d_from 起把某檔所有價格設成固定值。"""
    for key in ("O", "H", "L", "C") + (("RO",) if raw else ()):
        a[key][d_from:, j] = price


def run(a, signals, cfg=None, rank=None, capital=20000.0):
    """signals：[(訊號日, 股票索引)]；回傳 Result。排名預設：索引小的優先。"""
    cfg = cfg or account.Cfg(ma_exit=None, time_stop=5)
    n_days, n_st = a["O"].shape
    m = account.Market(**a)
    entry = np.zeros((n_days, n_st), dtype=bool)
    for d, j in signals:
        entry[d, j] = True
    if rank is None:
        rank = np.tile(-np.arange(n_st, dtype="float64"), (n_days, 1))
    return account.simulate(m, entry, rank, np.ones_like(entry), cfg, 1, n_days, capital)


def approx(x, y, tol=1e-6):
    return abs(x - y) <= tol * max(1.0, abs(y))


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def one_trade(res):
    expect(len(res.trades) == 1, f"預期 1 筆交易，實際 {len(res.trades)} 筆：{[t['reason'] for t in res.trades]}")
    return res.trades[0]


# ---------------------------------------------------------------- 情境
def t01_time_stop_and_costs():
    """平坦價格、持有滿 5 個交易日開盤賣：股數、進出場日、手續費、證交稅、淨損益。"""
    res = run(mk(), [(3, 0)])
    t = one_trade(res)
    shares = int(min(0.02 * 20000 / (0.07 * 100), 20000 / 3 / 100))
    expect(t["shares"] == shares == 57, f"股數 {t['shares']}，預期 57")
    expect((t["entry_idx"], t["exit_idx"], t["reason"]) == (4, 9, "時間停損"), f"{t['entry_idx']},{t['exit_idx']},{t['reason']}")
    amount = shares * 100
    expect(approx(t["fee_buy"], fee(amount)) and approx(t["fee_sell"], fee(amount)) and approx(t["tax"], amount * TAX), "費用不符")
    expect(approx(t["net"], -(2 * fee(amount) + amount * TAX)), f"淨損益 {t['net']}")
    expect(approx(res.equity[-1], 20000 - 2 * fee(amount) - amount * TAX), "期末淨值不符")


def t02_gap_stop():
    """跳空 −8% 低開並跌破停損價：以開盤價 92 成交（不是停損價 93）。"""
    a = mk(); set_day(a, 6, 0, o=92, h=92, l=91, c=91); set_all(a, 7, 0, 91)
    t = one_trade(run(a, [(3, 0)]))
    expect(t["reason"] == "停損" and t["exit_idx"] == 6, f"{t['reason']} {t['exit_idx']}")
    expect(approx(t["exit_adj"], 92), f"成交價 {t['exit_adj']}，預期 92")


def t03_intraday_stop():
    """沒有跳空，盤中最低價 92 ≤ 停損價 93：以停損價 93 成交。"""
    a = mk(); set_day(a, 6, 0, o=100, h=100, l=92, c=95); set_all(a, 7, 0, 95)
    t = one_trade(run(a, [(3, 0)]))
    expect(t["reason"] == "停損" and approx(t["exit_adj"], 93), f"{t['reason']} {t['exit_adj']}")


def t04_entry_day_stop():
    """進場當天盤中就碰到停損價：當天以 93 賣出。"""
    a = mk(); set_day(a, 4, 0, o=100, h=100, l=92, c=95); set_all(a, 5, 0, 95)
    t = one_trade(run(a, [(3, 0)]))
    expect(t["entry_idx"] == t["exit_idx"] == 4 and approx(t["exit_adj"], 93), f"{t}")


def t05_take_profit():
    """停利 2%：盤中最高 103 → 以停利價 102 成交；跳空高開 105 → 以開盤 105 成交。"""
    cfg = account.Cfg(ma_exit=None, time_stop=5, tp_pct=0.02)
    a = mk(); set_day(a, 6, 0, o=100, h=103, l=100, c=102); set_all(a, 7, 0, 102)
    t = one_trade(run(a, [(3, 0)], cfg))
    expect(t["reason"] == "停利" and approx(t["exit_adj"], 102), f"{t['reason']} {t['exit_adj']}")
    a = mk(); set_day(a, 6, 0, o=105, h=106, l=105, c=105); set_all(a, 7, 0, 105)
    t = one_trade(run(a, [(3, 0)], cfg))
    expect(t["reason"] == "停利" and approx(t["exit_adj"], 105), f"跳空 {t['exit_adj']}")


def t06_stop_before_take_profit():
    """同一天最低碰停損、最高碰停利：停損優先，成交價 min(開盤, 停損價) ＝ 93。"""
    cfg = account.Cfg(ma_exit=None, time_stop=5, tp_pct=0.02)
    a = mk(); set_day(a, 6, 0, o=100, h=105, l=90, c=100); set_all(a, 7, 0, 100)
    t = one_trade(run(a, [(3, 0)], cfg))
    expect(t["reason"] == "停損" and approx(t["exit_adj"], 93), f"{t['reason']} {t['exit_adj']}")


def t07_limit_up_cannot_buy():
    """開盤 110（前收 100，+10%）買不到；開盤 109.4（+9.4%）買得到。"""
    a = mk(); set_all(a, 4, 0, 110)
    res = run(a, [(3, 0)])
    expect(len(res.trades) == 0 and res.counters["候選略過_漲停買不到"] == 1, f"{res.counters}")
    a = mk(); set_all(a, 4, 0, 109.4)
    res = run(a, [(3, 0)])
    t = one_trade(res)
    expect(approx(t["entry_raw"], 109.4), f"進場價 {t['entry_raw']}")


def t08_limit_down_blocks_exit():
    """時間停損當天開盤跌停鎖死（90 ≤ 前收 100 × 0.905）賣不掉，順延到隔天開盤 89 賣出。"""
    cfg = account.Cfg(stop_pct=0.5, ma_exit=None, time_stop=5)
    a = mk(); set_day(a, 9, 0, o=90, h=90, l=90, c=90); set_all(a, 10, 0, 89)
    res = run(a, [(3, 0)], cfg)
    t = one_trade(res)
    expect(t["exit_idx"] == 10 and approx(t["exit_adj"], 89) and t["reason"] == "時間停損", f"{t['exit_idx']} {t['exit_adj']} {t['reason']}")
    expect(res.counters["出場順延_跌停鎖死"] == 1, f"{res.counters}")


def t09_limit_down_blocks_stop():
    """停損日開盤跌停鎖死：停損順延；隔天開盤 89（不再鎖死）才以 min(89, 93) ＝ 89 賣出。"""
    a = mk(); set_day(a, 6, 0, o=90, h=90, l=88, c=89); set_all(a, 7, 0, 89)
    res = run(a, [(3, 0)])
    t = one_trade(res)
    expect(t["reason"] == "停損" and t["exit_idx"] == 7 and approx(t["exit_adj"], 89), f"{t['reason']} {t['exit_idx']} {t['exit_adj']}")
    expect(res.counters["停損順延_跌停鎖死"] >= 1, f"{res.counters}")


def t10_delisting_write_off():
    """最後價格日 8 之後仍持有：第 9 天以全損（賣價 0、無手續費與稅）出清。"""
    a = mk()
    for key in ("O", "H", "L", "C", "RO"):
        a[key][9:, 0] = np.nan
    a["last_idx"][0] = 8
    res = run(a, [(3, 0)])
    t = one_trade(res)
    expect(t["reason"] == "下市全損" and t["exit_idx"] == 9, f"{t['reason']} {t['exit_idx']}")
    expect(t["exit_raw"] == 0 and t["fee_sell"] == 0 and t["tax"] == 0 and approx(t["net"], -t["basis"]), f"{t}")
    expect(approx(res.equity[-1], 20000 - t["basis"]) and not np.isnan(res.equity).any(), "出清後淨值或有 NaN")


def t11_attention_exit():
    """注意股公告日 6 收盤後 → 第 7 天開盤賣；關閉 exit_on_attention 則照時間停損。"""
    a = mk(); a["attn_exit"][6, 0] = True
    t = one_trade(run(a, [(3, 0)]))
    expect(t["reason"] == "注意股" and t["exit_idx"] == 7, f"{t['reason']} {t['exit_idx']}")
    t = one_trade(run(a, [(3, 0)], account.Cfg(ma_exit=None, time_stop=5, exit_on_attention=False)))
    expect(t["reason"] == "時間停損" and t["exit_idx"] == 9, f"{t['reason']} {t['exit_idx']}")


def t12_disposition_exit_extra_slip():
    """處置公告 → 隔天開盤賣；處置期間內賣出加 1% 額外滑價：賣價 100 × 0.99。"""
    a = mk(); a["disp_exit"][6, 0] = True; a["disp_in"][7, 0] = True
    t = one_trade(run(a, [(3, 0)]))
    expect(t["reason"] == "處置股" and t["exit_idx"] == 7 and approx(t["exit_raw"], 99.0), f"{t['reason']} {t['exit_idx']} {t['exit_raw']}")


def t13_exit_priority():
    """同一天同時符合：處置股 > 注意股 > 均線出場 > 時間停損。"""
    cfg = account.Cfg(ma_exit=10, time_stop=5)
    for flags, expected in ((("disp_exit", "attn_exit", "ma_break"), "處置股"), (("attn_exit", "ma_break"), "注意股"),
                            (("ma_break",), "均線出場")):
        a = mk()
        for f in flags:
            a[f][6, 0] = True
        t = one_trade(run(a, [(3, 0)], cfg))
        expect(t["reason"] == expected and t["exit_idx"] == 7, f"{flags} → {t['reason']}，預期 {expected}")


def t14_price_cap():
    """每檔預算 6,666.67：股價 700 → 9.5 股 < 10 → 略過；股價 600 → 買 9 股（風險 2% 反推 9.52 → 9）。"""
    a = mk(); set_all(a, 0, 0, 700)
    res = run(a, [(3, 0)])
    expect(len(res.trades) == 0 and res.counters["候選略過_股價超過價格上限"] == 1, f"{res.counters}")
    a = mk(); set_all(a, 0, 0, 600)
    expect(one_trade(run(a, [(3, 0)]))["shares"] == 9, "股價 600 應買 9 股")


def t15_min_fee():
    """小部位：8 股 × 100 ＝ 800 元，手續費公式 0.34 元 → 取最低 1 元。"""
    res = run(mk(), [(3, 0)], capital=3000.0)
    t = one_trade(res)
    expect(t["shares"] == 8 and t["fee_buy"] == 1.0 and t["fee_sell"] == 1.0, f"{t['shares']} {t['fee_buy']} {t['fee_sell']}")


def t16_slippage():
    """單邊滑價 0.5%：買價 100.5、賣價 99.5；股數依含滑價的買價反推 56 股。"""
    t = one_trade(run(mk(), [(3, 0)], account.Cfg(ma_exit=None, time_stop=5, slip=0.005)))
    expect(approx(t["entry_raw"], 100.5) and approx(t["exit_raw"], 99.5), f"{t['entry_raw']} {t['exit_raw']}")
    expect(t["shares"] == int(min(0.02 * 20000 / (0.07 * 100.5), 20000 / 3 / 100.5)) == 56, f"{t['shares']}")


def t17_adjusted_scale():
    """還原價 50、原始價 100（scale ＝ 2）：股數依原始價 57 股；持有中淨值以 57 × 2 × 還原價標記；
    還原價漲到 55 賣出 → 原始賣價 110、毛利 570。"""
    a = mk()
    for key in ("O", "H", "L", "C"):
        a[key][:, 0] = 50
    a["O"][9, 0] = a["H"][9, 0] = a["L"][9, 0] = a["C"][9, 0] = 55
    a["O"][10:, 0] = a["H"][10:, 0] = a["L"][10:, 0] = a["C"][10:, 0] = 55
    res = run(a, [(3, 0)])
    t = one_trade(res)
    expect(t["shares"] == 57 and approx(t["scale"], 2.0), f"{t['shares']} {t['scale']}")
    expect(approx(t["exit_adj"], 55) and approx(t["exit_raw"], 110) and approx(t["gross"], 57 * 10), f"{t['exit_raw']} {t['gross']}")
    expect(approx(res.equity[5 - 1], 20000 - fee(5700)), f"持有中淨值 {res.equity[4]}")


def t18_stop_line_halt_and_restart():
    """停機線 1%、停手 5 日：第 5 天停損虧損使淨值回落 > 1% → 觸發一次；停手期間（第 7 天）的訊號不進場；
    第 10 天（重啟日）的訊號第 11 天進場，且高點已重設，不會立刻再次觸發。"""
    cfg = account.Cfg(ma_exit=None, time_stop=5, stop_line=0.01, restart_after=5)
    a = mk(); set_day(a, 5, 0, o=100, h=100, l=92, c=95); set_all(a, 6, 0, 95)
    res = run(a, [(3, 0), (7, 1), (10, 1)], cfg)
    expect(res.triggers == [5], f"觸發日 {res.triggers}")
    entries = sorted((t["entry_idx"], t["stock"]) for t in res.trades)
    expect(entries == [(4, 0), (11, 1)], f"進場 {entries}（停手期間第 7 天的訊號不應成交）")


def t19_stop_line_liquidates_other_positions():
    """停機觸發當天仍持有另一檔：隔天開盤全數賣出，原因為『停機』。"""
    cfg = account.Cfg(ma_exit=None, time_stop=5, stop_line=0.01, restart_after=5)
    a = mk(); set_day(a, 5, 0, o=100, h=100, l=92, c=95); set_all(a, 6, 0, 95)
    res = run(a, [(3, 0), (3, 1)], cfg)
    by_stock = {t["stock"]: t for t in res.trades}
    expect(by_stock[0]["reason"] == "停損" and by_stock[1]["reason"] == "停機" and by_stock[1]["exit_idx"] == 6, f"{[(t['stock'], t['reason'], t['exit_idx']) for t in res.trades]}")
    expect(res.counters["停機觸發"] == 1, f"{res.counters}")


def t20_slots_and_rank():
    """4 檔同日訊號、排名 4>3>2>1：只買前 3 檔；第 4 檔不買。"""
    rank = np.tile(np.array([4.0, 3.0, 2.0, 1.0]), (N_DAYS, 1))
    res = run(mk(n_st=4), [(3, 0), (3, 1), (3, 2), (3, 3)], rank=rank)
    expect(sorted(t["stock"] for t in res.trades) == [0, 1, 2], f"{[t['stock'] for t in res.trades]}")
    expect(res.n_held.max() == 3 and res.counters["候選未用_已持滿"] >= 1, f"{res.n_held.max()} {res.counters}")


def t21_cash_limit_shrinks_shares():
    """第一檔吃掉大部分現金，第二檔股數被現金限制縮到 19 股（20 股會超過現金）。"""
    cfg = account.Cfg(ma_exit=None, time_stop=5, risk_pct=0.5, pos_cap=0.9)
    res = run(mk(), [(3, 0), (3, 1)], cfg)
    by_stock = {t["stock"]: t for t in res.trades}
    expect(by_stock[0]["shares"] == 180, f"第一檔 {by_stock[0]['shares']}")
    cash_left = 20000 - 18000 - fee(18000)
    expect(19 * 100 + fee(1900) <= cash_left < 20 * 100 + fee(2000), "測試前提不成立")
    expect(by_stock[1]["shares"] == 19, f"第二檔 {by_stock[1]['shares']}，預期 19")
    expect(np.nanmin(res.cash) >= 0, "現金為負")


def t22_halted_on_exit_day():
    """應賣出當天停牌（無價格）：持有到復牌後第一天開盤賣出。"""
    a = mk()
    for key in ("O", "H", "L", "C", "RO"):
        a[key][9, 0] = np.nan
    res = run(a, [(3, 0)])
    t = one_trade(res)
    expect(t["reason"] == "時間停損" and t["exit_idx"] == 10, f"{t['reason']} {t['exit_idx']}")
    expect(not np.isnan(res.equity).any(), "停牌日淨值出現 NaN")


# ---------------------------------------------------------------- 訊號的獨立重算
def t23_signal_components_independent():
    """合成 40 檔 × 360 日隨機漫步（其中 10 檔流動性不足）：用最直接的逐股逐日迴圈獨立重算
    B3 漲幅排名（只在通過流動性的股票內排名）、B4 勝過大盤、大盤濾網，與 strong_stocks.build 逐元素比對。"""
    rng = np.random.default_rng(7)
    n_days, n_st = 360, 40
    cal = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"{1101 + i}" for i in range(n_st)]
    C = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (n_days, n_st)), axis=0)), index=cal, columns=cols)
    mkt = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days))), index=cal)
    M = pd.DataFrame(2e8, index=cal, columns=cols)
    M.iloc[:, 30:] = 1e6                                    # 後 10 檔流動性不足
    V = pd.DataFrame(rng.integers(1000, 5000, (n_days, n_st)).astype(float), index=cal, columns=cols)
    d = {"C": C.astype("float32"), "H": (C * 1.01).astype("float32"), "L": (C * 0.99).astype("float32"),
         "O": C.astype("float32"), "V": V.astype("float32"), "M": M.astype("float32")}
    sig = ss.Sig()
    z = np.zeros((n_days, n_st), dtype=bool)
    b = ss.build(d, mkt.astype("float32"), cal, z, z, sig, "breakout", 10)
    parts = b["parts"]
    Cv, Mv, mk_v = d["C"].to_numpy(dtype="float64"), d["M"].to_numpy(dtype="float64"), mkt.to_numpy(dtype="float64")
    bad = []
    for day in (80, 120, 200, 260, 330, 359):
        liquid = [j for j in range(n_st)
                  if np.mean(Mv[day - 19:day + 1, j]) >= sig.liq_min and not np.isnan(Cv[day - sig.rs_n, j])]
        ret = {j: Cv[day, j] / Cv[day - sig.rs_n, j] - 1 for j in liquid}
        n = len(liquid)
        mret = mk_v[day] / mk_v[day - sig.rs_n] - 1
        for j in range(n_st):
            if j in ret:
                above = sum(1 for k in liquid if ret[k] > ret[j])
                exp_b3 = above <= n * sig.rs_top + 1e-9
            else:
                exp_b3 = False
            exp_b4 = (Cv[day, j] / Cv[day - sig.rs_n, j] - 1) > mret
            if parts["B3漲幅排名"][day, j] != exp_b3:
                bad.append(("B3", day, j))
            if parts["B4勝過大盤"][day, j] != exp_b4:
                bad.append(("B4", day, j))
        exp_mkt = mk_v[day] > np.mean(mk_v[day - 59:day + 1])
        if not (parts["大盤濾網"][day] == exp_mkt).all():
            bad.append(("大盤濾網", day, -1))
    expect(not bad, f"獨立重算不一致 {len(bad)} 處：{bad[:6]}")
    expect(parts["B3漲幅排名"][:, 30:].sum() == 0, "流動性不足的股票不應通過 B3")


# ---------------------------------------------------------------- 帳戶模擬 v1 新增功能：固定比例部位、收盤停損
def fixed_cfg(**kw):
    base = dict(risk_sized=False, stop_close=True, stop_pct=0.15, ma_exit=None, exit_on_attention=False,
                time_stop=5, pos_cap=1 / 3)
    base.update(kw)
    return account.Cfg(**base)


def t24_fixed_fraction_shares():
    """固定比例：每檔買淨值 1/3 → int(6666.67÷100) ＝ 66 股（風險反推在 15% 停損下只會買 26 股）；本金 3 萬則為 100 股。"""
    t = one_trade(run(mk(), [(3, 0)], fixed_cfg()))
    expect(t["shares"] == 66, f"股數 {t['shares']}，預期 66")
    expect(approx(t["net"], -(2 * fee(6600) + 6600 * TAX)), f"淨損益 {t['net']}")
    t = one_trade(run(mk(), [(3, 0)], fixed_cfg(), capital=30000.0))
    expect(t["shares"] == 100, f"本金 3 萬股數 {t['shares']}，預期 100")
    t = one_trade(run(mk(), [(3, 0)], account.Cfg(stop_pct=0.15, ma_exit=None, time_stop=5)))
    expect(t["shares"] == 26, f"風險反推對照 {t['shares']}，預期 26")


def t25_fixed_fraction_uses_prior_equity():
    """第一檔漲到 130 後，第二檔以前一日淨值的 1/3 買進（不是本金的 1/3）。"""
    a = mk(); set_all(a, 5, 0, 130)
    res = run(a, [(3, 0), (5, 1)], fixed_cfg(time_stop=10))
    by = {t["stock"]: t for t in res.trades}
    cash = 20000 - 6600 - fee(6600)
    eq = cash + 66 * 130
    expect(by[1]["shares"] == int(eq / 3 / 100) == 73, f"第二檔 {by[1]['shares']}，預期 73")


def t26_close_stop_no_intraday():
    """收盤停損：盤中最低 50 但收盤 90 不停損；收盤剛好等於停損價 85 才標記，隔天開盤賣出。"""
    stop = 100 * (1 - 0.15)
    a = mk(); set_day(a, 6, 0, o=100, h=100, l=50, c=90); set_all(a, 7, 0, 90)
    t = one_trade(run(a, [(3, 0)], fixed_cfg()))
    expect(t["reason"] == "時間停損" and t["exit_idx"] == 9, f"盤中下殺不應停損：{t['reason']} {t['exit_idx']}")
    a = mk(); set_day(a, 6, 0, o=100, h=100, l=100, c=stop + 0.01); set_all(a, 7, 0, stop + 0.01)
    t = one_trade(run(a, [(3, 0)], fixed_cfg()))
    expect(t["reason"] == "時間停損", f"高於停損價一點不應停損：{t['reason']}")
    a = mk(); set_day(a, 6, 0, o=100, h=100, l=84, c=stop); set_day(a, 7, 0, o=83, h=83, l=83, c=83); set_all(a, 8, 0, 83)
    t = one_trade(run(a, [(3, 0)], fixed_cfg()))
    expect(t["reason"] == "停損" and t["exit_idx"] == 7 and approx(t["exit_adj"], 83), f"{t['reason']} {t['exit_idx']} {t['exit_adj']}")


def t27_close_stop_on_entry_day_and_priority():
    """進場當天收盤即跌破 → 隔天開盤賣；停損與時間停損同一天成立時原因為停損。"""
    a = mk(); set_day(a, 4, 0, o=100, h=100, l=84, c=84); set_all(a, 5, 0, 84)
    t = one_trade(run(a, [(3, 0)], fixed_cfg()))
    expect(t["reason"] == "停損" and t["exit_idx"] == 5, f"{t['reason']} {t['exit_idx']}")
    a = mk(); set_day(a, 8, 0, o=100, h=100, l=84, c=84); set_all(a, 9, 0, 84)
    t = one_trade(run(a, [(3, 0)], fixed_cfg()))
    expect(t["reason"] == "停損" and t["exit_idx"] == 9, f"優先順序：{t['reason']} {t['exit_idx']}")


def t28_no_stop_control():
    """對照「不設停損」（stop_pct ＝ 1.0）：價格跌到 10 仍持有到時間停損，以開盤 10 賣出（帳戶停機線也關掉，否則會先停機）。"""
    a = mk(); set_all(a, 6, 0, 10)
    t = one_trade(run(a, [(3, 0)], fixed_cfg(stop_pct=1.0, stop_line=1.0)))
    expect(t["reason"] == "時間停損" and t["exit_idx"] == 9 and approx(t["exit_adj"], 10), f"{t['reason']} {t['exit_idx']} {t['exit_adj']}")


def t29_close_stop_deferred_by_limit_down():
    """停損標記後隔天開盤跌停鎖死（75 ≤ 84×0.905）賣不掉，順延到再隔天開盤 74 賣出。"""
    a = mk(); set_day(a, 6, 0, o=100, h=100, l=84, c=84); set_day(a, 7, 0, o=75, h=75, l=75, c=75); set_all(a, 8, 0, 74)
    res = run(a, [(3, 0)], fixed_cfg())
    t = one_trade(res)
    expect(t["reason"] == "停損" and t["exit_idx"] == 8 and approx(t["exit_adj"], 74), f"{t['reason']} {t['exit_idx']} {t['exit_adj']}")
    expect(res.counters["出場順延_跌停鎖死"] == 1, f"{res.counters}")


def t30_seeded_rank_picks_top_slots():
    """同日 5 檔訊號、3 個空位：依傳入排名（模擬種子隨機排序）取前 3 檔；同一排名重跑結果相同。"""
    rng = np.random.default_rng(7)
    rank = rng.random((N_DAYS, 5))
    expected = sorted(np.argsort(-rank[3])[:3].tolist())
    sig = [(3, j) for j in range(5)]
    r1 = run(mk(n_st=5), sig, fixed_cfg(), rank=rank)
    r2 = run(mk(n_st=5), sig, fixed_cfg(), rank=rank)
    expect(sorted(t["stock"] for t in r1.trades) == expected, f"{[t['stock'] for t in r1.trades]} vs {expected}")
    expect([t["stock"] for t in r1.trades] == [t["stock"] for t in r2.trades], "同排名重跑結果不同")


TESTS = [t01_time_stop_and_costs, t02_gap_stop, t03_intraday_stop, t04_entry_day_stop, t05_take_profit,
         t06_stop_before_take_profit, t07_limit_up_cannot_buy, t08_limit_down_blocks_exit,
         t09_limit_down_blocks_stop, t10_delisting_write_off, t11_attention_exit, t12_disposition_exit_extra_slip,
         t13_exit_priority, t14_price_cap, t15_min_fee, t16_slippage, t17_adjusted_scale,
         t18_stop_line_halt_and_restart, t19_stop_line_liquidates_other_positions, t20_slots_and_rank,
         t21_cash_limit_shrinks_shares, t22_halted_on_exit_day, t23_signal_components_independent,
         t24_fixed_fraction_shares, t25_fixed_fraction_uses_prior_equity, t26_close_stop_no_intraday,
         t27_close_stop_on_entry_day_and_priority, t28_no_stop_control, t29_close_stop_deferred_by_limit_down,
         t30_seeded_rank_picks_top_slots]


def main():
    lines, passed = [], 0
    for fn in TESTS:
        title = (fn.__doc__ or fn.__name__).strip().splitlines()[0]
        try:
            fn()
            passed += 1
            lines.append(f"[通過] {fn.__name__}：{title}")
        except Exception as e:  # noqa: BLE001  測試要回報所有失敗，不能在第一個就中止
            lines.append(f"[失敗] {fn.__name__}：{title}\n        {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                lines.append("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    head = f"# 帳戶模擬引擎情境測試報告：{passed}／{len(TESTS)} 通過"
    text = "\n".join([head, *lines])
    out = ROOT / "data" / "experiments" / "engine_tests_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
