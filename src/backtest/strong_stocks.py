"""強勢股規則回測（依 docs/台股強勢股規則草案 v0.1.md）：突破型與回檔型，帳戶模擬。

資料切分（使用者同意，2026-10-08）：開發期 2007～2016 看參數敏感度；驗證期 2017～2021；
最後測試期 2022 起只跑一次（必須加 --final-confirm，且每次執行都寫入 data/experiments/trials.csv）。
每個設定（版本 × 出場方式 × 參數）算一次試驗；滑價與手續費情境屬假設敏感度，不算新試驗。

用法：python src/backtest/strong_stocks.py [--period dev|val|final] [--version breakout|pullback|both]
                                           [--set 參數=值 ...] [--no-control]
"""
import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import account  # noqa: E402
import prices  # noqa: E402

OUT = prices.OUT
PERIODS = {"dev": ("2007-01-01", "2016-12-31"), "val": ("2017-01-01", "2021-12-31"), "final": ("2022-01-01", "2099-12-31")}
SLIPS = (0.0, 0.0025, 0.005)
N_CONTROL = 30


@dataclass
class Sig:
    liq_min: float = 1e8          # 近 20 日平均成交金額門檻（元）
    attn_days: int = 5            # 近 N 個交易日內被列為注意股就不進場
    market_filter: bool = True    # 0050 收盤低於 60 日均線時不開新倉
    ma_trend: bool = True         # B1：收盤 > 20MA > 60MA 且兩條均線向上
    near_high: float = 0.9        # B2：現價 ÷ 52 週最高價 ≥ 此值；0 ＝ 不用
    rs_n: int = 60                # B3/B4 的漲幅天數，同時用來排序候選
    rs_top: float = 0.2           # B3：漲幅排名前 X%（以通過流動性的股票排名）；1.0 ＝ 不用
    beat_market: bool = True      # B4：漲幅大於 0050 同期
    breakout_n: int = 20          # 突破型：收盤創 N 日新高
    vol_k: float = 1.5            # 突破型：成交量 ≥ 前 5 日均量的 k 倍
    pullback_ma: int = 10         # 回檔型：回測此均線不破
    pullback_tol: float = 0.01    # 回檔型：最低價在均線上方 tol 以內視為回測
    bias_max: float = 0.15        # 收盤相對 20MA 乖離上限


def load_data(con):
    ids = [r[0] for r in con.execute("SELECT DISTINCT stock_id FROM stock_price WHERE stock_id GLOB '[1-9][0-9][0-9][0-9]'")]
    adj = prices.load_adjusted(con, ids + ["0050"])
    cal = pd.DatetimeIndex(sorted(adj[(adj.stock_id == "0050") & (adj.date >= prices.CAL_START)].date.unique()))
    adj = adj[adj.date.isin(set(cal.strftime("%Y-%m-%d")))]
    d = {k: prices.wide(adj, c, cal) for k, c in (("O", "o"), ("H", "h"), ("L", "l"), ("C", "c"), ("V", "volume"),
                                              ("M", "money"), ("RO", "open"))}
    mkt = d["C"]["0050"]
    for k in d:
        d[k] = d[k].drop(columns="0050").astype("float32")
    return d, mkt.astype("float32"), cal


def alert_matrices(con, cols, cal, attn_days):
    """注意股／處置股矩陣（日 × 股）。注意股公告日 D 收盤後公布；處置股先公布、隔天起生效。"""
    col = {s: i for i, s in enumerate(cols)}
    n, m = len(cal), len(cols)
    attn = np.zeros((n, m), dtype=bool)
    rows = con.execute("SELECT stock_id, date FROM attention").fetchall()
    if not rows:
        raise RuntimeError("attention 資料表是空的，請先執行 scripts/download_alerts.py 與 scripts/init_db.py")
    for sid, dt in rows:
        if sid in col:
            i = cal.searchsorted(pd.Timestamp(dt))
            if i < n:
                attn[i, col[sid]] = True
    disp_known = np.zeros((n, m), dtype=bool)   # 已公布且尚未結束（用來擋進場與觸發出場）
    disp_in = np.zeros((n, m), dtype=bool)      # 當天正處於處置期間（賣出加額外滑價）
    drows = con.execute("SELECT stock_id, announce_date, period_start, period_end FROM disposition").fetchall()
    if not drows:
        raise RuntimeError("disposition 資料表是空的")
    for sid, ann, start, end in drows:
        if sid not in col:
            continue
        j = col[sid]
        i_a, i_s = cal.searchsorted(pd.Timestamp(ann)), cal.searchsorted(pd.Timestamp(start))
        i_e = cal.searchsorted(pd.Timestamp(end), side="right")       # 期間結束日的下一個索引
        disp_known[min(i_a, n):min(max(i_e - 1, i_a), n), j] = True
        disp_in[min(i_s, n):min(i_e, n), j] = True
    attn_recent = pd.DataFrame(attn).astype(float).rolling(attn_days, min_periods=1).max().to_numpy() > 0
    return attn, attn_recent, disp_known, disp_in


def build(d, mkt, cal, attn_recent, disp_known, sig: Sig, version: str, ma_exit):
    C, H, L, O, V, M = d["C"], d["H"], d["L"], d["O"], d["V"], d["M"]
    ma = {n: C.rolling(n, min_periods=int(n * 0.8)).mean() for n in {10, 20, 60, 5, sig.pullback_ma, ma_exit or 10}}
    liq = M.rolling(20, min_periods=15).mean()
    liq_ok = liq >= sig.liq_min
    mkt_ok = (mkt > mkt.rolling(60, min_periods=48).mean()) if sig.market_filter else pd.Series(True, index=cal)

    ret_n = C / C.shift(sig.rs_n) - 1
    b1 = (C > ma[20]) & (ma[20] > ma[60]) & (ma[20] > ma[20].shift(5)) & (ma[60] > ma[60].shift(5))
    b2 = C / H.rolling(250, min_periods=200).max() >= (sig.near_high or 0.9)
    b3 = ret_n.where(liq_ok).rank(axis=1, pct=True) >= 1 - min(sig.rs_top, 1.0)
    b4 = ret_n.sub(mkt / mkt.shift(sig.rs_n) - 1, axis=0) > 0
    strong = pd.DataFrame(True, index=C.index, columns=C.columns)
    if sig.ma_trend:
        strong &= b1
    if sig.near_high > 0:
        strong &= b2
    if sig.rs_top < 1.0:
        strong &= b3
    if sig.beat_market:
        strong &= b4

    daily = C.pct_change(fill_method=None)
    no_chase = ~((daily.rolling(3).max() >= 0.095) | (C / C.shift(3) - 1 >= 0.15))
    bias_ok = (C / ma[20] - 1) <= sig.bias_max
    if version == "breakout":
        trigger = (C >= C.rolling(sig.breakout_n, min_periods=sig.breakout_n).max()) & \
                  (V >= sig.vol_k * V.shift(1).rolling(5, min_periods=5).mean())
    else:
        pm = ma[sig.pullback_ma]
        trigger = (L <= pm * (1 + sig.pullback_tol)) & (C >= pm) & (C > O)

    attn_df = pd.DataFrame(attn_recent, index=C.index, columns=C.columns)
    disp_df = pd.DataFrame(disp_known, index=C.index, columns=C.columns)
    base = (liq_ok & ma[60].notna() & ~attn_df & ~disp_df).mul(mkt_ok, axis=0).astype(bool)
    parts = {"價格存在": C.notna().to_numpy(), "流動性": liq_ok.to_numpy(), "均線60日有值": ma[60].notna().to_numpy(),
             "大盤濾網": np.broadcast_to(mkt_ok.to_numpy()[:, None], C.shape), "非近期注意股": ~attn_df.to_numpy(),
             "非處置股": ~disp_df.to_numpy(), "B1均線多頭": b1.to_numpy(), "B2接近52週高": b2.to_numpy(),
             "B3漲幅排名": b3.to_numpy(), "B4勝過大盤": b4.to_numpy(), "進場觸發": trigger.to_numpy(),
             "未追高": no_chase.to_numpy(), "乖離合格": bias_ok.to_numpy()}
    return {"entry": (strong & trigger).to_numpy(), "elig": (base & no_chase & bias_ok).to_numpy(),
            "elig_base": base.to_numpy(), "rank": ret_n.to_numpy(dtype="float64"), "ma": ma, "C": C,
            "parts": parts, "strong": strong.to_numpy()}


def market(d, cal, built, cfg, alerts):
    attn, _, disp_known, disp_in = alerts
    C = built["C"]
    ma_break = (C < built["ma"][cfg.ma_exit]).to_numpy() if cfg.ma_exit else np.zeros(C.shape, dtype=bool)
    last_idx = np.array([np.flatnonzero(~np.isnan(d["O"][c].to_numpy()))[-1] if d["O"][c].notna().any() else -1
                         for c in d["O"].columns])
    f = lambda k: d[k].to_numpy(dtype="float64")  # noqa: E731
    return account.Market(O=f("O"), H=f("H"), L=f("L"), C=f("C"), RO=f("RO"), ma_break=ma_break,
                          attn_exit=attn, disp_exit=disp_known, disp_in=disp_in, last_idx=last_idx)


def summarize(label, res, cal, start_idx):
    equity, trades, triggers, cost = res.equity, res.trades, res.triggers, res.cost_total
    eq = pd.Series(equity, index=cal[start_idx:start_idx + len(equity)])
    years = len(eq) / 245
    cagr = (eq.iloc[-1] / account.START_CAPITAL) ** (1 / years) - 1
    mdd = float((1 - eq / eq.cummax()).max())
    t = pd.DataFrame(trades)
    yearly = eq.groupby(eq.index.year).last().pipe(lambda s: s / s.shift(1).fillna(account.START_CAPITAL) - 1)
    stat = {"label": label, "final": float(eq.iloc[-1]), "cagr": cagr, "mdd": mdd, "triggers": len(triggers),
            "trades": len(t), "win": float((t.net > 0).mean()) if len(t) else 0.0,
            "avg_net": float(t.net.mean()) if len(t) else 0.0,
            "avg_ret": float((t.net / t.basis).mean()) if len(t) else 0.0, "cost": float(cost)}
    lines = [f"[{label}] 期末 {eq.iloc[-1]:,.0f} 元（{eq.iloc[-1] / account.START_CAPITAL - 1:+.0%}），年化 {cagr:+.1%}，"
             f"最大回撤 {mdd:.0%}，停機線 {len(triggers)} 次（每年 {len(triggers) / years:.1f}）",
             f"    交易 {stat['trades']} 筆，勝率 {stat['win']:.0%}，每筆平均淨利 {stat['avg_net']:+.0f} 元"
             f"（{stat['avg_ret']:+.2%}），手續費與稅合計 {cost:,.0f} 元",
             "    各年：" + "  ".join(f"{y}:{r:+.0%}" for y, r in yearly.items())]
    if len(t):
        lines.append("    出場：" + "、".join(f"{k} {v}" for k, v in t.reason.value_counts().items()))
    return lines, stat


def log_trial(row):
    path = OUT / "trials.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def apply_overrides(sig, cfg, pairs):
    for p in pairs:
        key, _, val = p.partition("=")
        for obj in (sig, cfg):
            if key in {f.name for f in fields(obj)}:
                cur = getattr(obj, key)
                if val.lower() == "none":
                    new = None
                elif isinstance(cur, bool):
                    new = val.lower() in ("1", "true", "yes")
                elif cur is None or isinstance(cur, float):
                    new = float(val)
                else:
                    new = type(cur)(val)
                setattr(obj, key, new)
                break
        else:
            raise SystemExit(f"未知參數：{key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="dev", choices=PERIODS)
    ap.add_argument("--version", default="both", choices=("breakout", "pullback", "both"))
    ap.add_argument("--set", nargs="*", default=[], help="覆寫參數，例如 stop_pct=0.05 tp_pct=0.02")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--final-confirm", action="store_true")
    a = ap.parse_args()
    if a.period == "final" and not a.final_confirm:
        raise SystemExit("最後測試期只能跑一次，需明確加上 --final-confirm")

    con = sqlite3.connect(prices.DB)
    print("載入資料…", flush=True)
    d, mkt, cal = load_data(con)
    start_idx = int(cal.searchsorted(pd.Timestamp(PERIODS[a.period][0])))
    end_idx = int(cal.searchsorted(pd.Timestamp(PERIODS[a.period][1]), side="right"))
    print(f"股票 {d['C'].shape[1]} 檔、期間 {cal[start_idx].date()}～{cal[end_idx - 1].date()}", flush=True)

    out = []
    base_sig, base_cfg = Sig(), account.Cfg()
    apply_overrides(base_sig, base_cfg, a.set)
    alerts = alert_matrices(con, list(d["C"].columns), cal, base_sig.attn_days)
    versions = ("breakout", "pullback") if a.version == "both" else (a.version,)
    exits = {"b(無固定停利)": None, "a(停利2%)": 0.02} if base_cfg.tp_pct is None else {f"自訂(停利{base_cfg.tp_pct:.1%})": base_cfg.tp_pct}

    for ver in versions:
        built = build(d, mkt, cal, alerts[1], alerts[2], base_sig, ver, base_cfg.ma_exit)
        n_sig = int((built["entry"] & built["elig"])[start_idx:end_idx].sum())
        out += ["", f"##### {ver}：期間內符合進場條件的股票日 {n_sig} 個"]
        for ename, tp in exits.items():
            cfg0 = account.Cfg(**{**asdict(base_cfg), "tp_pct": tp})
            m = market(d, cal, built, cfg0, alerts)
            for slip in SLIPS:
                cfg = account.Cfg(**{**asdict(cfg0), "slip": slip})
                label = f"{ver} / {ename} / 滑價 {slip:.2%}"
                print(f"模擬 {label}…", flush=True)
                lines, st = summarize(label, account.simulate(m, built["entry"], built["rank"], built["elig"], cfg,
                                                             start_idx, end_idx), cal, start_idx)
                out += ["", *lines]
                log_trial({"time": datetime.now().isoformat(timespec="seconds"), "period": a.period, "version": ver,
                           "exit": ename, "slip": slip, "sig": json.dumps(asdict(base_sig)), "cfg": json.dumps(asdict(cfg)),
                           **{k: st[k] for k in ("cagr", "mdd", "triggers", "trades", "win", "avg_net", "avg_ret")}})
                if not a.no_control and ver == versions[0]:
                    rng = np.random.default_rng(0)
                    cagrs, mdds = [], []
                    for _ in range(N_CONTROL):
                        r = rng.random(built["rank"].shape)
                        _, s2 = summarize("c", account.simulate(m, np.ones_like(built["entry"]), r, built["elig_base"],
                                                               cfg, start_idx, end_idx), cal, start_idx)
                        cagrs.append(s2["cagr"]); mdds.append(s2["mdd"])
                    out.append(f"  隨機挑選對照（{N_CONTROL} 次，相同出場與過濾）：年化中位數 {np.median(cagrs):+.1%}"
                               f"（5%～95% {np.percentile(cagrs, 5):+.1%}～{np.percentile(cagrs, 95):+.1%}），"
                               f"最大回撤中位數 {np.median(mdds):.0%}")
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(out)
    (OUT / f"strong_{a.period}_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
