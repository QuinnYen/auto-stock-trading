"""試驗 #1：用籌碼與基本面特徵預測未來 10 個交易日的相對報酬（試跑 200 檔）。

設計（事先固定，不調參；規格第 9.5 節）
- 決策日 t（收盤後）；進場 t+1 開盤，出場 t+1+HOLD 開盤；報酬相對 0050 同期間。
- 特徵只用 t 日收盤後已公布的資料（法人、融資券、外資持股、PER 皆在 t 日晚間 21:00 前公布，
  月營收取公布月 15 日之後才可用）；每個特徵在每個決策日對股票做橫斷面百分位排名。
- 標籤：t 日橫斷面「超額報酬百分位」；下市後無法出場者以 -100% 計（規格的保守假設）。
- 模型：HistGradientBoostingRegressor，參數固定；每月重訓一次，訓練樣本的標籤出場日必須早於決策日。
- 驗證：2020-01-01 起每 HOLD 個交易日一個決策日（持有期不重疊），只看樣本外。
- 評估：排名相關（IC）、前 K 名等權組合扣成本後的超額報酬、與全體等權／動能／隨機挑選對照。
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from prices import CAL_START, DB, OUT, PILOT, load_adjusted, sql, wide  # noqa: E402,F401

HOLD = 10
TRAIN_START = "2006-01-01"
TEST_START = "2020-01-01"
SEGMENT_SPLIT = "2024-07-01"
TRAIN_STEP = 5
LIQ_MIN = 50_000_000       # 近 20 日平均成交金額；與突破回測相同的假設
TOPKS = (3, 20)
COST = {"3 折": 2 * 0.001425 * 0.3 + 0.003, "不打折": 2 * 0.001425 + 0.003}
N_RANDOM = 1000
DELIST_CUTOFF_DAYS = 30    # 最後價格日早於資料結束日這麼多天，才視為已下市
PERIODS_PER_YEAR = 245 / HOLD

FEATURES = [
    "ret5", "ret20", "ret60", "vol20", "dist_high60", "liq20",
    "f_net5", "f_net20", "t_net5", "t_net20",
    "margin_ratio", "margin_chg20", "short_ratio", "short_chg20",
    "foreign_ratio", "foreign_chg20",
    "per", "pbr", "div_yield", "rev_yoy", "rev_yoy3",
]


def build(con, ids):
    """回傳 (特徵排名長表, 標籤資料, 日曆)。"""
    adj = load_adjusted(con, ids + ["0050"])
    cal = pd.DatetimeIndex(sorted(adj[(adj.stock_id == "0050") & (adj.date >= CAL_START)].date.unique())).astype("datetime64[ns]")
    cal_str = set(cal.strftime("%Y-%m-%d"))
    adj = adj[adj.date.isin(cal_str)]

    O, C = wide(adj, "o", cal), wide(adj, "c", cal)
    V, M = wide(adj, "volume", cal), wide(adj, "money", cal)
    bench_o = O["0050"]
    O, C, V, M = (x.drop(columns="0050") for x in (O, C, V, M))
    stocks = list(O.columns)
    qs = ",".join(f"'{s}'" for s in stocks)

    daily = C.pct_change(fill_method=None)
    jumps = int((daily.abs() > 0.3).sum().sum())
    print(f"價格調整後單日漲跌幅超過 30% 的次數：{jumps}（資料品質檢查）", flush=True)

    F = {
        "ret5": C / C.shift(5) - 1, "ret20": C / C.shift(20) - 1, "ret60": C / C.shift(60) - 1,
        "vol20": daily.rolling(20, min_periods=15).std(),
        "dist_high60": C / C.rolling(60, min_periods=50).max() - 1,
    }
    liq = M.rolling(20, min_periods=15).mean()
    F["liq20"] = np.log(liq.where(liq > 0))

    inst = sql(con, f"SELECT stock_id,date,name,buy-sell AS net FROM institutional "
                    f"WHERE stock_id IN ({qs}) AND name IN ('Foreign_Investor','Investment_Trust')")
    for name, tag in (("Foreign_Investor", "f"), ("Investment_Trust", "t")):
        net = wide(inst[inst.name == name].groupby(["date", "stock_id"], as_index=False).net.sum(), "net", cal)
        for w in (5, 20):
            F[f"{tag}_net{w}"] = (net.rolling(w, min_periods=int(w * 0.75)).sum()
                                  / V.rolling(w, min_periods=int(w * 0.75)).sum())

    sh = sql(con, f"SELECT stock_id,date,shares_issued,foreign_shares_ratio FROM shareholding WHERE stock_id IN ({qs})")
    shares = wide(sh, "shares_issued", cal).ffill()
    fr = wide(sh, "foreign_shares_ratio", cal).ffill(limit=10)
    F["foreign_ratio"], F["foreign_chg20"] = fr, fr - fr.shift(20)

    mg = sql(con, f"SELECT stock_id,date,margin_today_balance,short_today_balance FROM margin WHERE stock_id IN ({qs})")
    mr = wide(mg, "margin_today_balance", cal) * 1000 / shares     # 融資餘額單位為張
    sr = wide(mg, "short_today_balance", cal) * 1000 / shares
    F["margin_ratio"], F["margin_chg20"] = mr, mr - mr.shift(20)
    F["short_ratio"], F["short_chg20"] = sr, sr - sr.shift(20)

    pe = sql(con, f"SELECT stock_id,date,dividend_yield,per,pbr FROM per WHERE stock_id IN ({qs})")
    F["per"] = wide(pe, "per", cal).where(lambda x: x > 0)
    F["pbr"] = wide(pe, "pbr", cal).where(lambda x: x > 0)
    F["div_yield"] = wide(pe, "dividend_yield", cal)

    rev = sql(con, f"SELECT stock_id,date,revenue,revenue_year,revenue_month FROM month_revenue WHERE stock_id IN ({qs})")
    rev["ym"] = rev.revenue_year * 12 + rev.revenue_month
    prev = rev[["stock_id", "ym", "revenue"]].assign(ym=lambda d: d.ym + 12).rename(columns={"revenue": "prev"})
    rev = rev.merge(prev, on=["stock_id", "ym"], how="left").sort_values(["stock_id", "ym"])
    rev["yoy"] = np.where(rev.prev > 0, rev.revenue / rev.prev - 1, np.nan)
    rev["yoy3"] = rev.groupby("stock_id").yoy.transform(lambda s: s.rolling(3, min_periods=2).mean())
    rev["avail"] = pd.to_datetime(rev.date.str[:8] + "15")     # 資料日為公布月 1 日；保守取 15 日起可用
    for col, name in (("yoy", "rev_yoy"), ("yoy3", "rev_yoy3")):
        w = rev.pivot(index="avail", columns="stock_id", values=col)
        F[name] = w.reindex(w.index.union(cal)).ffill().reindex(cal)

    # 標籤：t+1 開盤進場，t+1+HOLD 開盤出場（出場日無價就取之後 5 個交易日內第一個）
    entry = O.shift(-1)
    exit_ = O.bfill(limit=5).shift(-(1 + HOLD))
    ret = exit_ / entry - 1
    exit_date = pd.Series(cal).shift(-(1 + HOLD))
    last = pd.Series({s: O[s].last_valid_index() for s in stocks})
    cutoff = cal[-1] - pd.Timedelta(days=DELIST_CUTOFF_DAYS)
    ended = (exit_date.values[:, None] > last.values[None, :]) & (last.values < cutoff)[None, :]
    ret = ret.where(~(ret.isna() & entry.notna() & pd.DataFrame(ended, index=cal, columns=stocks)), -1.0)
    bench = bench_o.bfill(limit=5).shift(-(1 + HOLD)) / bench_o.shift(-1) - 1
    excess = ret.sub(bench, axis=0)

    tradable = entry.notna() & (liq >= LIQ_MIN)
    X = pd.DataFrame({k: F[k].rank(axis=1, pct=True).stack(future_stack=True) for k in FEATURES})
    X["ret"] = ret.stack(future_stack=True)
    X["excess"] = excess.stack(future_stack=True)
    X["y"] = excess.rank(axis=1, pct=True).stack(future_stack=True)
    X["ok"] = tradable.stack(future_stack=True)
    X["pos"] = cal.get_indexer(X.index.get_level_values(0))
    ndel = int((ret == -1.0).sum().sum())
    print(f"股票 {len(stocks)} 檔、交易日 {len(cal)}、下市全損標籤 {ndel} 筆", flush=True)
    return X, bench, cal


def fit(train):
    m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=4,
                                      min_samples_leaf=300, l2_regularization=1.0,
                                      early_stopping=False, random_state=0)
    m.fit(train[FEATURES], train["y"])
    return m


def walk_forward(X, bench, cal):
    X = X.copy()
    start = int(np.searchsorted(cal, pd.Timestamp(TEST_START)))
    rebal = [i for i in range(start, len(cal) - (2 + HOLD), HOLD)]
    elig = X[X.ok & X.y.notna() & (X.pos % TRAIN_STEP == 0) & (X.index.get_level_values(0) >= TRAIN_START)]
    by_pos = X.groupby("pos")
    models, rows, picks = {}, [], []
    rng = np.random.default_rng(0)

    for n, i in enumerate(rebal):
        key = (cal[i].year, cal[i].month)
        if key not in models:
            models[key] = fit(elig[elig.pos <= i - (HOLD + 2)])
            print(f"  重訓 {key[0]}-{key[1]:02d}（{n + 1}/{len(rebal)}）", flush=True)
        day = by_pos.get_group(i)
        day = day[day.ok & day.ret.notna()].copy()
        if len(day) < 25:
            raise RuntimeError(f"{cal[i].date()} 可交易股票只有 {len(day)} 檔，無法評估")
        day["score"] = models[key].predict(day[FEATURES])
        b = float(bench.iloc[i])
        rec = {"date": cal[i], "n": len(day), "bench": b, "univ": day.ret.mean(),
               "ic": day.score.corr(day.excess, method="spearman")}
        for name, col in (("m", "score"), ("mom", "ret60")):
            ranked = day.sort_values(col, ascending=False, kind="stable")
            for k in TOPKS:
                rec[f"{name}{k}"] = ranked.ret.iloc[:k].mean()
        rec["rand"] = [day.ret.values[rng.choice(len(day), size=k, replace=False)].mean()
                       for _ in range(N_RANDOM) for k in TOPKS]
        rows.append(rec)
        picks.append(day.assign(date=cal[i])[["date", "score", "ret", "excess"]])
    return pd.DataFrame(rows).set_index("date"), pd.concat(picks)


def stats(excess):
    n = len(excess)
    mean, sd = excess.mean(), excess.std(ddof=1)
    return {"n": n, "mean%": mean * 100, "t": mean / (sd / np.sqrt(n)) if sd > 0 else 0.0,
            "年化%": mean * PERIODS_PER_YEAR * 100, "勝率%": (excess > 0).mean() * 100}


def mdd(returns):
    eq = (1 + returns).cumprod()
    return float((1 - eq / eq.cummax()).max())


def report(res):
    lines = []
    segs = (("全部樣本外", res), (f"{TEST_START[:4]}～{SEGMENT_SPLIT[:7]}", res[res.index < SEGMENT_SPLIT]),
            (f"{SEGMENT_SPLIT[:7]} 起", res[res.index >= SEGMENT_SPLIT]))
    for label, r in segs:
        lines.append(f"\n=== {label}：{len(r)} 個決策日（每 {HOLD} 個交易日一次），平均可交易 {r.n.mean():.0f} 檔 ===")
        ic = r.ic.dropna()
        lines.append(f"IC（模型分數與超額報酬的排名相關）：平均 {ic.mean():+.4f}，t={ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))):+.2f}，"
                     f"正的比例 {(ic > 0).mean():.0%}")
        lines.append(f"0050 同期間：平均每期 {r.bench.mean() * 100:+.2f}%，全體等權（不計成本）{r.univ.mean() * 100:+.2f}%")
        rand = np.array(r.rand.tolist())            # (期數, N_RANDOM * len(TOPKS))
        for ki, k in enumerate(TOPKS):
            lines.append(f"-- 前 {k} 名等權 --")
            rand_k = rand[:, ki::len(TOPKS)]
            for cname, c in COST.items():
                for name, col in (("模型", f"m{k}"), ("動能60日", f"mom{k}")):
                    ex = r[col] - c - r.bench
                    s = stats(ex)
                    lines.append(f"  [{cname}] {name:<6} 每期超額 {s['mean%']:+.2f}%（t={s['t']:+.2f}，年化 {s['年化%']:+.1f}%，"
                                 f"勝率 {s['勝率%']:.0f}%），組合最大回撤 {mdd(r[col] - c):.0%}")
                dist = (rand_k - c - r.bench.values[:, None]).mean(axis=0)
                model_mean = (r[f"m{k}"] - c - r.bench).mean()
                lines.append(f"  [{cname}] 隨機挑 {k} 檔：每期超額平均 {dist.mean() * 100:+.2f}%，"
                             f"模型勝過 {(dist < model_mean).mean():.0%} 的隨機組合")
        lines.append(f"  0050 買進持有最大回撤 {mdd(r.bench):.0%}")
    return "\n".join(lines)


def main():
    con = sqlite3.connect(DB)
    ids = json.loads(PILOT.read_text(encoding="utf-8"))
    print("建立特徵與標籤…", flush=True)
    X, bench, cal = build(con, ids)
    print("走動式驗證…", flush=True)
    res, picks = walk_forward(X, bench, cal)

    OUT.mkdir(parents=True, exist_ok=True)
    text = report(res)
    (OUT / "pilot_v1_report.txt").write_text(text, encoding="utf-8")
    picks.to_csv(OUT / "pilot_v1_predictions.csv")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
