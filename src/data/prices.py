"""價格資料的共用載入：路徑常數、還原價（除息、減資、分割、面額變更）、寬表轉換。

原本放在 src/model/pilot_v1.py，因被回測與模型共同使用而獨立出來（函式內容未改）。
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "market.db"
PILOT = ROOT / "data" / "pilot_stocks.json"
OUT = ROOT / "data" / "experiments"
CAL_START = "2005-01-01"


def sql(con, query):
    return pd.read_sql(query, con)


def load_adjusted(con, ids):
    q = ",".join(f"'{s}'" for s in ids)
    px = sql(con, f"SELECT stock_id,date,open,high,low,close,volume,money FROM stock_price "
                  f"WHERE stock_id IN ({q}) AND close>0 AND open>0")
    div = sql(con, f"SELECT stock_id,date,reference_price/before_price AS k FROM dividend_result "
                   f"WHERE stock_id IN ({q}) AND before_price>0 AND reference_price>0")
    cap = sql(con, f"SELECT stock_id,date,post_reduction_ref_price/last_close AS k FROM capital_reduction "
                   f"WHERE stock_id IN ({q}) AND last_close>0 AND post_reduction_ref_price>0")
    split = sql(con, f"SELECT stock_id,date,after_price/before_price AS k FROM split_price "
                     f"WHERE stock_id IN ({q}) AND before_price>0 AND after_price>0")
    par = sql(con, f"SELECT stock_id,date,after_ref_close/before_close AS k FROM par_value_change "
                   f"WHERE stock_id IN ({q}) AND before_close>0 AND after_ref_close>0")
    # 面額變更與分割表可能是同一事件（如 2327），以分割表為準，避免重複調整
    par = par.merge(split[["stock_id", "date"]], on=["stock_id", "date"], how="left", indicator=True)
    par = par[par["_merge"] == "left_only"].drop(columns="_merge")
    ev = pd.concat([div, cap, split, par], ignore_index=True)
    dup = ev.groupby(["stock_id", "date"]).size()
    if (dup > 1).any():
        raise RuntimeError(f"同一檔同一天有多個價格調整事件，需先釐清是否重複：\n{dup[dup > 1]}")

    parts = []
    for sid, g in px.groupby("stock_id"):
        g = g.sort_values("date").copy()
        e = ev[ev.stock_id == sid].sort_values("date")
        if len(e):
            suffix = np.append(np.cumprod(e.k.values[::-1])[::-1], 1.0)
            idx = np.searchsorted(e.date.values.astype(str), g.date.values.astype(str), side="right")
            f = suffix[idx]
        else:
            f = 1.0
        g["o"] = g.open * f
        g["h"] = g.high * f
        g["l"] = g.low * f
        g["c"] = g.close * f
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def wide(df, col, cal):
    out = df.pivot(index="date", columns="stock_id", values=col)
    out.index = pd.to_datetime(out.index)
    return out.reindex(cal)
