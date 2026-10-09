"""帳戶模擬 K1 執行器（依 docs/帳戶模擬事前約定 v5（K1）.md，2026-10-09 確認）：開發期、驗證期、最後測試期三關。

進場訊號 K1（盤整 60 天後，帶量 ≥ 2.5 倍、實體紅 K ≥ 5% 的 60 日新高突破，定義見 signal_scan_v4.py）。
開發期 4 次正式試驗：持有 10／20 個交易日 × 結構 S2（3 檔）／S4（6 檔），帳戶停機線 20%、單檔收盤停損 15%、種子 1～20。
通過條件 7 條（含滑價 0.5% 壓力測試）；偏好順序 S2 ＞ S4、同結構 h＝20 ＞ h＝10。

三關順序由結果檔鎖住：開發期結束寫 k1_dev_result.json（通過者與選出的組合）；驗證期必須帶 --combo 且與之相符、結果寫 k1_val_result.json；
最後測試期需驗證期通過且組合相同。任何一關的前置結果不存在／未通過／組合不符／已執行過，一律在寫入帳本之前拒絕。

用法：python src/backtest/account_sim_k1.py
      python src/backtest/account_sim_k1.py --period val --val-confirm --combo K1-20-S2
      python src/backtest/account_sim_k1.py --period final --final-confirm --combo K1-20-S2
報告：data/experiments/acct5_dev_report.txt／acct5_val_report.txt／acct5_final_report.txt
"""
import argparse
import functools
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import account  # noqa: E402
import account_sim as sim  # noqa: E402
import account_sim_v2 as v2  # noqa: E402
import holdout  # noqa: E402
import prices  # noqa: E402
import signal_scan_v4 as v4  # noqa: E402
import strong_stocks as ss  # noqa: E402

START = account.START_CAPITAL
N_SEEDS = sim.N_SEEDS
MIN_FINAL_DEV = START * 1.05 ** 10
MIN_FINAL_VAL = START * 1.05 ** 5
MIN_POS_YEARS_DEV, MIN_POS_YEARS_LATER = 6, 3
MAX_MDD = 0.40
STRESS_MIN = START                     # 滑價 0.5% 時期末淨值中位數至少不虧本

COMBOS = [sim.Rule(f"K1-{h}-{s}", f"K1 盤整後帶量長紅突破持有 {h} 天｜{v2.STRUCTS[s][0]} 檔、停機線 {v2.STRUCTS[s][1]:.0%}", h)
          for h in (10, 20) for s in v2.STRUCTS]
PREFERENCE = ["K1-20-S2", "K1-10-S2", "K1-20-S4", "K1-10-S4"]      # S2 ＞ S4；同結構 h＝20 ＞ h＝10


def pick_preferred(passed_ids):
    ranked = [i for i in PREFERENCE if i in set(passed_ids)]
    return ranked[0] if ranked else None


def judge_k1(finals, control_finals, years_ret, mdd, *, stress_finals, stage="dev", n_days=None):
    """事前約定 v5 的 7 條通過條件；回傳 [(說明, 是否成立)]。stage ＝ dev／val／final；final 需帶模擬交易日數 n_days。"""
    if stage == "dev":
        min_final, min_years = MIN_FINAL_DEV, MIN_POS_YEARS_DEV
    elif stage == "val":
        min_final, min_years = MIN_FINAL_VAL, MIN_POS_YEARS_LATER
    else:
        min_final, min_years = START * 1.05 ** (n_days / 245), MIN_POS_YEARS_LATER
    checks = v2.judge2(finals, control_finals, years_ret, mdd, min_final=min_final, min_pos_years=min_years, max_mdd=MAX_MDD)
    sm = float(np.median(stress_finals))
    return checks + [(f"滑價 0.5% 時期末中位數 {sm:,.0f} ≥ {STRESS_MIN:,.0f}（至少不虧本）", bool(sm >= STRESS_MIN))]


def k1_entry(d):
    """K1 進場訊號矩陣（日 × 股），由還原後的 O、C、V 寬表計算。"""
    return v4.k1_signal(d["O"].to_numpy(dtype="float64"), d["C"].to_numpy(dtype="float64"), d["V"].to_numpy(dtype="float64"))


def prepare_k1(con):
    ctx = sim.prepare(con)
    ctx["entries"]["K1"] = k1_entry(ctx["d"])
    return ctx


def run_combo(ctx, combo, controls, lo, hi, seeds):
    """跑一個組合：回傳 (照規則的 Result 清單, 隨機進場對照, 敏感度情境)。對照只取決於（持有期、結構），算過就快取。"""
    _, h, sname = combo.sid.split("-")
    h = int(h)
    entry, mkt, std = ctx["entries"]["K1"], ctx["mkt"], ctx["std"]
    if (h, sname) not in controls:
        controls[(h, sname)] = sim.run_control(mkt, std, v2.make_cfg2(h, sname), lo, hi, seeds)
    res = sim.run_seeds(mkt, entry, std, v2.make_cfg2(h, sname), lo, hi, seeds)
    extra = {"滑價 0%": sim.run_seeds(mkt, entry, std, v2.make_cfg2(h, sname, slip=0.0), lo, hi, seeds),
             "滑價 0.5%": sim.run_seeds(mkt, entry, std, v2.make_cfg2(h, sname, slip=0.005), lo, hi, seeds),
             "不設單檔停損": sim.run_seeds(mkt, entry, std, v2.make_cfg2(h, sname, stop_pct=1.0), lo, hi, seeds)}
    return res, controls[(h, sname)], extra


def summarize_combo(combo, results, control, extra, cal, lo, stage="dev", n_days=None):
    judge = functools.partial(judge_k1, stress_finals=sim.finals_of(extra["滑價 0.5%"]), stage=stage, n_days=n_days)
    return sim.summarize_rule(combo, results, control, extra, cal, lo, judge_fn=judge, title=f"■ 組合 {combo.sid}：{combo.name}")


def dev_result(passed_ids):
    """開發期結果檔內容：通過的組合與依偏好順序選出的組合。"""
    return {"passed": list(passed_ids), "chosen": pick_preferred(passed_ids)}


def stage_result(combo_id, ok):
    """驗證期／最後測試期結果檔內容。"""
    return {"combo": combo_id, "passed": bool(ok)}


def stage_config(stage):
    cfg = {"dev": ("開發期", "acct5-dev", "acct5_dev_report.txt"), "val": ("驗證期", "acct5-val", "acct5_val_report.txt"),
           "final": ("最後測試期", "acct5-final", "acct5_final_report.txt")}[stage]
    return {"period": holdout.PERIODS[stage], "name": cfg[0], "tag": cfg[1], "report": cfg[2]}


# ---------------------------------------------------------------- 三關順序鎖
def save_result(name, obj, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def load_result(name, out_dir):
    p = out_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def check_gate(stage, combo_id, out_dir):
    """開發期不需前置；驗證期需開發期結果檔且組合 ＝ 選出者、尚未跑過；最後測試期需驗證期通過且組合相同、尚未跑過。不符就 SystemExit。"""
    if stage == "dev":
        return
    if stage == "val":
        if load_result("k1_val_result.json", out_dir) is not None:
            raise SystemExit("驗證期已執行過，只能跑一次（k1_val_result.json 已存在）")
        dev = load_result("k1_dev_result.json", out_dir)
        if dev is None:
            raise SystemExit("找不到開發期結果（k1_dev_result.json）：需先完成開發期")
        if not dev.get("chosen"):
            raise SystemExit("開發期沒有任何組合通過，不進驗證期")
        if dev["chosen"] != combo_id:
            raise SystemExit(f"依事前約定 v5 第 5 節，進驗證期的組合是 {dev['chosen']}，不是 {combo_id}")
        return
    if load_result("k1_final_result.json", out_dir) is not None:
        raise SystemExit("最後測試期已執行過，只能跑一次（k1_final_result.json 已存在）")
    val = load_result("k1_val_result.json", out_dir)
    if val is None or not val.get("passed"):
        raise SystemExit("驗證期結果不存在或未通過：不進最後測試期")
    if val.get("combo") != combo_id:
        raise SystemExit(f"最後測試期的組合必須與驗證期相同（{val.get('combo')}），不是 {combo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "val", "final"], default="dev")
    ap.add_argument("--val-confirm", action="store_true")
    ap.add_argument("--final-confirm", action="store_true")
    ap.add_argument("--combo", choices=[c.sid for c in COMBOS])
    args = ap.parse_args()
    if args.period != "dev" and not args.combo:
        ap.error("驗證期與最後測試期需指定 --combo（依事前約定 v5 選出的組合）")
    check_gate(args.period, args.combo, ss.OUT)                       # 先檢查前置關卡，再寫帳本
    holdout.require_access(args.period, "account_sim_k1", val_confirm=args.val_confirm, final_confirm=args.final_confirm)
    cfg_s = stage_config(args.period)
    start, end = cfg_s["period"]
    combos = COMBOS if args.period == "dev" else [c for c in COMBOS if c.sid == args.combo]
    con = sqlite3.connect(prices.DB)
    ctx = prepare_k1(con)
    cal = ctx["cal"]
    lo = int(cal.searchsorted(pd.Timestamp(start)))
    hi = int(cal.searchsorted(pd.Timestamp(end), side="right"))
    assert start <= str(cal[lo].date()) and str(cal[hi - 1].date()) <= end, "模擬期間超出指定範圍"
    seeds = range(1, N_SEEDS + 1)
    controls, body, summary = {}, [], []
    for combo in combos:
        t0 = time.time()
        res, ctrl, extra = run_combo(ctx, combo, controls, lo, hi, seeds)
        lines, passed, stat = summarize_combo(combo, res, ctrl, extra, cal, lo, stage=args.period, n_days=hi - lo)
        body += lines + [""]
        summary.append((combo, passed, stat))
        _, h, sname = combo.sid.split("-")
        ss.log_trial({"time": datetime.now().isoformat(timespec="seconds"), "period": cfg_s["tag"], "version": combo.sid,
                      "exit": f"h={h} stop15% line{v2.STRUCTS[sname][1]:.0%} slots={v2.STRUCTS[sname][0]}", "slip": v2.SLIP,
                      "sig": combo.name, "cfg": f"seeds={N_SEEDS}", "cagr": stat["cagr"], "mdd": stat["mdd"],
                      "triggers": stat["triggers"], "trades": stat["trades"], "win": stat["win"],
                      "avg_net": stat["avg_net"], "avg_ret": stat["avg_ret"]})
        print(f"組合 {combo.sid} 完成，{time.time() - t0:.0f} 秒", flush=True)

    bh_final, bh_mdd = v2.bh_0050(con, cal, lo, hi)
    last = str(cal[hi - 1].date())
    head = [f"# 帳戶模擬 v5（K1）{cfg_s['name']}報告（{start}～{min(end, last)}，本金 {START:,.0f} 元）", "",
            f"參考：同期間買進持有 0050，2 萬元變 {bh_final:,.0f} 元，最大回檔 {bh_mdd:.0%}（只是參考，不是通過條件）。", ""]
    if args.period == "dev":
        passed_ids = [c.sid for c, ok, _ in summary if ok]
        chosen = dev_result(passed_ids)["chosen"]
        save_result("k1_dev_result.json", dev_result(passed_ids), ss.OUT)
        head += [f"【白話摘要】4 個組合中 **{len(passed_ids)} 個**通過全部 7 條件。"
                 + (f"依偏好順序進驗證期的是 **{chosen}**（尚未執行）。" if chosen else "沒有任何組合通過：K1 放棄，不進驗證期；依事前約定 v5 第 8 節寫專案結論。"), "",
                 "| 組合 | 期末中位數 | 年化 | 最大回檔 | 停機線觸發 | 結論 |", "|---|---|---|---|---|---|"]
        head += [f"| {c.sid} | {st['final']:,.0f} 元 | {st['cagr']:+.1%} | {st['mdd']:.0%} | {st['triggers']} 次 | {'✅ 通過' if ok else '❌ 不通過'} |"
                 for c, ok, st in summary]
        head.append("")
    else:
        combo, ok, st = summary[0]
        save_result(f"k1_{args.period}_result.json", stage_result(combo.sid, ok), ss.OUT)
        nxt = ("進最後測試期（尚未執行）。" if args.period == "val" else "三關都通過：進入紙上交易觀察，不下真單；是否實盤由你依實際成交與 0050 對照決定。") if ok \
            else "依約定放棄 K1，不換參數重試；專案結論依事前約定 v5 第 8 節。"
        head += [f"【白話摘要】{combo.sid} 在{cfg_s['name']}**{'通過' if ok else '不通過'}**：2 萬元照規則做約 {max(1, round((hi - lo) / 245, 1))} 年，"
                 f"20 次模擬的期末中位數 {st['final']:,.0f} 元（年化 {st['cagr']:+.1%}），最大回檔 {st['mdd']:.1%}。{nxt}", ""]
    head += ["假設：每檔買淨值的 1/3（S2）或 1/6（S4）；收盤跌破進場價 15% 隔天開盤賣；帳戶停機線 20%（停 20 日後人工重啟）；"
             "單邊滑價 0.25%（壓力測試 0.5%）、手續費 3 折（每筆最低 1 元）、賣出證交稅 0.3%；持股遇處置期間賣出再加 1% 滑價；"
             "同日訊號多於空位時依種子隨機排序。", ""]
    text = "\n".join(head + body)
    ss.OUT.mkdir(parents=True, exist_ok=True)
    (ss.OUT / cfg_s["report"]).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
