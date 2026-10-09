"""資料切分與驗證集防護：開發期可隨意執行；驗證期、最後測試期必須明確確認，且每次存取都寫入帳本。

帳本 data/experiments/holdout_access.log 只附加（時間、腳本、期間），用來事後核對驗證集被看過幾次。
"""
from datetime import datetime
from pathlib import Path

PERIODS = {"dev": ("2007-01-01", "2016-12-31"), "val": ("2017-01-01", "2021-12-31"), "final": ("2022-01-01", "2099-12-31")}
LEDGER = Path(__file__).resolve().parents[2] / "data" / "experiments" / "holdout_access.log"
FLAG = {"val": "--val-confirm", "final": "--final-confirm"}


def require_access(period: str, script: str, *, val_confirm: bool = False, final_confirm: bool = False,
                   ledger: Path = LEDGER) -> None:
    """開發期直接放行；其餘期間未確認就中止，確認後寫入帳本。"""
    if period not in PERIODS:
        raise SystemExit(f"未知期間：{period}")
    if period == "dev":
        return
    confirmed = {"val": val_confirm, "final": final_confirm}[period]
    if not confirmed:
        raise SystemExit(f"{period} 期間是保留的驗證資料，需先有已確認的事前約定，再明確加上 {FLAG[period]}")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{script}\t{period}\n")
