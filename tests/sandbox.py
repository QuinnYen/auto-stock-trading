"""在沙盒中執行入口腳本：資料庫、輸出資料夾、帳本都指向暫存位置，真實資料與紀錄不會被讀寫。

用途：測試「驗證集防護有效」時，萬一防護被弄壞（或測試植入錯誤時被移除），腳本也只會在空資料庫上立刻失敗，
不會真的跑出驗證期結果、不會寫入 trials.csv 或報告。（2026-10-09 曾因沒有沙盒，植入錯誤的測試真的跑了驗證期。）
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_WRAPPER = r'''
import runpy, sys
from pathlib import Path
root = Path(sys.argv[1]); tmp = Path(sys.argv[2]); script = sys.argv[3]
sys.path.insert(0, str(root / "src" / "data")); sys.path.insert(0, str(root / "src" / "backtest"))
import prices
prices.DB = tmp / "empty.db"
prices.OUT = tmp / "experiments"
import holdout
holdout.LEDGER = tmp / "experiments" / "holdout_access.log"
sys.argv = [script] + sys.argv[4:]
runpy.run_path(str(root / "src" / "backtest" / script), run_name="__main__")
'''


def run_sandboxed(script: str, *args: str, timeout: int = 60):
    """回傳 (CompletedProcess, 暫存資料夾內的檔案清單)；暫存資料夾在回傳前才刪除。"""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        r = subprocess.run([sys.executable, "-B", "-c", _WRAPPER, str(ROOT), str(tmp), script, *args],
                           capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        return r, sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file())
