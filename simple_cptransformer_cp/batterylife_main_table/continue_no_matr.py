"""Run the first MATR-held-out predictor only after main-table DLinear succeeds."""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
MAIN_STATE = HERE / "pipeline_state.json"
STATE = HERE / "heldout_pipeline_state.json"
LOG = HERE / "no_matr_dlinear_console.log"


def status(state, **extra):
    payload = {"state": state, "time": datetime.now().isoformat(timespec="seconds"), **extra}
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(100):
        try:
            temporary.replace(STATE)
            break
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.1)


def main():
    status("waiting_for_main_table_dlinear")
    deadline = time.monotonic() + 48 * 3600
    while True:
        try:
            main_state = json.loads(MAIN_STATE.read_text(encoding="utf-8"))["state"]
        except (FileNotFoundError, json.JSONDecodeError):
            main_state = None
        if main_state == "full_split_complete":
            break
        if main_state in {"full_split_failed", "download_timeout"}:
            status("main_table_failed", main_state=main_state)
            return 2
        if time.monotonic() > deadline:
            status("main_table_wait_timeout")
            return 2
        time.sleep(30)
    command = [sys.executable, "-u", str(HERE / "run_single_gpu.py"),
               "--data-root", r"C:\BatteryLife_v4", "--exclude-matr",
               "--epochs", "100", "--seed", "2021"]
    status("running_no_matr_dlinear", log=str(LOG))
    with LOG.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=HERE.parents[1], stdout=stream,
                                stderr=subprocess.STDOUT)
    status("no_matr_complete" if result.returncode == 0 else "no_matr_failed",
           exit_code=result.returncode, log=str(LOG))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
