"""Run v4 Li-ion CPMLP main-table then MATR-held-out training serially."""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
STATE = HERE / "cpmlp_pipeline_state.json"


def status(state, **extra):
    payload = {"state": state, "time": datetime.now().isoformat(timespec="seconds"), **extra}
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(100):
        try:
            temporary.replace(STATE)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.1)


def run(stage, extra_args, log_name):
    log = HERE / log_name
    command = [sys.executable, "-u", str(HERE / "run_single_gpu.py"),
               "--data-root", r"C:\BatteryLife_v4", "--model", "CPMLP",
               "--epochs", "100", "--seed", "2021", *extra_args]
    status(f"running_{stage}", log=str(log))
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=HERE.parents[1], stdout=stream,
                                stderr=subprocess.STDOUT)
    status(f"{stage}_complete" if result.returncode == 0 else f"{stage}_failed",
           exit_code=result.returncode, log=str(log))
    return result.returncode


def main():
    if run("main_table_cpmlp", [], "cpmlp_main_console.log"):
        return 1
    if run("no_matr_cpmlp", ["--exclude-matr"], "cpmlp_no_matr_console.log"):
        return 1
    status("complete", main_log=str(HERE / "cpmlp_main_console.log"),
           heldout_log=str(HERE / "cpmlp_no_matr_console.log"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
