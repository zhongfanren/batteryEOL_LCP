"""Wait for verified v4 staging, then run one full-split DLinear seed."""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from audit_mix_large import audit


HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:\BatteryLife_v4")
STATE = HERE / "pipeline_state.json"


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


def ready():
    inventory = audit(ROOT)
    return all(inventory["phases"][phase]["files_present"] == inventory["phases"][phase]["expected"]
               for phase in ("train", "val", "test"))


def main():
    status("waiting_for_v4_downloads")
    deadline = time.monotonic() + 48 * 3600
    while not ready():
        if time.monotonic() > deadline:
            status("download_timeout")
            return 2
        time.sleep(60)
    status("running_full_split_dlinear")
    log = HERE / "full_split_console.log"
    command = [sys.executable, "-u", str(HERE / "run_single_gpu.py"),
               "--data-root", str(ROOT), "--epochs", "100", "--seed", "2021"]
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=HERE.parents[1], stdout=stream, stderr=subprocess.STDOUT)
    status("full_split_complete" if result.returncode == 0 else "full_split_failed",
           exit_code=result.returncode, log=str(log))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
