"""Serial single-GPU main-table reproduction queue for the remaining nine models."""

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
STATE = HERE / "remaining_models_state.json"
LOG_DIR = HERE / "remaining_models_logs"
MODELS = ("MLP", "iTransformer", "PatchTST", "CNN", "CPBiGRU",
          "CPBiLSTM", "CPGRU", "CPLSTM", "CPTransformer")


def now():
    return datetime.now().isoformat(timespec="seconds")


def save(payload):
    payload["updated_at"] = now()
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


def log_progress(path):
    if not path.is_file():
        return {}
    data = path.read_bytes().decode("utf-8", errors="replace")
    epochs = re.findall(r"(?m)^epoch=(\d+) .*?val_mape=([0-9.]+)", data)
    run_dirs = re.findall(r"(?m)^RUN_DIR=(.+)$", data)
    stages = [tag for tag in ("loading train prefixes", "loading validation prefixes",
                              "loading test prefixes") if tag in data]
    result = {"log_bytes": path.stat().st_size,
              "last_stage": stages[-1] if stages else None,
              "last_epoch": int(epochs[-1][0]) if epochs else None,
              "last_val_mape": float(epochs[-1][1]) if epochs else None,
              "run_dir": run_dirs[-1].strip() if run_dirs else None}
    return result


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if STATE.is_file():
        payload = json.loads(STATE.read_text(encoding="utf-8"))
        if tuple(payload.get("model_order", [])) != MODELS:
            raise RuntimeError("Existing queue model order differs; inspect before resuming")
    else:
        payload = {"state": "starting", "started_at": now(),
                   "model_order": list(MODELS), "models": {}}
    for name in MODELS:
        previous = payload["models"].get(name, {})
        result_path = Path(previous.get("run_dir") or ".") / "metrics.json"
        if previous.get("state") == "complete" and result_path.is_file():
            continue
        log = LOG_DIR / f"{name}.log"
        command = [sys.executable, "-u", str(HERE / "run_single_gpu.py"),
                   "--data-root", r"C:\BatteryLife_v4", "--model", name,
                   "--epochs", "100", "--seed", "2021"]
        payload["state"] = "running"
        payload["current_model"] = name
        payload["models"][name] = {"state": "running", "started_at": now(),
                                   "log": str(log), "command": command}
        save(payload)
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=HERE.parents[1], stdout=output,
                                       stderr=subprocess.STDOUT)
            payload["models"][name]["pid"] = process.pid
            save(payload)
            while process.poll() is None:
                time.sleep(30)
                payload["models"][name].update(log_progress(log))
                save(payload)
            exit_code = process.returncode
        row = payload["models"][name]
        row.update(log_progress(log))
        row.update({"exit_code": exit_code, "finished_at": now(),
                    "state": "complete" if exit_code == 0 else "failed"})
        if exit_code == 0:
            result_path = Path(row["run_dir"]) / "metrics.json"
            metrics = json.loads(result_path.read_text(encoding="utf-8"))
            row["best_epoch"] = metrics["best_epoch"]
            row["test_mape"] = metrics["test"]["mape_fraction"]
            row["test_accuracy_15"] = metrics["test"]["accuracy_15_fraction"]
        save(payload)
    payload["current_model"] = None
    payload["state"] = "complete" if all(
        payload["models"].get(name, {}).get("state") == "complete" for name in MODELS
    ) else "completed_with_failures"
    save(payload)
    return 0 if payload["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
