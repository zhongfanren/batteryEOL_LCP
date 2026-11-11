"""Apply the recorded optional-dependency patch without resetting local changes."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY = ROOT / "third_party/BatteryLife"
PATCH = ROOT / "patches/batterylife-optional-reformer.patch"


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={DEPENDENCY.as_posix()}", "-C", str(DEPENDENCY), *args],
        capture_output=True, text=True, encoding="utf-8",
    )


def main():
    if not (DEPENDENCY / "layers/SelfAttention_Family.py").exists():
        raise SystemExit("Initialize dependencies first: git submodule update --init --recursive")
    if git("apply", "--reverse", "--check", str(PATCH)).returncode == 0:
        print("BatteryLife optional-Reformer patch already applied.")
        return
    check = git("apply", "--check", str(PATCH))
    if check.returncode:
        raise SystemExit("Patch conflicts with local code; no changes made.\n" + check.stderr)
    applied = git("apply", str(PATCH))
    if applied.returncode:
        raise SystemExit(applied.stderr)
    print("BatteryLife optional-Reformer patch applied.")


if __name__ == "__main__":
    main()
