"""Audit the main repository upload; leave all research artifacts in place."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PINNED = {
    "third_party/BatteryLife": "9572e47b1d36ecb31fe58f7d2874a7355dbb6fea",
    "baselines/MambaLithium": "8f83b38b3f1fba02ce2e1be34950c898189eea77",
}
ARTIFACTS = {
    ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".npz", ".npy", ".joblib",
    ".csv", ".parquet", ".h5", ".hdf5", ".pdf", ".png", ".jpg", ".jpeg",
    ".gif", ".mp4", ".ipynb", ".zip", ".7z", ".tar", ".gz", ".bin", ".log",
    ".pyc", ".pem", ".key", ".p12", ".pfx", ".token",
}
SECRET = re.compile(
    rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
    rb"hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16}|"
    rb"sk-(?:proj-)?[A-Za-z0-9_-]{32,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


def git(*args, cwd=ROOT):
    result = subprocess.run(
        ["git", "-c", f"safe.directory={cwd.as_posix()}", "-C", str(cwd), *args],
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Audit indexed blobs, including unchanged tracked files")
    parser.add_argument("--write-local", action="store_true", help="Save full inventory under ignored .local/")
    args = parser.parse_args()
    errors, rows, dependencies = [], [], []
    if not args.staged:
        parser.error("Use --staged after reviewing and staging the intended files")
    entries = git("ls-files", "--stage", "-z").split(b"\0")
    for entry in filter(None, entries):
        metadata, raw_path = entry.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if stage != "0":
            errors.append(f"Unresolved merge: {path}")
            continue
        if mode == "160000":
            dependencies.append({"path": path, "commit": oid})
            if PINNED.get(path) != oid:
                errors.append(f"Unexpected dependency revision: {path}")
            continue
        data = git("cat-file", "blob", oid)
        rows.append({"path": path, "bytes": len(data), "git_blob": oid})
        p = Path(path)
        if p.suffix.lower() in ARTIFACTS or any(part in {".local", ".aws", ".codex", ".agents", ".vscode"} for part in p.parts):
            errors.append(f"Local artifact or workspace state staged: {path}")
        if p.name == ".env" or (p.name.startswith(".env.") and p.name != ".env.example") or p.name == "credentials.json":
            errors.append(f"Credential file staged: {path}")
        if p.name == "paper.txt":
            errors.append(f"Full extracted paper text staged: {path}")
        if len(data) > 5 * 1024**2:
            errors.append(f"File exceeds source-repository budget (5 MiB): {path}")
        if SECRET.search(data):
            errors.append(f"Possible embedded credential (content withheld): {path}")
    for path, revision in PINNED.items():
        if not any(d["path"] == path for d in dependencies):
            errors.append(f"Missing dependency gitlink: {path}")
            continue
        actual = git("rev-parse", "HEAD", cwd=ROOT / path).decode().strip()
        if actual != revision:
            errors.append(f"Working dependency differs from pin: {path}")
    patch = str(ROOT / "patches/batterylife-optional-reformer.patch")
    check = subprocess.run(
        ["git", "-c", f"safe.directory={(ROOT / 'third_party/BatteryLife').as_posix()}",
         "-C", str(ROOT / "third_party/BatteryLife"), "apply", "--reverse", "--check", patch],
        capture_output=True,
    )
    if check.returncode:
        errors.append("Recorded BatteryLife compatibility patch is not applied cleanly")
    if not rows:
        errors.append("No main-repository files in index")
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--no-index", "--stdin", "-z"],
        input=b"\0".join(r["path"].encode("utf-8") for r in rows) + b"\0",
        capture_output=True,
    )
    if ignored.returncode not in (0, 1):
        raise RuntimeError(ignored.stderr.decode("utf-8", errors="replace"))
    for path in filter(None, ignored.stdout.decode("utf-8").split("\0")):
        errors.append(f"Ignored local file is already staged: {path}")
    inventory, totals = [], defaultdict(lambda: {"files": 0, "bytes": 0})
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", ".local"}]
        for name in files:
            if name == ".git":
                continue
            p = Path(base) / name
            rel = p.relative_to(ROOT).as_posix()
            size = p.stat().st_size
            inventory.append({"path": rel, "bytes": size})
            total = totals[p.relative_to(ROOT).parts[0]]
            total["files"] += 1
            total["bytes"] += size
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Main index only; upstream gitlinks checked, upstream contents not scanned",
        "workspace_bytes": sum(r["bytes"] for r in inventory),
        "staged_files": len(rows), "staged_bytes": sum(r["bytes"] for r in rows),
        "dependencies": dependencies, "errors": errors, "top_level": dict(totals),
        "staged": rows, "local_inventory": inventory,
    }
    if args.write_local:
        out = ROOT / ".local/repository_audit"
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("workspace_bytes", "staged_files", "staged_bytes", "dependencies", "errors")}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
