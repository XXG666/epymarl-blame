"""Run records on disk, the jobs' processes and state, launching, and reading a job's logged metrics."""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

from .config import COMMON_SETTINGS, BENCH_SETTINGS, KINDS, ROOT, RUNS_DIR, SACRED_DIR


def run_dir(run_id):
    return RUNS_DIR / run_id


def label(run_id, kind):
    """Tag passed to EPyMARL as label=...: identifies the job's processes and its Sacred results."""
    return f"longrun.{run_id}.{kind}"


def attempt_log(run_id, kind, n):
    return run_dir(run_id) / f"{kind}.{n}.log"


def load_run(run_id):
    path = run_dir(run_id) / "run.json"
    if not path.exists():
        sys.exit(f"no run {run_id}")
    return json.loads(path.read_text())


def save_run(rec):
    (run_dir(rec["id"]) / "run.json").write_text(json.dumps(rec, indent=2))


def read_json(path, tries=5):
    """Sacred rewrites metrics.json while a job runs, so a read can catch half a file: retry briefly."""
    for _ in range(tries):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            time.sleep(0.5)
    return None


def git_state():
    """(commit, git's hash of src/ at that commit, whether src/ has uncommitted changes)."""
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD"), git("rev-parse", "--short", "HEAD:src"), bool(git("status", "--short", "src"))


def live_jobs():
    """{label: pid} of every job of this tool that is running right now."""
    alive = {}
    for d in pathlib.Path("/proc").iterdir():
        try:
            args = (d / "cmdline").read_bytes().decode().split("\0")
            cwd = os.readlink(d / "cwd")
        except OSError:
            continue
        tag = next((a.split("=", 1)[1] for a in args if a.startswith("label=longrun.")), None)
        if tag and "src/main.py" in args and pathlib.Path(cwd) == ROOT:
            alive[tag] = int(d.name)
    return alive


def job_state(rec, kind, alive):
    """(status, steps reached); status is running / done / stopped / crashed / not started."""
    attempts = rec["attempts"][kind]
    if not attempts:
        return "not started", 0
    path = attempt_log(rec["id"], kind, attempts[-1]["n"])
    text = path.read_text(errors="ignore") if path.exists() else ""
    t = max([int(s) for s in re.findall(r"t_env:\s*(\d+)", text)] + [attempts[-1]["start_t"]])
    if "Finished Training" in text:
        return "done", max(t, rec["t_max"])
    if label(rec["id"], kind) in alive:
        return "running", t
    return ("crashed" if "Traceback" in text else "stopped"), t


def counted_attempts(rec, kind):
    """The attempts whose results count: the latest one that started at step 0 and every one after it."""
    attempts = rec["attempts"][kind]
    first = max((i for i, att in enumerate(attempts) if att["start_t"] == 0), default=0)
    return attempts[first:]


def latest_checkpoint(run_id, kind):
    """(checkpoint directory, step) of the job's newest complete checkpoint, or None."""
    steps = [a.parent for a in (run_dir(run_id) / kind / "models").glob("*/*/agent.th")
             if a.parent.name.isdigit() and (a.parent / "mixer.th").exists() and (a.parent / "opt.th").exists()]
    best = max(steps, key=lambda s: int(s.name), default=None)
    return (best, int(best.name)) if best else None


def launch(rec, kind, resume_from=None):
    """Start one attempt of a job in the background. resume_from: (checkpoint dir, step) or None for step 0."""
    attempts = rec["attempts"][kind]
    n = len(attempts) + 1
    jdir = run_dir(rec["id"]) / kind
    jdir.mkdir(parents=True, exist_ok=True)
    cmd = ([sys.executable, "src/main.py", "--config=qmix", "--env-config=gymma", "with"]
           + COMMON_SETTINGS + BENCH_SETTINGS + KINDS[kind]
           + [f"t_max={rec['t_max']}", f"seed={rec['seed']}", f"label={label(rec['id'], kind)}",
              f"local_results_path={jdir.relative_to(ROOT)}"])
    start_t = 0
    if resume_from:
        # checkpoint folder names contain spaces and colons: copy the step to a plain path first
        src_dir, start_t = resume_from
        shutil.rmtree(jdir / "resume_from", ignore_errors=True)
        shutil.copytree(src_dir, jdir / "resume_from" / str(start_t))
        cmd += [f"checkpoint_path={(jdir / 'resume_from').relative_to(ROOT)}", f"load_step={start_t}"]
    subprocess.Popen(cmd, cwd=ROOT, stdout=open(attempt_log(rec["id"], kind, n), "w"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    attempts.append({"n": n, "start_t": start_t, "commit": git_state()[0], "command": "python " + " ".join(cmd[1:])})
    save_run(rec)
    return start_t


def job_metrics(rec, kind):
    """The job's logged metrics, stitched over its counted attempts: each attempt contributes its values
    from its own start step up to the step the next attempt resumed from."""
    attempts = counted_attempts(rec, kind)
    merged = {}
    for i, att in enumerate(attempts):
        path = attempt_log(rec["id"], kind, att["n"])
        ids = re.findall(r'Started run with ID "(\d+)"', path.read_text(errors="ignore")) if path.exists() else []
        m = read_json(SACRED_DIR / ids[-1] / "metrics.json") if ids else None
        if not m:
            continue
        lo, hi = att["start_t"], attempts[i + 1]["start_t"] if i + 1 < len(attempts) else float("inf")
        for key, series in m.items():
            dst = merged.setdefault(key, {"steps": [], "values": []})
            for t, v in zip(series["steps"], series["values"]):
                if lo <= t < hi:
                    dst["steps"].append(t)
                    dst["values"].append(v)
    return merged
