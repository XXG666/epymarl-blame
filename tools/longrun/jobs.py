"""Run records on disk, the jobs' processes and state, launching, and reading a job's logged metrics.

A run trains two jobs, one per entry of KINDS ("qmix" and "blame"). Each job is started once, at step 0.

A run's record (rec in the code) is the dictionary stored in results/long_runs/runs/<id>/run.json:
    id, seed, t_max, note, created    what was asked for, and when
    commit, src, dirty                the code the run started with (see git_state)
    attempts                          {"qmix": [...], "blame": [...]}: an empty list until the job is started,
                                      then one entry,
                                      {"n": 1, "start_t": 0, "commit": the commit then,
                                       "command": its command line}
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import time

from .config import COMMON_SETTINGS, BENCH_SETTINGS, KINDS, ROOT, RUNS_DIR, SACRED_DIR


# ---------------------------------------------------------------------------------------------------------
# Paths and names
# ---------------------------------------------------------------------------------------------------------

def run_dir(run_id):
    """In: a run's ID. Out: the path of the run's folder."""
    return RUNS_DIR / run_id


def label(run_id, kind):
    """Tag passed to EPyMARL as label=...: identifies the job's processes and its Sacred results."""
    return f"longrun.{run_id}.{kind}"


def job_log(run_id, kind):
    """In: a run's ID and the job's kind. Out: the path of the file that holds everything the job printed."""
    return run_dir(run_id) / f"{kind}.1.log"


# ---------------------------------------------------------------------------------------------------------
# Reading and writing files
# ---------------------------------------------------------------------------------------------------------

def load_run(run_id):
    """In: a run's ID. Out: the run's record. Ends the program if there is no such run."""
    path = run_dir(run_id) / "run.json"
    if not path.exists():
        sys.exit(f"no run {run_id}")
    return json.loads(path.read_text())


def unique_ids(ids):
    """In: a list of run IDs. Out: the same IDs in the same order, each one only once."""
    unique = []
    for run_id in ids:
        if run_id not in unique:
            unique.append(run_id)
    return unique


def load_runs(ids):
    """In: a list of run IDs. Out: the list of their records. Ends the program if one of them is missing."""
    recs = []
    for run_id in ids:
        recs.append(load_run(run_id))
    return recs


def save_run(rec):
    """In: a run's record. Writes it to the run's run.json."""
    path = run_dir(rec["id"]) / "run.json"
    path.write_text(json.dumps(rec, indent=2))


def read_json(path, tries=5):
    """Sacred rewrites metrics.json while a job runs, so a read can catch half a file: retry briefly.
    In: the path of a JSON file and how often to try. Out: the file's content, or None if no try worked."""
    for try_number in range(tries):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            time.sleep(0.5)
    return None


def read_log(path):
    """In: the path of a job's log. Out: its text, or "" if the file does not exist."""
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


# ---------------------------------------------------------------------------------------------------------
# The state of the code
# ---------------------------------------------------------------------------------------------------------

def git_output(arguments):
    """In: the words of a git command, without "git" itself. Out: what git prints for it in the repository,
    without the white space around it."""
    result = subprocess.run(["git"] + arguments, cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip()


def git_state():
    """(commit, git's hash of src/ at that commit, whether src/ has uncommitted changes)."""
    commit = git_output(["rev-parse", "--short", "HEAD"])
    src_hash = git_output(["rev-parse", "--short", "HEAD:src"])
    uncommitted_files = git_output(["status", "--short", "src"])
    dirty = uncommitted_files != ""
    return commit, src_hash, dirty


def commit_text(rec):
    """In: a run's record. Out: the commit the run started with, followed by "*" if src/ had uncommitted
    changes then."""
    if rec["dirty"]:
        return rec["commit"] + "*"
    return rec["commit"]


# ---------------------------------------------------------------------------------------------------------
# The state of the jobs
# ---------------------------------------------------------------------------------------------------------

def live_jobs():
    """Out: the list of the labels of every job of this tool that is running right now."""
    alive = []
    # /proc has one folder per running process, named by the process's ID
    for process_dir in pathlib.Path("/proc").iterdir():
        try:
            # the file cmdline holds the words of the process's command line, separated by zero bytes
            command_bytes = (process_dir / "cmdline").read_bytes()
            command_words = command_bytes.decode().split("\0")
            working_dir = os.readlink(process_dir / "cwd")
        except OSError:
            # not a process, or a process that this user may not look at, or one that just ended
            continue
        job_label = None
        for word in command_words:
            if word.startswith("label=longrun."):
                job_label = word.split("=", 1)[1]  # "label=longrun.a1.qmix" -> "longrun.a1.qmix"
                break
        if job_label is None:
            continue
        if "src/main.py" in command_words and pathlib.Path(working_dir) == ROOT:
            alive.append(job_label)
    return alive


def job_state(rec, kind, alive):
    """(status, steps reached); status is running / done / stopped / crashed / not started.
    In: a run's record, the job's kind and the result of live_jobs()."""
    if len(rec["attempts"][kind]) == 0:
        return "not started", 0
    log_text = read_log(job_log(rec["id"], kind))

    # the log names the step again and again (t_env: 12345): the largest one is how far the job got
    steps_reached = 0
    for digits in re.findall(r"t_env:\s*(\d+)", log_text):
        steps_reached = max(steps_reached, int(digits))

    if "Finished Training" in log_text:
        return "done", max(steps_reached, rec["t_max"])
    if label(rec["id"], kind) in alive:
        return "running", steps_reached
    if "Traceback" in log_text:
        return "crashed", steps_reached
    return "stopped", steps_reached


# ---------------------------------------------------------------------------------------------------------
# Starting a job
# ---------------------------------------------------------------------------------------------------------

def training_command(rec, kind, job_dir):
    """In: a run's record, the job's kind and the job's folder. Out: the command line that trains this job
    from step 0, as a list of words."""
    t_max = rec["t_max"]
    seed = rec["seed"]
    job_label = label(rec["id"], kind)
    results_path = job_dir.relative_to(ROOT)
    command = [sys.executable, "src/main.py", "--config=qmix", "--env-config=gymma", "with"]
    command = command + COMMON_SETTINGS + BENCH_SETTINGS + KINDS[kind]
    command.append(f"t_max={t_max}")
    command.append(f"seed={seed}")
    command.append(f"label={job_label}")
    command.append(f"local_results_path={results_path}")
    return command


def launch(rec, kind):
    """Start a job in the background, at step 0.
    In: a run's record and the job's kind. The job's entry is added to the record, and the record is
    saved."""
    job_dir = run_dir(rec["id"]) / kind
    job_dir.mkdir(parents=True, exist_ok=True)
    command = training_command(rec, kind, job_dir)

    # everything the job prints goes to the job's log; with a session of its own the job keeps going
    # after the terminal closes
    log_file = open(job_log(rec["id"], kind), "w")
    subprocess.Popen(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)

    commit, src_hash, dirty = git_state()
    # the first word is the full path of this Python: the record keeps the short "python" instead
    command_text = "python " + " ".join(command[1:])
    rec["attempts"][kind].append({"n": 1, "start_t": 0, "commit": commit, "command": command_text})
    save_run(rec)


# ---------------------------------------------------------------------------------------------------------
# A job's logged metrics
# ---------------------------------------------------------------------------------------------------------

def job_metrics(rec, kind):
    """In: a run's record and the job's kind. Out: everything Sacred logged for this job,
    {metric name: {"steps": [...], "values": [...]}}; an empty dictionary if there is nothing to read."""
    log_text = read_log(job_log(rec["id"], kind))
    # the log names the numbered folder Sacred gave this training process
    sacred_ids = re.findall(r'Started run with ID "(\d+)"', log_text)
    if len(sacred_ids) == 0:
        return {}
    logged = read_json(SACRED_DIR / sacred_ids[-1] / "metrics.json")
    if not logged:
        return {}  # no readable file, or nothing logged yet
    # Sacred stores the time of every value as well: keep the steps and the values only
    metrics = {}
    for name in logged:
        metrics[name] = {"steps": logged[name]["steps"], "values": logged[name]["values"]}
    return metrics
