r"""
Long runs -- benchmark QMIX against blame-head QMIX, managed as runs with an ID.
================================================================================

A RUN has an ID you choose (e.g. s1) and ONE seed. It runs this pair, both with that seed:

    job       what                                  flags
    qmix      benchmark QMIX                        blame_head=False split_backward=False
    blame     benchmark QMIX, blame head trained    blame_head=True  split_backward=True

The two jobs of a run are started, stopped and resumed together. For several seeds, start several
runs (e.g. s1, s2, s3) and summarise them together:  summary s1 s2 s3

    Benchmark = the tuned QMIX settings for LBF (Papoudakis et al. 2021, Table 28):
    GRU, lr 0.0003, epsilon annealed over 200k steps, soft target update 0.01,
    evaluation epsilon 0.05. Task: Foraging-2s-10x10-3p-3f-v3, on the CPU.

WHERE THINGS ARE
    results/long_runs/runs/<id>/               everything of one run:
        run.json                               its record: seed, steps, commit, note, attempts
        <job>.<attempt>.log                    console output of each attempt (job = qmix or blame)
        <job>/models/                          checkpoints every 100k steps, used by resume
        <job>/metrics.json                     the job's logged metrics, stitched over its attempts  } written by
        summary.txt, summary.png               the run's summary                                     } summary <id>
    results/long_runs/summaries/<name>/        a summary of several runs: summary.txt, summary.png
                                               (name = the IDs joined by +, or --name)
    results/sacred/qmix/<task>/<n>/            Sacred's raw results (EPyMARL always writes them here); each
                                               attempt is tagged label=longrun.<id>.<job>, so it can always be found

COMMANDS
    run <id> --seed S          start a new run (detached: it keeps going after the terminal closes)
    list                       every run: ID, seed, steps, state of both jobs; and the summaries written so far
    status <id> [<id> ...]     progress bars, updated in place until every job is done
    summary <id>               the run's results: test return of both jobs, the blame head's accuracy
                               against its lazy base, and a plot; saved in the run's folder
    summary <id> <id> ...      several runs side by side and the MEAN over them, with notes when they differ
                               in step count or code version, and a plot; saved in a new
                               folder in summaries/ (--name to choose its name)
    stop <id> [<id> ...]       stop the runs' jobs; they can be resumed
    resume <id> [<id> ...]     continue the runs' unfinished jobs from their latest checkpoint
                               (--from-scratch: start them again from step 0)
    delete <id>                remove everything of the run: its folder and its Sacred results
                               (summaries of several runs are kept: they are snapshots)

RESUME, HONESTLY
    EPyMARL reloads the agent network, the mixer and Adam from the checkpoint and continues at the
    checkpoint's step (up to 100k steps before the stop are redone). It does not save the replay
    buffer, which refills first, nor the target mixer, which restarts from the loaded mixer.
    A resumed job is close to, but not bit-identical with, one that never stopped; it is marked "r".
    resume --from-scratch starts the job again at step 0: identical to a job that never stopped, so
    it is not marked, and only the new attempt's results count.
    Resume refuses if the training code (src/) changed since the run started (--force overrides).

USAGE
    python tools/run_long.py run s1 --seed 1 --note "4-way choice loss"
    python tools/run_long.py run s2 --seed 2 --note "4-way choice loss"
    python tools/run_long.py run quick --seed 7 --t-max 500000
    python tools/run_long.py list
    python tools/run_long.py status s1 s2           # Ctrl+C stops the display only
    python tools/run_long.py summary s1             # -> results/long_runs/runs/s1/
    python tools/run_long.py summary s1 s2          # -> results/long_runs/summaries/s1+s2/
    python tools/run_long.py summary s1 s2 --name choice_loss
    python tools/run_long.py stop s1 s2
    python tools/run_long.py resume s1 s2
    python tools/run_long.py delete s1
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")  # draw into a file; WSL usually has no screen
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_KEY = "lbforaging:Foraging-2s-10x10-3p-3f-v3"
SACRED_DIR = ROOT / "results" / "sacred" / "qmix" / ENV_KEY
RUNS_DIR = ROOT / "results" / "long_runs" / "runs"
SUMMARIES_DIR = ROOT / "results" / "long_runs" / "summaries"

COMMON_SETTINGS = [
    "env_args.time_limit=50",
    f"env_args.key={ENV_KEY}",
    "use_cuda=False",  # deterministic, like every other run in this project
    "log_interval=10000",
    "runner_log_interval=10000",
    "learner_log_interval=10000",
    "test_interval=20000",
    "test_nepisode=20",
    "save_model=True",
    "save_model_interval=100000",  # checkpoints for resume
]
# Papoudakis et al. 2021, Table 28 ("Hyperparameters for QMIX with parameter sharing"), LBF column.
BENCH_SETTINGS = [
    "use_rnn=True",
    "lr=0.0003",
    "epsilon_anneal_time=200000",
    "target_update_interval_or_tau=0.01",
    "evaluation_epsilon=0.05",
]
KINDS = {
    "qmix": ["blame_head=False", "split_backward=False"],
    "blame": ["blame_head=True", "split_backward=True"],
}

# colours: blame QMIX in categorical slot 1, then the blame head's questions in slots 2-5
# ("each agent", "team vote", "who", "whether"); plain QMIX neutral grey
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
C_REF, C_INK, C_GRID = "#8a8984", "#52514e", "#e1e0d9"
LINE = {"qmix": (C_REF, "--", "benchmark QMIX"), "blame": (C_BLUE, "-", "blame QMIX")}


# ----------------------------------------------------------------------------------
# Runs, jobs and attempts on disk
# ----------------------------------------------------------------------------------
def all_runs():
    """Every run's record, oldest first."""
    recs = [json.loads(p.read_text()) for p in RUNS_DIR.glob("*/run.json")]
    return sorted(recs, key=lambda r: r["created_epoch"])


def run_dir(run_id):
    return RUNS_DIR / run_id


def load_run(run_id):
    path = run_dir(run_id) / "run.json"
    if not path.exists():
        sys.exit(f"no run {run_id}  (see: run_long.py list)")
    return json.loads(path.read_text())


def save_run(rec):
    (run_dir(rec["id"]) / "run.json").write_text(json.dumps(rec, indent=2))


def label(run_id, kind):
    """Tag passed to EPyMARL as label=...: identifies the job's processes and its Sacred results."""
    return f"longrun.{run_id}.{kind}"


def job_dir(run_id, kind):
    return run_dir(run_id) / kind


def attempt_log(run_id, kind, n):
    return run_dir(run_id) / f"{kind}.{n}.log"


def unique(ids):
    """The IDs in the order given, each once."""
    return list(dict.fromkeys(ids))


def read_json(path, tries=5):
    """Sacred rewrites metrics.json every few seconds while a run is going, so a read can catch
    half a file. Retry briefly; None if it still fails."""
    for _ in range(tries):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            time.sleep(0.5)
    return None


def git_state():
    """(commit, src, dirty): the commit, git's hash of the src/ folder at that commit, and whether
    src/ has uncommitted changes. The src hash identifies the training code: commits that change
    only tools/ or other files leave it unchanged."""
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD"), git("rev-parse", "--short", "HEAD:src"), bool(git("status", "--short", "src"))


# ----------------------------------------------------------------------------------
# Processes and job state
# ----------------------------------------------------------------------------------
def live_jobs():
    """{label: pid} for every EPyMARL process started by this tool that is alive right now."""
    alive = {}
    for d in pathlib.Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            args = (d / "cmdline").read_bytes().decode().split("\0")
            cwd = os.readlink(d / "cwd")
        except (OSError, PermissionError):
            continue
        if "src/main.py" not in args or pathlib.Path(cwd) != ROOT:
            continue
        tag = next((a.split("=", 1)[1] for a in args if a.startswith("label=longrun.")), None)
        if tag:
            alive[tag] = int(d.name)
    return alive


def job_state(rec, kind, alive):
    """(status, t_env reached, number of attempts). status: running / done / crashed / stopped / not started."""
    attempts = rec["attempts"][kind]
    if not attempts:
        return "not started", 0, 0
    path = attempt_log(rec["id"], kind, attempts[-1]["n"])
    text = path.read_text(errors="ignore") if path.exists() else ""
    t = max([int(s) for s in re.findall(r"t_env:\s*(\d+)", text)] + [attempts[-1]["start_t"]])
    done = "Finished Training" in text
    if label(rec["id"], kind) in alive and not done:
        return "running", t, len(attempts)
    if done:
        return "done", max(t, rec["t_max"]), len(attempts)
    return ("crashed" if "Traceback" in text else "stopped"), t, len(attempts)


def counted_attempts(rec, kind):
    """The attempts whose results count: the latest one that started at step 0 and every one after it.
    A restart from scratch (resume --from-scratch) replaces everything before it."""
    attempts = rec["attempts"][kind]
    first = max((i for i, att in enumerate(attempts) if att["start_t"] == 0), default=0)
    return attempts[first:]


def resumed(rec, kind):
    """True if the job's counted results include an attempt that continued from a checkpoint."""
    return any(att["start_t"] > 0 for att in counted_attempts(rec, kind))


def latest_checkpoint(run_id, kind):
    """(checkpoint directory, t_env) of the job's newest complete checkpoint, or None."""
    best = None
    for agent in (job_dir(run_id, kind) / "models").glob("*/*/agent.th"):
        step = agent.parent
        if step.name.isdigit() and (step / "mixer.th").exists() and (step / "opt.th").exists():
            if best is None or int(step.name) > best[1]:
                best = (step, int(step.name))
    return best


# ----------------------------------------------------------------------------------
# Launching
# ----------------------------------------------------------------------------------
def launch(rec, kind, resume_from=None):
    """Start one attempt of a job, detached. resume_from: (checkpoint dir, t_env) or None for step 0."""
    attempts = rec["attempts"][kind]
    n = len(attempts) + 1
    jdir = job_dir(rec["id"], kind)
    jdir.mkdir(parents=True, exist_ok=True)
    cmd = ([sys.executable, "src/main.py", "--config=qmix", "--env-config=gymma", "with"]
           + COMMON_SETTINGS + BENCH_SETTINGS + KINDS[kind]
           + [f"t_max={rec['t_max']}", f"seed={rec['seed']}", f"label={label(rec['id'], kind)}",
              f"local_results_path={jdir.relative_to(ROOT)}"])
    start_t = 0
    if resume_from:
        # EPyMARL's checkpoint folders have spaces and colons in their names: copy the step to a plain path
        src_dir, start_t = resume_from
        shutil.rmtree(jdir / "resume_from", ignore_errors=True)
        shutil.copytree(src_dir, jdir / "resume_from" / str(start_t))
        cmd += [f"checkpoint_path={(jdir / 'resume_from').relative_to(ROOT)}", f"load_step={start_t}"]
    commit, src, dirty = git_state()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=open(attempt_log(rec["id"], kind, n), "w"),
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    attempts.append({"n": n, "start_t": start_t, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "pid": proc.pid, "commit": commit, "src": src, "dirty": dirty,
                     "command": "python " + " ".join(cmd[1:])})
    save_run(rec)  # record each attempt as soon as it starts, so no job can go unrecorded
    return proc.pid, start_t


# ----------------------------------------------------------------------------------
# run / list / status / stop / resume / delete
# ----------------------------------------------------------------------------------
def cmd_run(a):
    run_id = a.id
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        sys.exit("an ID may contain only letters, digits, - and _")
    if run_dir(run_id).exists():
        sys.exit(f"run {run_id} already exists  (resume it, or delete it first)")
    commit, src, dirty = git_state()
    if dirty:
        print("WARNING: src/ has uncommitted changes; this run will not be reproducible from a commit\n")
    run_dir(run_id).mkdir(parents=True)
    rec = {"id": run_id, "seed": a.seed, "t_max": a.t_max, "note": a.note, "task": ENV_KEY,
           "created": time.strftime("%Y-%m-%d %H:%M"), "created_epoch": time.time(),
           "commit": commit, "src": src, "dirty": dirty, "attempts": {kind: [] for kind in KINDS}}
    save_run(rec)
    for kind in KINDS:
        pid, _ = launch(rec, kind)
        print(f"start run {run_id}  {kind:<6} pid {pid}")
    print(f"\nrun {run_id}: seed {a.seed}, {a.t_max:,} steps, "
          f"commit {commit}{' (uncommitted changes in src/)' if dirty else ''}")
    print(f"follow it with:  run_long.py status {run_id}")


def job_cell(status, t, t_max, was_resumed):
    """A job's state in one short cell, e.g. 'running 37%', 'done', 'stopped 45% r'."""
    pct = "" if status in ("done", "not started") else f" {min(100, t * 100 // t_max)}%"
    return status + pct + (" r" if was_resumed else "")


def cmd_list(a):
    recs = all_runs()
    if not recs:
        print("no runs yet")
    else:
        alive = live_jobs()
        print(f"{len(recs)} run{'s' if len(recs) != 1 else ''}: {', '.join(r['id'] for r in recs)}\n")
        w = max(8, max(len(r["id"]) for r in recs) + 2)
        print(f"   {'id':<{w}}{'created':<18}{'seed':>5}{'t_max':>12}   {'qmix':<16}{'blame':<16}{'commit':<10}note")
        for rec in recs:
            qmix, blame = (job_cell(*job_state(rec, kind, alive)[:2], rec["t_max"], resumed(rec, kind)) for kind in KINDS)
            print(f"   {rec['id']:<{w}}{rec['created']:<18}{rec['seed']:>5}{rec['t_max']:>12,}   "
                  f"{qmix:<16}{blame:<16}{rec['commit'] + ('*' if rec['dirty'] else ''):<10}{rec['note']}")
        print("\n   r = resumed from a checkpoint;  commit* = started with uncommitted changes in src/")
    summaries = sorted(p.parent.name for p in SUMMARIES_DIR.glob("*/summary.txt"))
    if summaries:
        print(f"\nsummaries of several runs, in {SUMMARIES_DIR.relative_to(ROOT)}/:  {', '.join(summaries)}")


def status_lines(rec, alive):
    lines = [f"run {rec['id']}  seed {rec['seed']}  {time.strftime('%H:%M:%S')}  (t_max {rec['t_max']:,})"
             f"{'  ' + rec['note'] if rec['note'] else ''}"]
    all_done = True
    for kind in KINDS:
        status, t, n = job_state(rec, kind, alive)
        pct = min(100, t * 100 // rec["t_max"])
        all_done &= status == "done"
        lines.append(f"   {kind:<6}[{'#' * (pct // 2):<50}] {pct:3d}%  {t:>9,}  {status}{f'  (attempt {n})' if n > 1 else ''}")
    return lines, all_done


def cmd_status(a):
    """Progress bars, redrawn in place every 5 s until every job is done (Ctrl+C stops the display only)."""
    ids = unique(a.ids)

    def draw():
        alive, lines, all_done = live_jobs(), [], True
        for run_id in ids:
            run_lines, done = status_lines(load_run(run_id), alive)
            lines += run_lines
            all_done &= done
        return lines, all_done

    lines, all_done = draw()  # also stops here with a message if an ID does not exist
    if all_done or not sys.stdout.isatty():  # nothing to watch, or not a terminal: print once
        print("\n".join(lines))
        return
    drawn = 0
    print("\033[?25l", end="")                        # hide the cursor while redrawing
    try:
        while True:
            if drawn:
                print(f"\033[{drawn}F", end="")       # back to the first line drawn last time
            for line in lines:
                print("\033[2K" + line)               # clear the old line, then write the new one
            drawn = len(lines)
            if all_done:
                print(f"\nall jobs finished; next:  run_long.py summary {' '.join(ids)}")
                return
            time.sleep(5)
            lines, all_done = draw()
    except KeyboardInterrupt:
        print("\n(display stopped; the jobs keep going)")
    finally:
        print("\033[?25h", end="", flush=True)        # show the cursor again


def cmd_stop(a):
    recs = [load_run(run_id) for run_id in unique(a.ids)]
    alive = live_jobs()
    for rec in recs:
        stopped = 0
        for kind in KINDS:
            pid = alive.get(label(rec["id"], kind))
            if pid:
                os.killpg(pid, signal.SIGTERM)  # the job is its session's leader: stop it and anything it started
                print(f"stopped run {rec['id']}  {kind:<6} pid {pid}")
                stopped += 1
        print(f"run {rec['id']}: {stopped} job(s) stopped"
              + (f"; continue later with:  run_long.py resume {rec['id']}" if stopped else ""))


def cmd_resume(a):
    recs = [load_run(run_id) for run_id in unique(a.ids)]
    alive = live_jobs()
    commit, src, dirty = git_state()
    plans = []
    for rec in recs:
        todo = [(kind, *job_state(rec, kind, alive)[:2]) for kind in KINDS]
        todo = [(kind, status, t) for kind, status, t in todo if status not in ("running", "done")]
        if not todo:
            print(f"run {rec['id']}: nothing to resume, both jobs are running or done")
        elif (src, dirty) != (rec["src"], rec["dirty"]) and not a.force:
            sys.exit(f"the training code (src/) changed since run {rec['id']} started (commit {rec['commit']}"
                     f"{' with uncommitted changes' if rec['dirty'] else ''}; now {commit}"
                     f"{' with uncommitted changes' if dirty else ''}).\nResuming would mix code versions "
                     f"inside one run; add --force if that is really intended. Nothing was resumed.")
        else:
            plans.append((rec, todo))
    for rec, todo in plans:  # launch only after every run passed the check
        for kind, status, t in todo:
            ckpt = None if a.from_scratch else latest_checkpoint(rec["id"], kind)
            pid, start_t = launch(rec, kind, resume_from=ckpt)
            how = f"from checkpoint at {start_t:,}" if ckpt else "from step 0"
            print(f"resume run {rec['id']}  {kind:<6} pid {pid}  ({status} at {t:,}; continuing {how})")


def sacred_dirs_of(run_id):
    """Sacred result folders whose config carries this run's label (every attempt of both jobs)."""
    found = []
    for d in SACRED_DIR.glob("[0-9]*"):
        cfg = read_json(d / "config.json", tries=1)
        if cfg and str(cfg.get("label", "")).startswith(f"longrun.{run_id}."):
            found.append(d)
    return found


def cmd_delete(a):
    rec = load_run(a.id)
    alive = live_jobs()
    if any(label(rec["id"], kind) in alive for kind in KINDS):
        sys.exit(f"run {rec['id']} still has running jobs; stop them first:  run_long.py stop {rec['id']}")
    sacred = sacred_dirs_of(rec["id"])
    n_att = sum(len(v) for v in rec["attempts"].values())
    print(f"run {rec['id']}: seed {rec['seed']}, {n_att} attempt(s), {len(sacred)} Sacred result(s), "
          f"folder {run_dir(rec['id']).relative_to(ROOT)}")
    if not a.yes and input(f"type {rec['id']} to delete all of it: ").strip() != rec["id"]:
        sys.exit("not deleted")
    for d in sacred:
        shutil.rmtree(d)
    shutil.rmtree(run_dir(rec["id"]))
    print(f"deleted run {rec['id']}")


# ----------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------
def job_metrics(rec, kind):
    """The job's metrics, stitched over its counted attempts: attempt k contributes its values from its
    own start step up to (not including) the step the next attempt resumed from."""
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


def late_mean(m, key, t_from):
    if key not in m or not m[key]["values"]:
        return float("nan")
    vals = [v for t, v in zip(m[key]["steps"], m[key]["values"]) if t >= t_from]
    vals = vals or m[key]["values"][-1:]  # short run, nothing logged in the window: use the last value
    return sum(vals) / len(vals)


def share_of_the_way(acc, base):
    """(acc - base) / (1 - base): how far the head gets from the lazy strategy (0) to perfect (1)."""
    return (acc - base) / (1 - base) if base < 1 else float("nan")


def fmt(x, spec):
    """A number in the given format, e.g. '<7.3f', or a '-' of the same width when it is missing (nan)."""
    if x == x:
        return format(x, spec)
    return format("-", ("<" if spec.startswith("<") else ">") + re.search(r"\d+", spec).group())


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


BLAME_KEYS = ["blame_acc", "blame_vote_acc", "blame_base",
              "blame_who_acc", "blame_who_base", "blame_whether_acc", "blame_whether_base"]


# the questions the blame head is scored on: (name in a run's summary, column in a summary of several runs,
# accuracy key, base key)
BLAME_ROWS = [
    ("choice, each agent", "each", "blame_acc", "blame_base"),
    ("choice, team vote", "vote", "blame_vote_acc", "blame_base"),
    ("who", "who", "blame_who_acc", "blame_who_base"),
    ("whether", "whether", "blame_whether_acc", "blame_whether_base"),
]
BLAME_COLOUR = {"each": C_ORANGE, "vote": C_AQUA, "who": C_YELLOW, "whether": C_MAGENTA}
LEGEND = [
    "   return   = test return (1.0 = all food collected); the paper's QMIX on this task: average 0.67, maximum 0.90",
    "   choice   = pick the right one of: agent 1, 2, 3 or nobody",
    "              each agent (each) = every agent's own pick;  team vote (vote) = the agents' majority vote",
    "   who      = only at steps where someone is to blame: is the most likely agent the blamed one",
    "   whether  = say \"someone\" or \"nobody\" right",
    "   base     = always give the most common answer (the lazy strategy)",
    "   gap      = acc - base: above 0 = beats lazy",
    "   share    = gap / (1 - base): 0% = lazy, 100% = perfect",
]
MARKS = (("*", "* = still running: not final"),
         ("r", "r = resumed from a checkpoint (close to, not bit-identical with, an uninterrupted job)"),
         ("!", "! = stopped or crashed before the end"))


def marks_lines(marks):
    return ["   " + ";  ".join(text for sym, text in MARKS if sym in marks)] if marks else []


def run_result(rec, alive):
    """Both jobs of a run: reached, mark and return of each; the blame head's metrics; the stitched metrics."""
    res = {"rec": rec, "marks": set(), "metrics": {}}
    for kind in KINDS:
        status, t, _ = job_state(rec, kind, alive)
        m = job_metrics(rec, kind)
        mark = (("*" if status == "running" else "") + ("r" if resumed(rec, kind) else "")
                + ("!" if status in ("crashed", "stopped") else ""))
        res["marks"] |= set(mark)
        res["metrics"][kind] = m
        res[kind] = {"t": t, "mark": mark, "return": late_mean(m, "test_return_mean", t // 2)}
    m, t = res["metrics"]["blame"], res["blame"]["t"]
    res["blame_metrics"] = {k: late_mean(m, k, t // 2) for k in BLAME_KEYS} if m else None
    return res


def save_lines(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return path


def summary_one(rec):
    r = run_result(rec, live_jobs())
    out = [f"run {rec['id']}  seed {rec['seed']}  t_max {rec['t_max']:,}  created {rec['created']}  "
           f"commit {rec['commit']}{'*' if rec['dirty'] else ''}{'  -- ' + rec['note'] if rec['note'] else ''}",
           f"written {time.strftime('%Y-%m-%d %H:%M')}; averages over the second half of what each job has reached",
           "",
           f"   {'job':<8}{'reached':<12}return"]
    out += [f"   {kind:<8}{r[kind]['t']:<12,}{fmt(r[kind]['return'], '<8.3f')}{r[kind]['mark']}".rstrip() for kind in KINDS]
    out += ["", f"   {'blame head':<21}{'acc':<8}{'base':<8}{'gap':<9}share"]
    b = r["blame_metrics"]
    for name, _, acc_key, base_key in BLAME_ROWS if b else []:
        acc, base = b[acc_key], b[base_key]
        out.append(f"   {name:<21}{fmt(acc, '<8.3f')}{fmt(base, '<8.3f')}{fmt(acc - base, '<+9.3f')}"
                   f"{fmt(share_of_the_way(acc, base), '<+5.0%')}".rstrip())
    out += ([] if b else ["   (no results saved yet)"]) + [""] + LEGEND + marks_lines(r["marks"])
    print("\n".join(out))

    folder = run_dir(rec["id"])
    saved = [save_lines(folder / "summary.txt", out)]
    for kind, m in r["metrics"].items():
        if m:
            job_dir(rec["id"], kind).mkdir(exist_ok=True)
            saved.append(job_dir(rec["id"], kind) / "metrics.json")
            saved[-1].write_text(json.dumps(m))
    if any(r["metrics"].values()):
        saved.append(plot({(rec["id"], kind): m for kind, m in r["metrics"].items()}, folder / "summary.png",
                          f"run {rec['id']} (seed {rec['seed']}, t_max {rec['t_max']:,})"
                          f"{'  -- ' + rec['note'] if rec['note'] else ''}"))
    print(f"\nsaved in {folder.relative_to(ROOT)}/:  {', '.join(str(p.relative_to(folder)) for p in saved)}")


def summary_many(recs, name):
    alive = live_jobs()
    results = [run_result(rec, alive) for rec in recs]
    w = max(8, max(len(rec["id"]) for rec in recs) + 4)  # room for the ID and its marks
    out = [f"summary {name}: {len(recs)} runs ({', '.join(rec['id'] for rec in recs)})",
           f"written {time.strftime('%Y-%m-%d %H:%M')}; per run, averages over the second half of what each job has reached",
           "",
           f"   {'run':<{w}}{'seed':<6}{'t_max':<12}{'qmix':<8}{'blame':<10}blame head: gap = acc - base",
           f"   {'':<{w}}{'':<6}{'':<12}{'return':<8}{'return':<10}" + "".join(f"{col:<9}" for _, col, _, _ in BLAME_ROWS).rstrip()]

    def gaps(b):
        if not b:
            return "(no blame results yet)"
        return "".join(fmt(b[acc_key] - b[base_key], "<+9.3f") for _, _, acc_key, base_key in BLAME_ROWS).rstrip()

    for r in results:
        rec = r["rec"]
        marks = "".join(sym for sym, _ in MARKS if sym in r["marks"])
        out.append(f"   {(rec['id'] + ' ' + marks).rstrip():<{w}}{rec['seed']:<6}{rec['t_max']:<12,}"
                   f"{fmt(r['qmix']['return'], '<8.3f')}{fmt(r['blame']['return'], '<10.3f')}" + gaps(r["blame_metrics"]))
    with_blame = [r["blame_metrics"] for r in results if r["blame_metrics"]]
    b = {k: mean([x[k] for x in with_blame]) for k in BLAME_KEYS} if with_blame else None
    out.append(f"   {'MEAN':<{w}}{f'{len(results)} runs':<18}{fmt(mean([r['qmix']['return'] for r in results]), '<8.3f')}"
               f"{fmt(mean([r['blame']['return'] for r in results]), '<10.3f')}" + (gaps(b) if b else ""))

    notes = []
    if len({rec["t_max"] for rec in recs}) > 1:
        notes.append("the runs have different step counts (t_max): the MEAN mixes different amounts of training")
    if len({(rec["src"], rec["dirty"]) for rec in recs}) > 1:
        notes.append("the runs come from different versions of the training code (src/): the MEAN mixes code versions")
    out += [""] + [f"   NOTE: {note}" for note in notes] + ([""] if notes else [])
    out += LEGEND + marks_lines(set().union(*(r["marks"] for r in results)))
    print("\n".join(out))

    folder = SUMMARIES_DIR / name
    existed = folder.exists()
    folder.mkdir(parents=True, exist_ok=True)
    saved = [save_lines(folder / "summary.txt", out)]
    curves = {(r["rec"]["id"], kind): m for r in results for kind, m in r["metrics"].items()}
    if any(curves.values()):
        saved.append(plot(curves, folder / "summary.png", f"{name}: {len(recs)} runs, one line per run"))
    print(f"\n{'updated' if existed else 'saved in'} {folder.relative_to(ROOT)}/:  {', '.join(p.name for p in saved)}")


def cmd_summary(a):
    ids = unique(a.ids)
    if len(ids) == 1:
        if a.name:
            sys.exit("--name is for a summary of several runs; a single run's summary goes in its own folder")
        summary_one(load_run(ids[0]))
        return
    name = a.name or "+".join(ids)
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", name):
        sys.exit("a summary name may contain only letters, digits, -, _ and +")
    summary_many([load_run(run_id) for run_id in ids], name)


def plot(curves, path, title):
    """Left: test return of every job. Right: each blame job's accuracy minus base, for every question in
    BLAME_ROWS. curves: {(run ID, job): metrics}."""
    plt.rcParams.update({"axes.edgecolor": C_GRID, "axes.labelcolor": C_INK, "xtick.color": C_INK,
                         "ytick.color": C_INK, "axes.titlecolor": C_INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)  # both panels: the same steps on the x axis
    labelled = set()

    def legend_name(name):
        if name in labelled:
            return None
        labelled.add(name)
        return name

    for (_, kind), m in curves.items():
        colour, style, name = LINE[kind]
        if "test_return_mean" in m:
            r = m["test_return_mean"]
            ax1.plot(r["steps"], r["values"], color=colour, linestyle=style, linewidth=2, alpha=0.85,
                     marker="o", markersize=2.5, label=legend_name(name))
        if kind != "blame":
            continue
        for name, col, acc_key, base_key in BLAME_ROWS:
            if acc_key not in m or base_key not in m:
                continue
            base_at = dict(zip(m[base_key]["steps"], m[base_key]["values"]))
            pts = [(t, v - base_at[t]) for t, v in zip(m[acc_key]["steps"], m[acc_key]["values"]) if t in base_at]
            ax2.plot([p[0] for p in pts], [p[1] for p in pts], color=BLAME_COLOUR[col], marker="o", markersize=2.5,
                     linewidth=1.5, alpha=0.8, label=legend_name(f"{name}: accuracy − base"))
    ax2.axhline(0.0, color=C_REF, linestyle="--", linewidth=1.5, label="base (lazy strategy)")
    ax1.set_title("task: test_return_mean")
    ax2.set_title("blame head: accuracy − base (above 0 = beats lazy)")
    ax1.set_xlim(left=0)  # shared: applies to both panels
    for ax in (ax1, ax2):
        ax.set_xlabel("t_env (environment steps)")
        ax.grid(color=C_GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{title}: {ENV_KEY.split(':')[1]}, benchmark settings", color=C_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("run", help="start a new run: benchmark QMIX and blame QMIX, with one seed")
    p.add_argument("id", help="a name for the run (letters, digits, - and _)")
    p.add_argument("--seed", type=int, required=True, help="the run's seed, e.g. --seed 1")
    p.add_argument("--t-max", type=int, default=2_000_000)
    p.add_argument("--note", default="", help="a short description, shown by list and summary")
    sub.add_parser("list", help="every run with the state of its two jobs, and the summaries written so far")
    for name, text in (("status", "progress bars of runs, updated in place until every job is done"),
                       ("stop", "stop the runs' jobs")):
        sub.add_parser(name, help=text).add_argument("ids", nargs="+", metavar="id")
    p = sub.add_parser("summary", help="one run's results (saved in its folder), or several runs side by side "
                                       "with their MEAN (saved in a new folder in summaries/)")
    p.add_argument("ids", nargs="+", metavar="id")
    p.add_argument("--name", help="folder name for a summary of several runs (default: the IDs joined by +)")
    p = sub.add_parser("resume", help="continue the runs' unfinished jobs from their latest checkpoint")
    p.add_argument("ids", nargs="+", metavar="id")
    p.add_argument("--from-scratch", action="store_true", help="start unfinished jobs again from step 0")
    p.add_argument("--force", action="store_true", help="resume even though src/ changed since the run started")
    p = sub.add_parser("delete", help="remove a run: its folder and its Sacred results")
    p.add_argument("id")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    a = parser.parse_args()
    {"run": cmd_run, "list": cmd_list, "status": cmd_status, "summary": cmd_summary,
     "stop": cmd_stop, "resume": cmd_resume, "delete": cmd_delete}[a.action](a)


if __name__ == "__main__":
    main()
