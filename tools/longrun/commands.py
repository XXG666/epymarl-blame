"""The commands that manage runs: run, list, status, stop, resume, delete.

Every cmd_... function takes args, the command line as argparse read it (see tools/run_long.py).
rec is a run's record, the dictionary stored in the run's run.json (see jobs.py).
"""
import json
import os
import re
import shutil
import signal
import sys
import time

from .config import KINDS, RUNS_DIR, SACRED_DIR
from .jobs import (commit_text, git_state, job_state, label, latest_checkpoint, launch, live_jobs, load_run,
                   load_runs, read_json, run_dir, save_run, unique_ids)


# ---------------------------------------------------------------------------------------------------------
# A helper that the list and the status share
# ---------------------------------------------------------------------------------------------------------

def percent_done(steps, t_max):
    """In: the steps a job reached and the steps it should reach. Out: a whole number from 0 to 100."""
    return min(100, steps * 100 // t_max)


# ---------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------

def cmd_run(args):
    """Start a new run: write its record and launch its two jobs. In: args.id, args.seed, args.t_max and
    args.note."""
    run_id = args.id
    seed = args.seed
    t_max = args.t_max
    if re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        sys.exit("an ID may contain only letters, digits, - and _")
    if run_dir(run_id).exists():
        sys.exit(f"run {run_id} already exists")
    commit, src_hash, dirty = git_state()
    if dirty:
        print("WARNING: src/ has uncommitted changes")
    run_dir(run_id).mkdir(parents=True)

    attempts = {}
    for kind in KINDS:
        attempts[kind] = []
    rec = {
        "id": run_id,
        "seed": seed,
        "t_max": t_max,
        "note": args.note,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "commit": commit,
        "src": src_hash,
        "dirty": dirty,
        "attempts": attempts,
    }
    save_run(rec)
    for kind in KINDS:
        launch(rec, kind)
    print(f"started run {run_id}: seed {seed}, {t_max:,} steps")


# ---------------------------------------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------------------------------------

def created_then_id(rec):
    """Sorting key of a run: by the minute it was created, and runs of the same minute by ID."""
    return rec["created"], rec["id"]


def load_all_runs():
    """Out: the record of every run there is, oldest first."""
    recs = []
    for path in RUNS_DIR.glob("*/run.json"):
        recs.append(json.loads(path.read_text()))
    recs.sort(key=created_then_id)
    return recs


def job_cell(rec, kind, alive):
    """In: a run's record, the job's kind and the result of live_jobs(). Out: the text of this job in the
    list: its status, followed by the percentage reached if the job is not at its start or its end."""
    status, steps = job_state(rec, kind, alive)
    if status in ("running", "stopped", "crashed"):
        percent = percent_done(steps, rec["t_max"])
        return f"{status} {percent}%"
    return status


def cmd_list(args):
    """Print a table with one row per run. args is not used."""
    recs = load_all_runs()
    if len(recs) == 0:
        print("no runs yet")
        return
    alive = live_jobs()

    # the ID column is as wide as the longest ID, plus three spaces
    id_width = 0
    for rec in recs:
        id_width = max(id_width, len(rec["id"]))
    id_width = id_width + 3

    header = "id".ljust(id_width) + "created".ljust(18) + "seed".ljust(6) + "t_max".ljust(12)
    for kind in KINDS:
        header = header + kind.ljust(16)
    header = header + "commit".ljust(10) + "note"
    print(header)

    for rec in recs:
        seed = str(rec["seed"])
        t_max = rec["t_max"]
        t_max_text = f"{t_max:,}"
        row = rec["id"].ljust(id_width) + rec["created"].ljust(18) + seed.ljust(6) + t_max_text.ljust(12)
        for kind in KINDS:
            row = row + job_cell(rec, kind, alive).ljust(16)
        note_text = str(rec["note"])
        row = row + commit_text(rec).ljust(10) + note_text
        print(row)
    print("\ncommit* = src/ had uncommitted changes")


# ---------------------------------------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------------------------------------

def status_lines(rec, alive):
    """In: a run's record and the result of live_jobs(). Out: (lines, all_done): a heading and one progress
    bar per job, as a list of strings, and whether both jobs are done."""
    run_id = rec["id"]
    seed = rec["seed"]
    clock = time.strftime("%H:%M:%S")
    lines = [f"run {run_id}  seed {seed}  {clock}"]
    all_done = True
    for kind in KINDS:
        status, steps = job_state(rec, kind, alive)
        if status != "done":
            all_done = False
        percent = percent_done(steps, rec["t_max"])
        # the bar is 50 characters wide: one # for every 2 percent
        filled = "#" * (percent // 2)
        bar = filled.ljust(50)
        kind_text = kind.ljust(6)
        # a line looks like this:
        #    qmix  [##########                                        ]  20%    400,000  running
        lines.append(f"   {kind_text}[{bar}] {percent:3d}%  {steps:>9,}  {status}")
    return lines, all_done


def status_of_runs(ids):
    """In: a list of run IDs. Out: (lines, all_done): the status lines of all these runs, one run after the
    other, and whether every job of every run is done."""
    alive = live_jobs()
    lines = []
    all_done = True
    for run_id in ids:
        run_lines, run_done = status_lines(load_run(run_id), alive)
        lines.extend(run_lines)
        if not run_done:
            all_done = False
    return lines, all_done


def redraw(lines):
    """In: a list of lines, as many as are on the screen from the last time. Prints them over the old ones."""
    line_count = len(lines)
    text = f"\033[{line_count}F"  # move the cursor back up to the first line
    for line in lines:
        text = text + "\033[2K" + line + "\n"  # erase the old line, write the new one
    print(text, end="", flush=True)


def cmd_status(args):
    """Print the progress of the runs args.ids. In a terminal, keep updating it until every job is done."""
    ids = unique_ids(args.ids)
    lines, all_done = status_of_runs(ids)
    print("\n".join(lines))
    if not sys.stdout.isatty():
        return
    try:
        while not all_done:
            time.sleep(5)
            lines, all_done = status_of_runs(ids)
            redraw(lines)
    except KeyboardInterrupt:
        print()


# ---------------------------------------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------------------------------------

def cmd_stop(args):
    """Stop every running job of the runs args.ids."""
    alive = live_jobs()
    recs = load_runs(unique_ids(args.ids))
    for rec in recs:
        run_id = rec["id"]
        stopped = 0
        for kind in KINDS:
            job_label = label(run_id, kind)
            if job_label in alive:
                # each job leads its own session: stop it and anything it started
                os.killpg(alive[job_label], signal.SIGTERM)
                stopped = stopped + 1
        print(f"run {run_id}: stopped {stopped} job(s)")


# ---------------------------------------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------------------------------------

def unfinished_jobs(recs, alive):
    """In: a list of run records and the result of live_jobs(). Out: a list of (rec, kind), one entry for
    every job that is neither running nor done."""
    todo = []
    for rec in recs:
        for kind in KINDS:
            status, steps = job_state(rec, kind, alive)
            if status != "running" and status != "done":
                todo.append((rec, kind))
    return todo


def cmd_resume(args):
    """Start the unfinished jobs of the runs args.ids again: from their latest checkpoint, or from step 0
    with args.from_scratch. Refuses if src/ is not what the run started with, unless args.force is set."""
    alive = live_jobs()
    recs = load_runs(unique_ids(args.ids))
    todo = unfinished_jobs(recs, alive)
    if len(todo) == 0:
        print("nothing to resume: every job is running or done")
        return

    # a job that continues with different code is not the same experiment any more
    commit, src_hash, dirty = git_state()
    changed = []
    for rec, kind in todo:
        if rec["src"] != src_hash or rec["dirty"] != dirty:
            if rec["id"] not in changed:
                changed.append(rec["id"])
    changed.sort()
    if len(changed) > 0 and not args.force:
        changed_ids = ", ".join(changed)
        sys.exit(f"src/ changed since run {changed_ids} started; nothing resumed (--force to resume anyway)")

    for rec, kind in todo:
        run_id = rec["id"]
        if args.from_scratch:
            checkpoint = None
        else:
            checkpoint = latest_checkpoint(run_id, kind)
        start_t = launch(rec, kind, resume_from=checkpoint)
        print(f"run {run_id} {kind}: resumed from step {start_t:,}")


# ---------------------------------------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------------------------------------

def cmd_delete(args):
    """Remove the folder of the run args.id and its Sacred results. Asks first, unless args.yes is set."""
    run_id = args.id
    load_run(run_id)  # stops here if there is no such run
    alive = live_jobs()
    for kind in KINDS:
        if label(run_id, kind) in alive:
            sys.exit(f"run {run_id} is still running; stop it first")
    if not args.yes:
        typed = input(f"type {run_id} to delete it: ")
        if typed.strip() != run_id:
            sys.exit("not deleted")

    # the run's Sacred folders are those whose saved configuration carries one of the run's labels
    for sacred_dir in SACRED_DIR.glob("[0-9]*"):
        config = read_json(sacred_dir / "config.json", tries=1)
        if not config:
            continue  # no readable file, or an empty one
        sacred_label = str(config.get("label", ""))
        if sacred_label.startswith(f"longrun.{run_id}."):
            shutil.rmtree(sacred_dir)
    shutil.rmtree(run_dir(run_id))
    print(f"deleted run {run_id}")
