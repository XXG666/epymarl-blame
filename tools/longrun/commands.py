"""The commands that manage runs: run, list, status, stop, resume, delete."""
import json
import os
import re
import shutil
import signal
import sys
import time

from .config import KINDS, RUNS_DIR, SACRED_DIR
from .jobs import (git_state, job_state, label, latest_checkpoint, launch, live_jobs, load_run, read_json,
                   run_dir, save_run)


def cmd_run(a):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.id):
        sys.exit("an ID may contain only letters, digits, - and _")
    if run_dir(a.id).exists():
        sys.exit(f"run {a.id} already exists")
    commit, src, dirty = git_state()
    if dirty:
        print("WARNING: src/ has uncommitted changes")
    run_dir(a.id).mkdir(parents=True)
    rec = {"id": a.id, "seed": a.seed, "t_max": a.t_max, "note": a.note, "created": time.strftime("%Y-%m-%d %H:%M"),
           "commit": commit, "src": src, "dirty": dirty, "attempts": {kind: [] for kind in KINDS}}
    save_run(rec)
    for kind in KINDS:
        launch(rec, kind)
    print(f"started run {a.id}: seed {a.seed}, {a.t_max:,} steps")


def cmd_list(a):
    recs = sorted((json.loads(p.read_text()) for p in RUNS_DIR.glob("*/run.json")), key=lambda r: (r["created"], r["id"]))
    if not recs:
        print("no runs yet")
        return
    alive = live_jobs()
    w = max(len(r["id"]) for r in recs) + 3
    print(f"{'id':<{w}}{'created':<18}{'seed':<6}{'t_max':<12}{'qmix':<16}{'blame':<16}{'commit':<10}note")
    for rec in recs:
        cells = []
        for kind in KINDS:
            status, t = job_state(rec, kind, alive)
            pct = f" {min(100, t * 100 // rec['t_max'])}%" if status in ("running", "stopped", "crashed") else ""
            cells.append(status + pct)
        commit = rec["commit"] + ("*" if rec["dirty"] else "")
        print(f"{rec['id']:<{w}}{rec['created']:<18}{rec['seed']:<6}{rec['t_max']:<12,}"
              f"{cells[0]:<16}{cells[1]:<16}{commit:<10}{rec['note']}")
    print("\ncommit* = src/ had uncommitted changes")


def status_lines(rec, alive):
    lines = [f"run {rec['id']}  seed {rec['seed']}  {time.strftime('%H:%M:%S')}"]
    all_done = True
    for kind in KINDS:
        status, t = job_state(rec, kind, alive)
        pct = min(100, t * 100 // rec["t_max"])
        all_done &= status == "done"
        lines.append(f"   {kind:<6}[{'#' * (pct // 2):<50}] {pct:3d}%  {t:>9,}  {status}")
    return lines, all_done


def cmd_status(a):
    ids = list(dict.fromkeys(a.ids))

    def draw():
        alive, lines, all_done = live_jobs(), [], True
        for run_id in ids:
            run_lines, done = status_lines(load_run(run_id), alive)
            lines += run_lines
            all_done &= done
        return lines, all_done

    lines, all_done = draw()
    print("\n".join(lines))
    if not sys.stdout.isatty():
        return
    try:
        while not all_done:
            time.sleep(5)
            lines, all_done = draw()
            # move the cursor back up to the first line, then overwrite every line
            print(f"\033[{len(lines)}F" + "".join(f"\033[2K{line}\n" for line in lines), end="", flush=True)
    except KeyboardInterrupt:
        print()


def cmd_stop(a):
    alive = live_jobs()
    for rec in [load_run(run_id) for run_id in dict.fromkeys(a.ids)]:
        pids = [alive[label(rec["id"], kind)] for kind in KINDS if label(rec["id"], kind) in alive]
        for pid in pids:
            os.killpg(pid, signal.SIGTERM)  # each job leads its own session: stop it and anything it started
        print(f"run {rec['id']}: stopped {len(pids)} job(s)")


def cmd_resume(a):
    alive = live_jobs()
    recs = [load_run(run_id) for run_id in dict.fromkeys(a.ids)]
    todo = [(rec, kind) for rec in recs for kind in KINDS if job_state(rec, kind, alive)[0] not in ("running", "done")]
    if not todo:
        print("nothing to resume: every job is running or done")
        return
    _, src, dirty = git_state()
    changed = sorted({rec["id"] for rec, _ in todo if (rec["src"], rec["dirty"]) != (src, dirty)})
    if changed and not a.force:
        sys.exit(f"src/ changed since run {', '.join(changed)} started; nothing resumed (--force to resume anyway)")
    for rec, kind in todo:
        start_t = launch(rec, kind, resume_from=None if a.from_scratch else latest_checkpoint(rec["id"], kind))
        print(f"run {rec['id']} {kind}: resumed from step {start_t:,}")


def cmd_delete(a):
    load_run(a.id)  # stops here if there is no such run
    alive = live_jobs()
    if any(label(a.id, kind) in alive for kind in KINDS):
        sys.exit(f"run {a.id} is still running; stop it first")
    if not a.yes and input(f"type {a.id} to delete it: ").strip() != a.id:
        sys.exit("not deleted")
    for d in SACRED_DIR.glob("[0-9]*"):
        cfg = read_json(d / "config.json", tries=1)
        if cfg and str(cfg.get("label", "")).startswith(f"longrun.{a.id}."):
            shutil.rmtree(d)
    shutil.rmtree(run_dir(a.id))
    print(f"deleted run {a.id}")
