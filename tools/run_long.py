r"""
Long runs: benchmark QMIX against blame-head QMIX on Foraging-2s-10x10-3p-3f-v3 (CPU).

A run has an ID and one seed, and trains two jobs with that seed:
    qmix    benchmark QMIX                         blame_head=False split_backward=False
    blame   benchmark QMIX + trained blame head    blame_head=True  split_backward=True
Benchmark settings: Papoudakis et al. 2021, Table 28 (QMIX, LBF).

COMMANDS
    run <id> --seed S [--t-max N] [--note TEXT]    start a run (it keeps going after the terminal closes)
    list                                           every run and the state of its two jobs
    status <id> ...                                progress bars, updated in place (Ctrl+C stops the display only)
    summary <id>                                   one run's results  -> results/long_runs/runs/<id>/
    summary <id> <id> ...                          several runs and their MEAN -> results/long_runs/summaries/<id>+<id>+.../
    stop <id> ...                                  stop the runs' jobs
    resume <id> ... [--from-scratch] [--force]     continue unfinished jobs from their latest checkpoint
    delete <id> [--yes]                            remove the run's folder and its Sacred results

FILES in results/long_runs/runs/<id>/
    run.json                          seed, steps, commit, attempts
    <job>.<n>.log                     output of the job's attempt n
    <job>/models/                     checkpoints every 100k steps, used by resume
    summary.txt, summary.png, <job>/metrics.json      written by summary

A resumed job continues from its latest checkpoint with an empty replay buffer: close to,
not identical with, a job that never stopped. resume --from-scratch starts at step 0 instead.
resume refuses if src/ changed since the run started (--force overrides).
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
    "use_cuda=False",
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

GREY = "#8a8984"
LINE = {"qmix": (GREY, "--", "benchmark QMIX"), "blame": ("#2a78d6", "-", "blame QMIX")}
# the blame head's questions: (name, accuracy key, base key, plot colour)
BLAME_ROWS = [
    ("each", "blame_acc", "blame_base", "#eb6834"),
    ("vote", "blame_vote_acc", "blame_base", "#1baf7a"),
    ("who", "blame_who_acc", "blame_who_base", "#eda100"),
    ("whether", "blame_whether_acc", "blame_whether_base", "#e87ba4"),
]
LEGEND = [
    "   averages over the second half of each job's steps",
    "   return  = test return (1.0 = all food collected)",
    "   each    = each agent's own 4-way choice (agent 1, 2, 3 or nobody) right;  vote = the team's majority vote right",
    "   who     = at blame steps, the most likely agent is the blamed one;  whether = someone vs nobody right",
    "   acc/base = the blame head's accuracy / the accuracy of always giving the most common answer",
    "   *  still running (not final)    !  stopped or crashed",
]


# ----------------------------------------------------------------------------------
# Runs and jobs on disk
# ----------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------
# run / list / status / stop / resume / delete
# ----------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------
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


def late_mean(m, key, t_from):
    """Mean of the key's values from step t_from on (the last value if nothing was logged there yet)."""
    if not m.get(key, {}).get("values"):
        return float("nan")
    vals = [v for t, v in zip(m[key]["steps"], m[key]["values"]) if t >= t_from] or m[key]["values"][-1:]
    return sum(vals) / len(vals)


def mean(xs):
    xs = [x for x in xs if x == x]  # skip missing values (nan)
    return sum(xs) / len(xs) if xs else float("nan")


def run_result(rec, alive):
    """Per job: steps reached, mark, metrics and late test return; plus the blame head's late accuracies."""
    res = {"rec": rec}
    for kind in KINDS:
        status, t = job_state(rec, kind, alive)
        m = job_metrics(rec, kind)
        mark = "*" if status == "running" else "!" if status in ("stopped", "crashed") else ""
        res[kind] = {"t": t, "mark": mark, "metrics": m, "return": late_mean(m, "test_return_mean", t // 2)}
    res["marks"] = "".join(s for s in "*!" if s in res["qmix"]["mark"] + res["blame"]["mark"])
    blame = res["blame"]
    res["acc"] = {k: late_mean(blame["metrics"], k, blame["t"] // 2) for _, acc, base, _ in BLAME_ROWS for k in (acc, base)}
    return res


def summary(recs, folder, title):
    """Print and save one table (a row per run, plus their MEAN if there are several) and the plot."""
    alive = live_jobs()
    results = [run_result(rec, alive) for rec in recs]
    w = max(len(rec["id"]) for rec in recs) + 4  # room for the ID and its marks

    def row(first, seed, t_max, qmix, blame, acc):
        cells = "".join(f"{f'{acc[a]:.3f}/{acc[b]:.3f}':<13}" for _, a, b, _ in BLAME_ROWS)
        return f"   {first:<{w}}{seed:<6}{t_max:<12}{qmix:<8.3f}{blame:<9.3f}{cells}".rstrip()

    head1 = f"   {'run':<{w}}{'seed':<6}{'t_max':<12}{'qmix':<8}{'blame':<9}" + "".join(f"{q:<13}" for q, _, _, _ in BLAME_ROWS)
    head2 = f"   {'':<{w}}{'':<6}{'':<12}{'return':<8}{'return':<9}" + "acc/base     " * len(BLAME_ROWS)
    out = [title, "", head1.rstrip(), head2.rstrip()]
    for r in results:
        rec = r["rec"]
        out.append(row(f"{rec['id']} {r['marks']}", rec["seed"], f"{rec['t_max']:,}",
                       r["qmix"]["return"], r["blame"]["return"], r["acc"]))
    if len(results) > 1:
        out.append(row("MEAN", "", "", mean([r["qmix"]["return"] for r in results]),
                       mean([r["blame"]["return"] for r in results]),
                       {k: mean([r["acc"][k] for r in results]) for k in results[0]["acc"]}))
    out += [""] + LEGEND
    print("\n".join(out))

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.txt").write_text("\n".join(out) + "\n")
    plot({(r["rec"]["id"], kind): r[kind]["metrics"] for r in results for kind in KINDS}, folder / "summary.png", title)
    print(f"\nsaved in {folder.relative_to(ROOT)}/")
    return results


def cmd_summary(a):
    recs = [load_run(run_id) for run_id in dict.fromkeys(a.ids)]
    if len(recs) == 1:  # one run: saved in its own folder, with its jobs' metrics
        rec = recs[0]
        commit = rec["commit"] + ("*" if rec["dirty"] else "")
        results = summary(recs, run_dir(rec["id"]), f"run {rec['id']}  commit {commit}  {rec['note']}".rstrip())
        for kind in KINDS:
            (run_dir(rec["id"]) / kind).mkdir(exist_ok=True)
            (run_dir(rec["id"]) / kind / "metrics.json").write_text(json.dumps(results[0][kind]["metrics"]))
        return
    name = "+".join(rec["id"] for rec in recs)
    summary(recs, SUMMARIES_DIR / name, f"summary {name}")


def plot(curves, path, title):
    """Left: test return of every job. Right: one panel per blame-head question, with its accuracy (solid)
    and its base (dashed), all on a 0-1 scale. curves: {(run ID, job): metrics}."""
    fig, ax = plt.subplot_mosaic([["return", "each", "vote"], ["return", "who", "whether"]], figsize=(14, 6),
                                 width_ratios=[2, 1, 1], sharex=True)  # same steps on every x axis
    for (_, kind), m in curves.items():
        colour, style, name = LINE[kind]
        if "test_return_mean" in m:
            ax["return"].plot(m["test_return_mean"]["steps"], m["test_return_mean"]["values"],
                              color=colour, linestyle=style, linewidth=1.5, label=name)
        if kind != "blame":
            continue
        for q, acc, base, colour in BLAME_ROWS:
            for key, c, style, lab in ((acc, colour, "-", "accuracy"), (base, GREY, "--", "base")):
                if key in m:
                    ax[q].plot(m[key]["steps"], m[key]["values"], color=c, linestyle=style, linewidth=1, label=lab)
    ax["return"].set_title("test return")
    ax["return"].set_xlim(left=0)
    for q, _, _, _ in BLAME_ROWS:
        ax[q].set_title(f"blame head: {q}")
    for name, a in ax.items():
        a.set_ylim(0, 1)  # every panel on the same 0-1 scale
        a.grid(alpha=0.3)
        if name in ("return", "who", "whether"):
            a.set_xlabel("environment steps")
        handles, labels = a.get_legend_handles_labels()
        if handles and name in ("return", "each"):  # one legend for the task, one for all four questions
            unique = dict(zip(labels, handles))  # one legend entry per name, however many runs
            a.legend(unique.values(), unique.keys(), fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("run", help="start a run: benchmark QMIX and blame QMIX with one seed")
    p.add_argument("id")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--t-max", type=int, default=2_000_000)
    p.add_argument("--note", default="")
    sub.add_parser("list", help="every run and the state of its jobs")
    for name in ("status", "stop"):
        sub.add_parser(name).add_argument("ids", nargs="+", metavar="id")
    p = sub.add_parser("summary", help="one run's results, or several runs and their MEAN")
    p.add_argument("ids", nargs="+", metavar="id")
    p = sub.add_parser("resume", help="continue unfinished jobs from their latest checkpoint")
    p.add_argument("ids", nargs="+", metavar="id")
    p.add_argument("--from-scratch", action="store_true", help="start them again from step 0")
    p.add_argument("--force", action="store_true", help="resume even though src/ changed")
    p = sub.add_parser("delete", help="remove a run's folder and its Sacred results")
    p.add_argument("id")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    a = parser.parse_args()
    {"run": cmd_run, "list": cmd_list, "status": cmd_status, "summary": cmd_summary,
     "stop": cmd_stop, "resume": cmd_resume, "delete": cmd_delete}[a.action](a)


if __name__ == "__main__":
    main()
