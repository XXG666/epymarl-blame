r"""
Bit-identity experiment -- plain QMIX against the switched-off modifications.
============================================================================

WHAT THIS DOES
    1. Runs EPyMARL three times, one run after the other, with identical
       settings apart from two flags:
           baseline  blame_head=False  split_backward=False   plain QMIX
           m2        blame_head=False  split_backward=True    the break (M2) in place
           inert     blame_head=True   split_backward=False   the blame head, untrained
    2. Compares every logged metric of m2 and of inert with the baseline,
       value by value, and prints PASS or FAIL for each.
    3. Draws each candidate's curves on top of the baseline's, with their
       difference underneath, and saves the picture as a PNG.

WHAT PASS MEANS
    m2     interrupting the backward pass at the agent utilities and resuming
           it gives exactly the gradient plain QMIX computes.
    inert  widening fc2 for the blame head changes nothing while the head is
           not trained: blame_head_rng_match=True keeps the random numbers
           the same, and the separate Q multiply keeps the float rounding
           the same.
    The fourth combination, blame_head=True split_backward=True, is Stage 1:
    the blame loss trains the head and the body, so it is meant to differ
    and is not tested here.

WHY ONE SCRIPT RUNS ALL
    Bit-identity only means something if the commands differ in nothing but
    the flags under test. Every command is built from the same list,
    COMMON_SETTINGS below, so an accidental difference (a typo, a forgotten
    log interval) is impossible.

WHY THE RUNS ARE NOT IN PARALLEL
    Running them one after the other means each new run is simply the one
    Sacred directory that appeared while it ran.

READING THE PICTURE
    Left column   both runs' logged values. The baseline is a thick pale blue
                  line, the candidate a thin red dashed line drawn over it.
                  When the runs match, the red line sits exactly inside the blue.
    Right column  candidate minus baseline, at every timestep both runs logged.
                  PASS is a flat line at exactly 0 (title in green).
                  The axis stretches to fit the difference, so a difference of
                  one part in ten million can still look like a big spike --
                  read its size from the "max rel diff" column of the table.

USAGE
    python tools/run_stage0_rng_match.py                            # run all three, compare, plot
    python tools/run_stage0_rng_match.py --seed 2                   # the same with another seed
    python tools/run_stage0_rng_match.py --baseline 10 --candidate 11
                                                                    # no new runs: compare and
                                                                    # plot two existing runs
"""

import argparse
import json
import math
import pathlib
import shlex
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")  # draw into a file; WSL usually has no screen to open a window on
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent  # the epymarl directory
ENV_KEY = "lbforaging:Foraging-2s-10x10-3p-3f-v3"
RESULTS_DIR = ROOT / "results" / "sacred" / "qmix" / ENV_KEY  # where Sacred writes these runs
OUT_DIR = ROOT / "results" / "stage0_rng_match"  # where this script writes logs and plots

# Every setting all runs share. Every command is built from this one list.
COMMON_SETTINGS = [
    "env_args.time_limit=50",
    f"env_args.key={ENV_KEY}",
    "log_interval=5000",
    "runner_log_interval=5000",
    "learner_log_interval=5000",
    "test_interval=10000",
    "test_nepisode=10",
    "use_cuda=False",  # some GPU kernels are not deterministic even with a fixed seed
]

# The ONLY settings that differ between the runs.
ROLE_SETTINGS = {
    "baseline": ["blame_head=False", "split_backward=False"],
    "m2": ["blame_head=False", "split_backward=True"],
    "inert": ["blame_head=True", "blame_head_rng_match=True", "split_backward=False"],
}
CANDIDATES = ["m2", "inert"]

# The metrics drawn in the picture, one row each. The table compares all of them.
PLOT_METRICS = [
    "return_mean",
    "test_return_mean",
    "loss",
    "grad_norm",
    "td_error_abs",
    "q_taken_mean",
]


def existing_run_ids():
    """The numbered run directories Sacred has written so far: {'1', '2', ...}."""
    return {p.name for p in RESULTS_DIR.glob("[0-9]*") if p.is_dir()}


def run_epymarl(role, seed, t_max):
    """Run EPyMARL once for `role` and return the Sacred run id it created."""
    settings = COMMON_SETTINGS + [f"seed={seed}", f"t_max={t_max}"] + ROLE_SETTINGS[role]
    # sys.executable is the Python running this script -- the venv's, if it is activated.
    cmd = [sys.executable, "src/main.py", "--config=qmix", "--env-config=gymma", "with"] + settings
    log_path = OUT_DIR / f"{role}_seed{seed}.log"

    print(f"\n--- {role}: {' '.join(ROLE_SETTINGS[role])} ---")
    print("command : " + shlex.join(["python"] + cmd[1:]))
    print(f"log     : {log_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # Note which runs exist BEFORE, so the new one can be found afterwards.
    # ------------------------------------------------------------------
    before = existing_run_ids()
    start = time.time()
    with open(log_path, "w") as log:
        # EPyMARL prints a lot; send it to the log file instead of the screen.
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:  # poll() is None while the run is still going
            print(f"\rrunning : {time.time() - start:4.0f}s", end="", flush=True)
            time.sleep(1)
    print(f"\rfinished: {time.time() - start:4.0f}s")

    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-25:]
        print("\n".join(tail))
        sys.exit(f"\nthe {role} run crashed (exit code {proc.returncode}); last lines of its log are above")

    new = existing_run_ids() - before
    if len(new) != 1:
        sys.exit(f"expected exactly one new run in {RESULTS_DIR}, found {sorted(new)}")
    run_id = new.pop()
    print(f"run id  : {run_id}")
    return run_id


def load_metrics(run_id):
    """Sacred's metrics.json: {metric: {"steps": [...], "values": [...], ...}}."""
    path = RESULTS_DIR / run_id / "metrics.json"
    if not path.exists():
        sys.exit(f"no metrics for run {run_id}: {path} does not exist")
    return json.loads(path.read_text())


def same_value(a, b):
    """Exact equality, except that two NaNs count as equal."""
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def compare(baseline_id, candidate_id):
    """Print a table comparing every metric of the two runs. Returns True if all are identical."""
    base, cand = load_metrics(baseline_id), load_metrics(candidate_id)
    names = sorted(set(base) | set(cand))

    print(f"\nbaseline run {baseline_id}  vs  candidate run {candidate_id}")
    print(f"   {'metric':<22}{'result':<12}{'first differs at':>18}{'max rel diff':>15}")
    n_same = 0
    for name in names:
        if name not in base or name not in cand:
            print(f"   {name:<22}{'DIFFERS':<12}{'only in ' + ('candidate' if name in cand else 'baseline'):>18}")
            continue
        b, c = base[name], cand[name]
        if b["steps"] == c["steps"] and all(same_value(x, y) for x, y in zip(b["values"], c["values"])):
            n_same += 1
            print(f"   {name:<22}{'identical':<12}")
            continue

        # ------------------------------------------------------------------
        # Where does it first differ, and by how much at most (relative)?
        # ------------------------------------------------------------------
        first = next((sb for sb, sc, vb, vc in zip(b["steps"], c["steps"], b["values"], c["values"])
                      if sb != sc or not same_value(vb, vc)), None)
        # relative differences at the t_env values both runs logged; if there are none,
        # the runs differ only in WHEN they logged (their episodes ended at different steps)
        base_at = dict(zip(b["steps"], b["values"]))
        rel = [abs(v - base_at[s]) / max(abs(base_at[s]), 1e-12)
               for s, v in zip(c["steps"], c["values"]) if s in base_at and not same_value(v, base_at[s])]
        first_txt = f"t_env={first}" if first is not None else "logged steps"
        rel_txt = f"{max(rel):.3e}" if rel else "steps differ"
        print(f"   {name:<22}{'DIFFERS':<12}{first_txt:>18}{rel_txt:>15}")

    passed = n_same == len(names)
    print(f"   {n_same}/{len(names)} metrics bit-identical  ->  {'PASS' if passed else 'FAIL'}")
    return passed


def plot_runs(baseline_id, candidate_id, passed, label):
    """Overlay the two runs' curves and plot their difference. Returns the PNG path."""
    base, cand = load_metrics(baseline_id), load_metrics(candidate_id)
    names = [m for m in PLOT_METRICS if m in base and m in cand]

    fig, axes = plt.subplots(len(names), 2, figsize=(13, 2.7 * len(names)), squeeze=False)
    for (left, right), name in zip(axes, names):
        b, c = base[name], cand[name]

        # ------------------------------------------------------------------
        # Left: the raw curves, candidate drawn over baseline.
        # ------------------------------------------------------------------
        left.plot(b["steps"], b["values"], color="tab:blue", linewidth=5, alpha=0.35,
                  label=f"baseline  (run {baseline_id})")
        left.plot(c["steps"], c["values"], color="tab:red", linewidth=1.2, linestyle="--",
                  marker=".", label=f"{label} (run {candidate_id})")
        left.set_title(name)

        # ------------------------------------------------------------------
        # Right: candidate - baseline, at the timesteps both runs logged.
        # ------------------------------------------------------------------
        base_at = dict(zip(b["steps"], b["values"]))
        steps = [s for s in c["steps"] if s in base_at]
        diffs = [v - base_at[s] for s, v in zip(c["steps"], c["values"]) if s in base_at]
        right.axhline(0, color="grey", linewidth=0.8)
        right.plot(steps, diffs, color="black", marker=".")

        differing = [s for s, d in zip(steps, diffs) if d != 0]  # NaN != 0 counts as differing
        if not differing and len(steps) == len(b["steps"]) == len(c["steps"]):
            right.set_title(f"{name}: {label} - baseline = exactly 0", color="green")
        elif differing:
            right.axvline(differing[0], color="red", linestyle=":")
            right.set_title(f"{name}: first differs at t_env={differing[0]}", color="red")
        else:
            right.set_title(f"{name}: logged at different timesteps", color="red")

    axes[0][0].legend(loc="best", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t_env (environment steps)")

    verdict = "PASS - bit-identical" if passed else "FAIL - not bit-identical"
    fig.suptitle(f"baseline run {baseline_id} vs {label} run {candidate_id}  --  {verdict}",
                 fontsize=14, color="green" if passed else "red")
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # leave the top 3% for the big title
    path = OUT_DIR / f"run{baseline_id}_vs_run{candidate_id}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1, help="seed used by ALL runs")
    parser.add_argument("--t-max", type=int, default=50000, help="environment steps per run")
    parser.add_argument("--baseline", help="existing baseline run id (skips running)")
    parser.add_argument("--candidate", help="existing candidate run id (skips running)")
    args = parser.parse_args()
    if (args.baseline is None) != (args.candidate is None):
        parser.error("give both --baseline and --candidate, or neither")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. The runs: make them, or use the two given.
    # ------------------------------------------------------------------
    if args.baseline:
        baseline_id = args.baseline
        candidates = {"candidate": args.candidate}
        print(f"using existing runs: baseline {baseline_id}, candidate {args.candidate}")
    else:
        baseline_id = run_epymarl("baseline", args.seed, args.t_max)
        candidates = {role: run_epymarl(role, args.seed, args.t_max) for role in CANDIDATES}

    # ------------------------------------------------------------------
    # 2. The comparison and the picture, one candidate at a time.
    # ------------------------------------------------------------------
    print("\n" + "#" * 78 + "\n# COMPARISON\n" + "#" * 78)
    verdicts = {}
    for label, run_id in candidates.items():
        verdicts[label] = compare(baseline_id, run_id)
        path = plot_runs(baseline_id, run_id, verdicts[label], label)
        print(f"   plot saved: {path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # 3. Verdict.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    for label, ok in verdicts.items():
        print(f"   {'PASS' if ok else 'FAIL'}   {label} is bit-identical to plain QMIX")
    print("=" * 78)
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
