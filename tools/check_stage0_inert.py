"""
Inert-head check -- the blame head, with nothing training it, never moves.
=========================================================================

THE HEAD
    Each agent network's output layer (fc2) is widened from
        n_actions            (the Q head)
    to
        n_actions + n_agents + 1  (the Q head followed by the B head: one output
                                   per agent, plus one for "nobody to blame").
    In the forward pass the output is split: only the Q half is fed to the
    mixer. The B half is computed alongside it.

WHICH RUN IS INERT
    blame_head=True  split_backward=False
        Plain QMIX backward: loss.backward() reaches the network only through
        the Q values the mixer used, so autograd gives the B columns exactly
        zero gradient. This is the inert-head control.

    blame_head=True  split_backward=True
        NOT inert, on purpose: Stage 1 trains the B rows (and the body) with
        the blame loss. This check correctly FAILS on such a run.

WHAT THIS CHECK SHOWS
    If the B block of the gradient is genuinely zero, then:
        - the B rows of fc2 receive zero gradient, and
        - Adam's update with a zero gradient is exactly zero, and
        - gradient clipping scales zero to zero,
    so those rows can never move from wherever initialisation put them.
    Comparing two checkpoints asks whether the values MOVED, which does not
    depend on where they started.

    The stronger evidence is now bit-identity: with blame_head_rng_match=True
    an inert-head run logs exactly the same metrics as plain QMIX. This check
    is the direct picture behind it: the B weights did not change by one bit.

PASS CONDITION
    Q rows (0 .. n_actions-1)          max|change| > 0            (still learning)
    B rows (n_actions .. end)          max|change| == 0.000e+00   (inert)

HOW TO PRODUCE THE CHECKPOINTS
    python src/main.py --config=qmix --env-config=gymma with \
        env_args.time_limit=50 env_args.key="lbforaging:Foraging-2s-10x10-3p-3f-v3" \
        t_max=100000 use_cuda=False seed=1 \
        blame_head=True split_backward=False \
        save_model=True save_model_interval=25000 \
        use_rnn=True lr=0.0003 epsilon_anneal_time=200000 \
        target_update_interval_or_tau=0.01 evaluation_epsilon=0.05
    (the last two lines: the benchmark settings the project uses; the check works without them too)

USAGE
    python tools/check_stage0_inert.py                    # most recent run
    python tools/check_stage0_inert.py --run "<run directory name>"   # quote it: names contain spaces
    python tools/check_stage0_inert.py --n-actions 5                   # a task with another number of actions
    The width of the B head is read from the checkpoint (all rows after the n_actions Q rows).
"""

import argparse
import pathlib
import sys

import torch as th


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models-dir",
        default="results/models",
        help="directory EPyMARL writes checkpoints to",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="a specific run directory; default is the most recently modified",
    )
    # Where the Q block ends and the B block begins. LBF has 6 actions (NONE/N/S/W/E/LOAD);
    # every row after them belongs to the B head, whatever its width.
    parser.add_argument("--n-actions", type=int, default=6)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Find the run to inspect.
    #    EPyMARL writes to results/models/<unique_token>/<t_env>/agent.th
    # ------------------------------------------------------------------
    models_dir = pathlib.Path(args.models_dir)
    if not models_dir.is_dir():
        sys.exit(f"no such directory: {models_dir}  (did the run use save_model=True?)")

    if args.run:
        run_dir = models_dir / args.run
    else:
        runs = [p for p in models_dir.iterdir() if p.is_dir()]
        if not runs:
            sys.exit(f"{models_dir} contains no runs  (did the run use save_model=True?)")
        run_dir = max(runs, key=lambda p: p.stat().st_mtime)

    # ------------------------------------------------------------------
    # 2. Collect the checkpoints. Sub-directories are named by t_env, so
    #    sorting them numerically puts them in training order.
    # ------------------------------------------------------------------
    ckpts = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if len(ckpts) < 2:
        sys.exit(
            f"need at least 2 checkpoints to compare, found {len(ckpts)} in {run_dir}\n"
            f"lower save_model_interval, or raise t_max"
        )

    print(f"run         : {run_dir.name}")
    print(f"checkpoints : {[c.name for c in ckpts]}  (t_env values)")
    print(f"comparing   : {ckpts[0].name}  ->  {ckpts[-1].name}\n")

    # ------------------------------------------------------------------
    # 3. Load the first and last checkpoints. Everything between them is
    #    ordinary training, so any parameter that learned will differ.
    # ------------------------------------------------------------------
    first = th.load(ckpts[0] / "agent.th", map_location="cpu")
    last = th.load(ckpts[-1] / "agent.th", map_location="cpu")

    # ------------------------------------------------------------------
    # 4. Confirm the head actually widened. Without a blame head this layer
    #    has n_actions rows; with it, n_actions + n_agents + 1.
    # ------------------------------------------------------------------
    n_act = args.n_actions
    rows = first["fc2.weight"].shape[0]
    n_blame = rows - n_act
    print(f"fc2.weight  : {tuple(first['fc2.weight'].shape)}")
    if n_blame <= 0:
        sys.exit(
            f"\nFAIL: fc2 has {rows} rows, no more than the {n_act} Q rows: there is no blame head.\n"
            f"      Was the run made with blame_head=True?"
        )
    print(f"              rows 0..{n_act - 1}  = Q head ({n_act} actions)")
    print(f"              rows {n_act}..{rows - 1}  = B head ({n_blame} outputs: {n_blame - 1} agents + nobody)")
    print()

    # ------------------------------------------------------------------
    # 5. The measurement. For each parameter tensor, split the rows into
    #    the Q block and the B block, and report the largest change of any
    #    single number between the two checkpoints.
    # ------------------------------------------------------------------
    verdict_ok = True
    for key in ["fc2.weight", "fc2.bias"]:
        q_change = (first[key][:n_act] - last[key][:n_act]).abs().max().item()
        b_change = (first[key][n_act:] - last[key][n_act:]).abs().max().item()

        q_ok = q_change > 0.0     # the Q half must be learning
        b_ok = b_change == 0.0    # the B half must not have moved at all
        verdict_ok &= q_ok and b_ok

        print(f"{key}")
        print(
            f"   Q rows  max|change| = {q_change:.3e}   "
            f"{'OK  (learning as normal)' if q_ok else 'FAIL (did not learn)'}"
        )
        print(
            f"   B rows  max|change| = {b_change:.3e}   "
            f"{'OK  (inert)' if b_ok else 'FAIL (gradient leaked into the blame head)'}"
        )
        print()

    # ------------------------------------------------------------------
    # 6. Show a few raw numbers side by side, so the result is visible
    #    rather than only asserted.
    # ------------------------------------------------------------------
    print("first 4 values of B row 0, in each checkpoint:")
    print(f"   at t_env={ckpts[0].name:>7}: "
          f"{[f'{v:+.6f}' for v in first['fc2.weight'][n_act][:4].tolist()]}")
    print(f"   at t_env={ckpts[-1].name:>7}: "
          f"{[f'{v:+.6f}' for v in last['fc2.weight'][n_act][:4].tolist()]}")
    print()
    print("first 4 values of Q row 0, for contrast:")
    print(f"   at t_env={ckpts[0].name:>7}: "
          f"{[f'{v:+.6f}' for v in first['fc2.weight'][0][:4].tolist()]}")
    print(f"   at t_env={ckpts[-1].name:>7}: "
          f"{[f'{v:+.6f}' for v in last['fc2.weight'][0][:4].tolist()]}")
    print()

    # ------------------------------------------------------------------
    # 7. Verdict.
    # ------------------------------------------------------------------
    print("=" * 72)
    if verdict_ok:
        print("PASS -- the Q head learned; the blame head did not move at all.")
        print("        The head is added without affecting training.")
    else:
        print("FAIL -- see the lines marked FAIL above.")
        print("        A non-zero change in the B rows means gradient is reaching")
        print("        the blame head. Expected if the run used split_backward=True")
        print("        (Stage 1 trains the head); the inert control must use")
        print("        blame_head=True split_backward=False.")
    print("=" * 72)
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
