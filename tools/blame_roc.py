"""How well the blame head ranks blame steps, read from the probability histograms the learner logs
(ph_<where>blame_<each|vote>_<truth>_<bin> in each blame job's Sacred metrics), pooled over the given runs.

    python tools/blame_roc.py <id> [<id> ...] [--who vote|each] [--set held_out|train]

Prints, per training window: the share of steps labelled blame, the AUC, and the share of blame steps caught at
several false-alarm rates; then, for the second half of training, where the head's probability sits for blame
and for not-blame steps.  Runs trained before the histograms were logged have nothing to show.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from longrun.jobs import job_metrics, load_run  # noqa: E402
from longrun.summary import P_BINS, hist, ranked  # noqa: E402

WINDOWS = [(0.25e6, 0.5e6), (0.5e6, 1e6), (1e6, 1.5e6), (1.5e6, 2e6), (2e6, 3e6), (3e6, 4e6), (2e6, 4e6)]
FALSE_ALARMS = [0.05, 0.10, 0.20, 0.30, 0.50]
NB = len(P_BINS) - 1


def pooled(metrics, who, where, lo, hi):
    """(P, N): the histograms of the blame / not-blame steps, summed over the runs and the log points in [lo, hi)."""
    P, N = [0.0] * NB, [0.0] * NB
    for m in metrics:
        for t, p, n in hist(m, who, where):
            if lo <= t < hi:
                P, N = [a + b for a, b in zip(P, p)], [a + b for a, b in zip(N, n)]
    return P, N


def caught(P, N, false_alarm):
    """Share of blame steps flagged when the most probable steps are flagged until the given share of not-blame
    steps is flagged too (a bin split pro rata)."""
    tp, tn, hit, false = sum(P), sum(N), 0.0, 0.0
    for j in reversed(range(NB)):
        if false + N[j] >= false_alarm * tn:
            f = (false_alarm * tn - false) / N[j] if N[j] else 1.0
            return (hit + f * P[j]) / tp
        hit, false = hit + P[j], false + N[j]
    return 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--who", default="vote", choices=["vote", "each"])
    parser.add_argument("--set", default="held_out", choices=["held_out", "train"])
    a = parser.parse_args()
    where = "held_out_" if a.set == "held_out" else ""
    metrics = [job_metrics(load_run(run_id), "blame") for run_id in a.ids]
    print(f"{'+'.join(a.ids)}: blame head, {a.who}, {a.set.replace('_', '-')} episodes\n")
    print(f"{'window':<11}{'blame rate':>11}{'auc':>8}   blame steps caught at a false-alarm rate of"
          + "".join(f"{int(f * 100):>5}%" for f in FALSE_ALARMS))
    for lo, hi in WINDOWS:
        P, N = pooled(metrics, a.who, where, lo, hi)
        if not sum(P) or not sum(N):
            continue
        print(f"{lo / 1e6:g}-{hi / 1e6:g}M".ljust(11) + f"{sum(P) / (sum(P) + sum(N)):>11.3f}{ranked(P, N)[0]:>8.3f}" + " " * 44
              + "".join(f"{caught(P, N, f):>6.2f}" for f in FALSE_ALARMS))
    P, N = pooled(metrics, a.who, where, 2e6, 4e6)
    if sum(P) and sum(N):
        print("\nwhere the head's probability sits, 2-4M (share of the steps of each kind per probability bin)")
        print(f"{'probability up to':<22}" + "".join(f"{e:>6}" for e in P_BINS[1:]))
        print(f"{'blame steps':<22}" + "".join(f"{x / sum(P):>6.2f}" for x in P))
        print(f"{'not-blame steps':<22}" + "".join(f"{x / sum(N):>6.2f}" for x in N))


if __name__ == "__main__":
    main()
