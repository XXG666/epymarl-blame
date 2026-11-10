"""Window labels from the saved per-step records: for every window of n consecutive real steps of an episode,
S = sum of delta (the team's shortfall over the window) and B_a = sum of dL/dQ_a (each agent's accumulated
blame).  Reports the distribution of S, how concentrated the blame is, and how the window 'who' relates to the
one-step 'who'.  Usage: window_labels.py <n> <npz> [<npz> ...]"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # tools/
from delta_scan import corrected  # noqa: E402

n = int(sys.argv[1])
rows = []
for name, step, d, delta, mean, var in corrected(sys.argv[2:]):
    # the stored dL/dQ_a was computed with the stored delta; keep the mixer weight and re-apply the corrected delta
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(d["delta"][:, None] != 0, d["w"] / d["delta"][:, None], 0.0)
    g = delta[:, None] * weight                                  # dL/dQ_a per step and agent, (steps, 3)
    ep, t = d["ep"], d["t"]
    S, B, last_delta, last_g = [], [], [], []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]                               # the episode's real steps, in order
        for k in range(n - 1, len(idx)):
            w = idx[k - n + 1 : k + 1]
            S.append(delta[w].sum())
            B.append(g[w].sum(0))
            last_delta.append(delta[idx[k]])
            last_g.append(g[idx[k]])
    S, B, last_delta, last_g = map(np.array, (S, B, last_delta, last_g))
    med = np.median(S)
    scale = 1.4826 * np.median(np.abs(S - med))
    pos = B.clip(min=0)
    conc = pos.max(1) / np.where(pos.sum(1) > 0, pos.sum(1), np.nan)   # top agent's share of the window's blame
    who_win = B.argmax(1)
    who_step = last_g.argmax(1)
    sd1 = np.std(delta)
    rows.append((name, step, len(S), med, scale, S.std(), sd1 * np.sqrt(n), (S > 0).mean(),
                 [(S > tau).mean() for tau in (1.0, 2.0, 3.0, 5.0)], np.nanmean(conc), (who_win == who_step).mean(),
                 (who_win[S > 2.0] == who_step[S > 2.0]).mean() if (S > 2.0).any() else np.nan,
                 np.nanmean(conc[S > 2.0]) if (S > 2.0).any() else np.nan))
print(f"window n = {n} steps.  S = sum of delta over the window;  scale = robust std of S;  sqrt(n)*sd1 = what the std of S would be "
      f"if the n deltas were independent (noise only)")
print(f"{'run':<12}{'step':>9} {'windows':>8} {'median S':>9} {'scale':>7} {'std S':>7} {'sqrt(n)sd1':>11} {'S>0':>6} "
      f"{'S>1':>6} {'S>2':>6} {'S>3':>6} {'S>5':>6} | {'top share':>9} {'who=step':>9} {'who=step|S>2':>13} {'top share|S>2':>14}")
for name, step, m, med, scale, sd, sdn, p0, shares, conc, agree, agree2, conc2 in rows:
    print(f"{name:<12}{step:>9,} {m:>8} {med:>9.2f} {scale:>7.2f} {sd:>7.2f} {sdn:>11.2f} {p0:>6.3f} "
          + " ".join(f"{x:>6.3f}" for x in shares) + f" | {conc:>9.2f} {agree:>9.2f} {agree2:>13.2f} {conc2:>14.2f}")
print("top share = the most-blamed agent's share of the window's positive blame (1/3 = spread evenly, 1 = all on one agent);"
      "  who=step = the window's most-blamed agent equals the one-step 'who' at the window's last step")
