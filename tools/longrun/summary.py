"""summary: measures computed from the logged confusion counts, the table and the plot."""
import json

import matplotlib

matplotlib.use("Agg")  # draw into a file; WSL usually has no screen
import matplotlib.pyplot as plt
import numpy as np

from .config import KINDS, ROOT, SUMMARIES_DIR
from .jobs import job_metrics, job_state, live_jobs, load_run, run_dir


GREY = "#8a8984"


LINE = {"qmix": (GREY, "--", "benchmark QMIX"), "blame": ("#2a78d6", "-", "blame QMIX")}


QUESTIONS = [("blame", "#e87ba4"), ("importance", "#eda100"), ("combined", "#1baf7a")]


WHO = ["each", "vote"]


P_BINS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # as the learner logs them


LEGEND = [
    "   pooled over the second half of each job's steps",
    "   return     = test return (1.0 = all food collected)",
    "   blame      = did the team do worse than predicted (1) or not (0): delta > 0, or above the run's blame_threshold / blame_margin,",
    "                summed over blame_window steps if the run has one;  importance = which agent has the largest mixer weight",
    "   combined   = the final answer: the most important agent if blame, else nobody",
    "   each       = every agent's own answer;  vote = the team's majority vote, or with ensemble_loss the team's mean-logit prediction",
    "                both graded on the held-out test episodes (epsilon 0.05), which the agents never train on;",
    "                (train) = a run from before held-out grading, graded on its training batches",
    "   acc/maj    = accuracy / the majority baseline (always give the most common answer)",
    "   sens/spec  = sensitivity / specificity (blame: answer 1 is positive; importance, combined: macro-averaged over the answers)",
    "   blame ranked = auc: chance that a blame step gets a higher blame probability than a not-blame step (0.5 = nothing learned);",
    "                sens/spec@rate: sensitivity / specificity when the head flags as many steps as there are blame steps",
    "   train - held-out = how much better the training batches look: importance accuracy, blame auc (near 0 unless the heads overfit)",
    "   *  still running (not final)    !  stopped or crashed",
]


def mean(xs):
    xs = [x for x in xs if x == x]  # skip missing values (nan)
    return sum(xs) / len(xs) if xs else float("nan")


def late_mean(m, key, t_from):
    """Mean of the key's values from step t_from on (the last value if nothing was logged there yet)."""
    s = m.get(key, {"steps": [], "values": []})
    return mean([v for t, v in zip(s["steps"], s["values"]) if t >= t_from] or s["values"][-1:])


def counts(m, q, who, where=""):
    """[(step, k x k confusion matrix)] of a question at every log point; rows = truth, columns = answer."""
    keys = [k for k in m if k.startswith(f"cm_{where}{q}_{who}_")]
    k = max((int(key.split("_")[-1]) + 1 for key in keys), default=0)
    cell = lambda i, j, n: m[f"cm_{where}{q}_{who}_{i}_{j}"]["values"][n]
    return [(t, [[cell(i, j, n) for j in range(k)] for i in range(k)]) for n, t in enumerate(m[keys[0]]["steps"])] if keys else []


def measures(c, binary):
    """(accuracy, majority baseline, sensitivity, specificity) of one confusion matrix.
    binary: plain sens/spec with answer 1 as the positive; otherwise macro-averaged over the answers."""
    k, total = len(c), sum(map(sum, c))
    if not total:
        return (float("nan"),) * 4
    rows = [sum(r) for r in c]
    cols = [sum(c[i][j] for i in range(k)) for j in range(k)]
    answers = [1] if binary else range(k)
    sens = mean(c[i][i] / rows[i] for i in answers if rows[i])
    spec = mean((total - rows[i] - cols[i] + c[i][i]) / (total - rows[i]) for i in answers if total - rows[i])
    return sum(c[i][i] for i in range(k)) / total, max(rows) / total, sens, spec


def hist(m, who, where=""):
    """[(step, P, N)] of the blame probability histogram at every log point: P[j] / N[j] = blame / not-blame steps
    whose probability fell in bin j of P_BINS."""
    nb = len(P_BINS) - 1
    keys = [f"ph_{where}blame_{who}_{i}_{j}" for i in range(2) for j in range(nb)]
    if any(k not in m for k in keys):
        return []
    return [(t, [m[f"ph_{where}blame_{who}_1_{j}"]["values"][n] for j in range(nb)],
                [m[f"ph_{where}blame_{who}_0_{j}"]["values"][n] for j in range(nb)]) for n, t in enumerate(m[keys[0]]["steps"])]


def ranked(P, N):
    """(auc, sensitivity, specificity) from the histogram of the blame probability of the blame steps (P) and the
    not-blame steps (N).  auc = the chance that a blame step gets a higher probability than a not-blame step, ties
    counting half.  sensitivity / specificity are taken where the head flags as many steps as there are blame steps
    (the most probable ones first, a bin split pro rata), i.e. at the label's own rate rather than at 0.5."""
    tp, tn = sum(P), sum(N)
    if not tp or not tn:
        return (float("nan"),) * 3
    auc = sum(P[j] * (sum(N[:j]) + 0.5 * N[j]) for j in range(len(P))) / (tp * tn)
    budget, hit, false = tp, 0.0, 0.0
    for j in reversed(range(len(P))):
        size = P[j] + N[j]
        if size:
            f = min(1.0, budget / size)
            hit, false, budget = hit + f * P[j], false + f * N[j], budget - f * size
        if budget <= 0:
            break
    return auc, hit / tp, 1 - false / tn


def run_result(rec, alive):
    """Per job: steps reached, mark, metrics, late test return.  The blame job's heads are graded on the held-out
    test episodes when the run has them ("set" = "held-out"), else on its training batches ("training").
    Per (question, who): the measures pooled over the second half of the job's steps ("scores") and (steps, accuracy,
    majority) at every log point ("curves").  Per who: the ranked blame measures ("ranked") and how much better the
    training batches look than the held-out episodes ("gap": importance accuracy, blame auc)."""
    res = {"rec": rec, "scores": {}, "curves": {}, "ranked": {}, "gap": {}}
    for kind in KINDS:
        status, t = job_state(rec, kind, alive)
        m = job_metrics(rec, kind)
        mark = "*" if status == "running" else "!" if status in ("stopped", "crashed") else ""
        res[kind] = {"t": t, "mark": mark, "metrics": m, "return": late_mean(m, "test_return_mean", t // 2)}
    res["marks"] = "".join(s for s in "*!" if s in res["qmix"]["mark"] + res["blame"]["mark"])
    m, t_half = res["blame"]["metrics"], res["blame"]["t"] // 2

    def scored(q, who, where):
        """(measures pooled over the second half, (steps, accuracy, majority) curve) of a question on one set."""
        cms = counts(m, q, who, where)
        late = [c for t, c in cms if t >= t_half] or [c for _, c in cms[-1:]]
        pooled = [[sum(c[i][j] for c in late) for j in range(len(c))] for i, c in enumerate(late[0])] if late else [[0]]
        return measures(pooled, q == "blame"), list(zip(*[(t, *measures(c, q == "blame")[:2]) for t, c in cms])) or ([], [], [])

    def rank(who, where):
        """The ranked blame measures (auc, sensitivity, specificity) pooled over the second half, on one set."""
        hs = hist(m, who, where)
        late = [(P, N) for t, P, N in hs if t >= t_half] or [(P, N) for _, P, N in hs[-1:]]
        if not late:
            return (float("nan"),) * 3
        return ranked(*[[sum(h[i][j] for h in late) for j in range(len(P_BINS) - 1)] for i in range(2)])

    where = "held_out_" if counts(m, "blame", "vote", "held_out_") else ""   # metric prefix of the set that is shown
    res["set"] = "held-out" if where else "training"
    for who in WHO:
        for q, _ in QUESTIONS:
            res["scores"][q, who], res["curves"][q, who] = scored(q, who, where)
        res["ranked"][who] = rank(who, where)
        res["gap"][who] = (float("nan"),) * 2
        if where:
            res["gap"][who] = (scored("importance", who, "")[0][0] - res["scores"]["importance", who][0],
                               rank(who, "")[0] - res["ranked"][who][0])
    return res


def summary(recs, folder, title):
    """Print and save one table (per run: each agent's answers and the team vote, graded on the held-out test
    episodes; plus their MEAN if there are several runs) and the plot."""
    results = [run_result(rec, live_jobs()) for rec in recs]
    w = max(len(rec["id"]) for rec in recs) + 4  # room for the ID and its marks

    def rows(first, seed, t_max, qmix, blame, r, note=""):
        lines = []
        for who in WHO:
            left = f"{first:<{w}}{seed:<6}{t_max:<12}{qmix:<8.3f}{blame:<9.3f}" if not lines else " " * (w + 35)
            cells = "".join(f"{'{:.3f}/{:.3f} {:.3f}/{:.3f}'.format(*r['scores'][q, who]):<24}" for q, _ in QUESTIONS)
            auc, sens, spec = r["ranked"][who]
            cells += f"{f'{auc:.3f} {sens:.3f}/{spec:.3f}' if auc == auc else '':<22}"   # blank without the histogram
            cells += "  ".join(f"{x:+.3f}" for x in r["gap"][who] if x == x)               # blank without held-out data
            lines.append(f"   {left}{who + note:<15}{cells}".rstrip())
        return lines

    def averaged(key, n):
        return {k: tuple(mean(r[key][k][i] for r in results) for i in range(n)) for k in results[0][key]}

    out = [title, "",
           f"   {'run':<{w}}{'seed':<6}{'t_max':<12}{'qmix':<8}{'blame':<9}{'':<15}" + "".join(f"{q:<24}" for q, _ in QUESTIONS)
           + f"{'blame ranked':<22}train - held-out",
           f"   {'':<{w}}{'':<6}{'':<12}{'return':<8}{'return':<9}{'':<15}" + "acc/maj     sens/spec   " * len(QUESTIONS)
           + f"{'auc   sens/spec@rate':<22}imp acc  blame auc"]
    for r in results:
        rec = r["rec"]
        out += rows(f"{rec['id']} {r['marks']}", rec["seed"], f"{rec['t_max']:,}", r["qmix"]["return"], r["blame"]["return"], r,
                    "" if r["set"] == "held-out" else " (train)")
    if len(results) > 1:
        out += rows("MEAN", "", "", mean(r["qmix"]["return"] for r in results), mean(r["blame"]["return"] for r in results),
                    {"scores": averaged("scores", 4), "ranked": averaged("ranked", 3), "gap": averaged("gap", 2)})
    out += [""] + LEGEND
    print("\n".join(out))

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.txt").write_text("\n".join(out) + "\n")
    plot(results, folder / "summary.png", title)
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
    else:
        name = "+".join(rec["id"] for rec in recs)
        summary(recs, SUMMARIES_DIR / name, f"summary {name}")


def band(ax, series, colour, style="-", label=None, width=1.5):
    """Draw the mean of several (steps, values) series as a line and their min-max range as a shaded band.
    The series are put on a common grid first (linear interpolation; a run contributes only where it has data)."""
    series = [(s, v) for s, v in series if s]
    if not series:
        return
    grid = np.linspace(0, max(s[-1] for s, _ in series), 200)
    rows = np.array([np.interp(grid, s, v, left=np.nan, right=np.nan) for s, v in series])
    ok = ~np.isnan(rows).all(0)   # grid points where at least one run has data
    grid, rows = grid[ok], rows[:, ok]
    ax.plot(grid, np.nanmean(rows, 0), color=colour, linestyle=style, linewidth=width, label=label)
    if len(series) > 1:
        ax.fill_between(grid, np.nanmin(rows, 0), np.nanmax(rows, 0), color=colour, alpha=0.15, linewidth=0)


def plot(results, path, title):
    """Left: test return of both jobs. Right: a panel per question (columns) and who (rows), with the accuracy
    (colour) and the majority baseline (grey dashed) on the held-out test episodes. Lines are the mean over the
    runs, bands their min-max range."""
    layout = [["return"] + [f"{q}_{who}" for q, _ in QUESTIONS] for who in WHO]
    fig, ax = plt.subplot_mosaic(layout, figsize=(17, 6.5), width_ratios=[2] + [1] * len(QUESTIONS), sharex=True)
    for kind in KINDS:
        colour, style, name = LINE[kind]
        series = [(r[kind]["metrics"].get("test_return_mean", {}).get("steps", []),
                   r[kind]["metrics"].get("test_return_mean", {}).get("values", [])) for r in results]
        band(ax["return"], series, colour, style, name)
    for q, colour in QUESTIONS:
        for who in WHO:
            a = ax[f"{q}_{who}"]
            band(a, [r["curves"][q, who][:2] for r in results], colour, label="accuracy")
            band(a, [r["curves"][q, who][::2] for r in results], GREY, "--", "majority baseline", 1)
            a.set_title(f"{q}: {who}")
    ax["return"].set_title("test return")
    ax["return"].set_xlim(left=0)
    for name, a in ax.items():
        a.set_ylim(0, 1)
        a.grid(alpha=0.3)
        a.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:g}M")   # steps in millions
        if name == "return" or name.endswith(WHO[-1]):
            a.set_xlabel("environment steps")
        if name in ("return", f"{QUESTIONS[0][0]}_{WHO[0]}"):
            a.legend(fontsize=8, frameon=False)
    sets = " / ".join(sorted({r["set"] for r in results}))
    fig.suptitle(f"{title}  ({len(results)} run{'s' if len(results) != 1 else ''}: line = mean, band = min to max; "
                 f"heads graded on {sets} episodes)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
