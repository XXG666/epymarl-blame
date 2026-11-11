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


QUESTIONS = [("whether", "#e87ba4"), ("who", "#eda100"), ("combined", "#1baf7a")]


OLD = {"whether": "blame", "who": "importance"}  # the two questions' names in runs logged before the renaming


BY = ["each", "vote"]  # who answered, as logged: every agent on its own, the team's majority vote


BASELINE = "central"  # the centralised head, the upper limit, as logged


NAIVE = "naive"  # the naive method, random guesses, as logged


EXTRA = [BASELINE, NAIVE]  # shown after the agents' answers, when a run has them


LABEL = {"each": "each", "vote": "maj", BASELINE: "baseline", NAIVE: "naive"}  # as shown


TITLE = {"whether": "whether", "who": "who", "combined": "blame"}  # the questions as shown; logged as in QUESTIONS


LEGEND = [
    "   pooled over the second half of each job's steps",
    "   return     = test return (1.0 = all food collected)",
    "   whether    = did the team do worse than predicted (1) or not (0);  who = which agent has the largest mixer weight",
    "   blame      = the final answer: the agent named by who if whether says yes, else nobody",
    "   each       = every agent's own answer;  maj = the team's majority vote",
    "   baseline   = the upper limit: a centralised head that reads the global state and all agents' hidden states",
    "   naive      = the naive method: yes or no to whether with probability 1/2, and for who each agent with equal probability",
    "                all graded on the held-out test episodes (epsilon 0.05), which nothing is trained on;",
    "                (train) = graded on the training batches instead, because a run in this summary has no held-out data",
    "   acc        = accuracy: the share of answers that are right",
    "   sens/spec  = sensitivity / specificity of the whether answer (answer 1, worse than predicted, is the positive)",
    "   *  still running (not final)    !  stopped or crashed",
]


def mean(xs):
    xs = [x for x in xs if x == x]  # skip missing values (nan)
    return sum(xs) / len(xs) if xs else float("nan")


def late_mean(m, key, t_from):
    """Mean of the key's values from step t_from on (the last value if nothing was logged there yet)."""
    s = m.get(key, {"steps": [], "values": []})
    return mean([v for t, v in zip(s["steps"], s["values"]) if t >= t_from] or s["values"][-1:])


def counts(m, q, by, where=""):
    """[(step, k x k confusion matrix)] of a question at every log point; rows = truth, columns = answer.
    where = "" for the training batches, "held_out_" for the test episodes."""
    keys = [k for k in m if k.startswith(f"cm_{where}{q}_{by}_")]
    if not keys and q in OLD:  # a run logged before the two questions were renamed
        q = OLD[q]
        keys = [k for k in m if k.startswith(f"cm_{where}{q}_{by}_")]
    k = max((int(key.split("_")[-1]) + 1 for key in keys), default=0)
    cell = lambda i, j, n: m[f"cm_{where}{q}_{by}_{i}_{j}"]["values"][n]
    return [(t, [[cell(i, j, n) for j in range(k)] for i in range(k)]) for n, t in enumerate(m[keys[0]]["steps"])] if keys else []


def measures(c):
    """(accuracy, sensitivity, specificity) of one confusion matrix; rows = truth, columns = answer.
    Sensitivity and specificity are for a yes/no question, with answer 1 as the positive; nan for the others."""
    nan, total = float("nan"), sum(map(sum, c))
    if not total:
        return nan, nan, nan
    acc = sum(c[i][i] for i in range(len(c))) / total
    if len(c) != 2:
        return acc, nan, nan
    return acc, c[1][1] / sum(c[1]) if sum(c[1]) else nan, c[0][0] / sum(c[0]) if sum(c[0]) else nan


def run_result(rec, alive, held_out=True):
    """Per job: steps reached, mark, metrics, late test return.  The blame job's heads are graded on the held-out
    test episodes when the run has them and held_out is set ("set" = "held-out"), else on its training batches
    ("training"); "set" is None while the job has logged no counts at all.
    Per (question, by): the measures pooled over the second half of the job's steps ("scores"), and (steps,
    accuracy) at every log point ("curves")."""
    res = {"rec": rec, "scores": {}, "curves": {}}
    for kind in KINDS:
        status, t = job_state(rec, kind, alive)
        m = job_metrics(rec, kind)
        mark = "*" if status == "running" else "!" if status in ("stopped", "crashed") else ""
        res[kind] = {"t": t, "mark": mark, "metrics": m, "return": late_mean(m, "test_return_mean", t // 2)}
    res["marks"] = "".join(s for s in "*!" if s in res["qmix"]["mark"] + res["blame"]["mark"])
    m, t_half = res["blame"]["metrics"], res["blame"]["t"] // 2

    def scored(q, by, where):
        """(measures pooled over the second half, (steps, accuracy) curve) of a question on one set."""
        cms = counts(m, q, by, where)
        late = [c for t, c in cms if t >= t_half] or [c for _, c in cms[-1:]]
        pooled = [[sum(c[i][j] for c in late) for j in range(len(c))] for i, c in enumerate(late[0])] if late else [[0]]
        return measures(pooled), list(zip(*[(t, measures(c)[0]) for t, c in cms])) or ([], [])

    where = "held_out_" if held_out and counts(m, "whether", "vote", "held_out_") else ""   # metric prefix of the set shown
    res["set"] = "held-out" if where else "training" if counts(m, "whether", "vote") else None   # None: nothing logged yet
    for by in BY + EXTRA:
        for q, _ in QUESTIONS:
            res["scores"][q, by], res["curves"][q, by] = scored(q, by, where)
    return res


def summary(recs, folder, title):
    """Print and save one table (per run: each agent's answers, the team's majority vote, the baseline and the naive method, graded on the held-out
    test episodes; plus their MEAN if there are several runs) and the plot.  If some of the runs have no held-out
    data, all of them are graded on their training batches instead, so that the MEAN and the plot compare like
    with like."""
    results = [run_result(rec, live_jobs()) for rec in recs]
    sets = {r["set"] for r in results} - {None}
    if len(sets) > 1:
        results = [run_result(rec, live_jobs(), held_out=False) for rec in recs]
    graded = "held-out" if sets == {"held-out"} else "training"
    note = "" if graded == "held-out" else " (train)"
    # the baseline and the naive method are shown in the MEAN and the plot only if every run with head data has them
    shown = [by for by in EXTRA if all(r["curves"]["whether", by][0] for r in results if r["set"])]
    w = max(len(rec["id"]) for rec in recs) + 4  # room for the ID and its marks

    def cell(q, score):
        """Accuracy for every question; sensitivity/specificity as well for whether, the yes/no question."""
        acc, sens, spec = score
        return f"{acc:<7.3f}{sens:.3f}/{spec:.3f}  " if q == "whether" else f"{acc:<12.3f}"

    def rows(first, seed, t_max, qmix, blame, scores):
        lines = []
        for i, by in enumerate(BY + EXTRA):
            if by in EXTRA and all(x != x for q, _ in QUESTIONS for x in scores[q, by]):
                continue  # a run from before the baseline or the naive method
            left = f"{first:<{w}}{seed:<6}{t_max:<12}{qmix:<8.3f}{blame:<9.3f}" if i == 0 else " " * (w + 35)
            cells = "".join(cell(q, scores[q, by]) for q, _ in QUESTIONS)
            lines.append(f"   {left}{LABEL[by] + note:<18}{cells}".rstrip())
        return lines

    widths = {q: len(cell(q, (0, 0, 0))) for q, _ in QUESTIONS}
    out = [title, "",
           f"   {'run':<{w}}{'seed':<6}{'t_max':<12}{'qmix':<8}{'blame':<9}{'':<18}"
           + "".join(f"{TITLE[q]:<{widths[q]}}" for q, _ in QUESTIONS).rstrip(),
           f"   {'':<{w}}{'':<6}{'':<12}{'return':<8}{'return':<9}{'':<18}"
           + "".join(f"{'acc    sens/spec' if q == 'whether' else 'acc':<{widths[q]}}" for q, _ in QUESTIONS).rstrip()]
    for r in results:
        rec = r["rec"]
        out += rows(f"{rec['id']} {r['marks']}", rec["seed"], f"{rec['t_max']:,}", r["qmix"]["return"], r["blame"]["return"],
                    r["scores"])
    if len(results) > 1:
        out += rows("MEAN", "", "", mean(r["qmix"]["return"] for r in results), mean(r["blame"]["return"] for r in results),
                    {key: tuple(mean(r["scores"][key][i] for r in results) if key[1] in BY + shown else float("nan")
                                for i in range(3)) for key in results[0]["scores"]})
    out += [""] + LEGEND
    print("\n".join(out))

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.txt").write_text("\n".join(out) + "\n")
    plot(results, folder / "summary.png", title, graded, shown)
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


def plot(results, path, title, graded, shown):
    """Left: test return of both jobs. Right: a panel per question (columns) and answerer (rows) with the accuracy on
    the held-out test episodes (colour), the baseline (grey dashed) and the naive method (grey dotted).
    Lines are the mean over the runs, bands their min-max range."""
    layout = [["return"] + [f"{q}_{by}" for q, _ in QUESTIONS] for by in BY]
    fig, ax = plt.subplot_mosaic(layout, figsize=(17, 6.5), width_ratios=[2] + [1] * len(QUESTIONS), sharex=True)
    for kind in KINDS:
        colour, style, name = LINE[kind]
        series = [(r[kind]["metrics"].get("test_return_mean", {}).get("steps", []),
                   r[kind]["metrics"].get("test_return_mean", {}).get("values", [])) for r in results]
        band(ax["return"], series, colour, style, name)
    for q, colour in QUESTIONS:
        for by in BY:
            a = ax[f"{q}_{by}"]
            band(a, [r["curves"][q, by] for r in results], colour, label="agents")
            for extra, style in ((BASELINE, "--"), (NAIVE, ":")):
                if extra in shown:
                    band(a, [r["curves"][q, extra] for r in results], GREY, style, LABEL[extra], 1.2)
            a.set_title(f"{TITLE[q]} accuracy: {LABEL[by]}")
    ax["return"].set_title("test return")
    ax["return"].set_xlim(left=0)
    for name, a in ax.items():
        a.set_ylim(0, 1)
        a.grid(alpha=0.3)
        a.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:g}M")   # steps in millions
        if name == "return" or name.endswith(BY[-1]):
            a.set_xlabel("environment steps")
        if name in ("return", f"{QUESTIONS[0][0]}_{BY[0]}"):
            a.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{title}  ({len(results)} run{'s' if len(results) != 1 else ''}: line = mean, band = min to max; "
                 f"heads graded on {graded} episodes)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
