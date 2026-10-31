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


LEGEND = [
    "   pooled over the second half of each job's steps",
    "   return     = test return (1.0 = all food collected)",
    "   blame      = did the team do worse than predicted (1) or not (0);  importance = which agent has the largest mixer weight",
    "   combined   = the final answer: the most important agent if blame, else nobody",
    "   each       = every agent's own answer;  vote = the team's majority vote",
    "   acc/maj    = accuracy / the majority baseline (always give the most common answer)",
    "   sens/spec  = sensitivity / specificity (blame: answer 1 is positive; importance, combined: macro-averaged over the answers)",
    "   *  still running (not final)    !  stopped or crashed",
]


def mean(xs):
    xs = [x for x in xs if x == x]  # skip missing values (nan)
    return sum(xs) / len(xs) if xs else float("nan")


def late_mean(m, key, t_from):
    """Mean of the key's values from step t_from on (the last value if nothing was logged there yet)."""
    s = m.get(key, {"steps": [], "values": []})
    return mean([v for t, v in zip(s["steps"], s["values"]) if t >= t_from] or s["values"][-1:])


def counts(m, q, who):
    """[(step, k x k confusion matrix)] of a question at every log point; rows = truth, columns = answer."""
    keys = [k for k in m if k.startswith(f"cm_{q}_{who}_")]
    k = max((int(key.split("_")[-1]) + 1 for key in keys), default=0)
    cell = lambda i, j, n: m[f"cm_{q}_{who}_{i}_{j}"]["values"][n]
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


def run_result(rec, alive):
    """Per job: steps reached, mark, metrics, late test return. Per (question, who): the measures pooled over the
    second half of the blame job's steps ("scores"), and (steps, accuracy, majority) at every log point ("curves")."""
    res = {"rec": rec, "scores": {}, "curves": {}}
    for kind in KINDS:
        status, t = job_state(rec, kind, alive)
        m = job_metrics(rec, kind)
        mark = "*" if status == "running" else "!" if status in ("stopped", "crashed") else ""
        res[kind] = {"t": t, "mark": mark, "metrics": m, "return": late_mean(m, "test_return_mean", t // 2)}
    res["marks"] = "".join(s for s in "*!" if s in res["qmix"]["mark"] + res["blame"]["mark"])
    t_half = res["blame"]["t"] // 2
    for q, _ in QUESTIONS:
        for who in WHO:
            cms = counts(res["blame"]["metrics"], q, who)
            late = [c for t, c in cms if t >= t_half] or [c for _, c in cms[-1:]]
            pooled = [[sum(c[i][j] for c in late) for j in range(len(c))] for i, c in enumerate(late[0])] if late else [[0]]
            res["scores"][q, who] = measures(pooled, q == "blame")
            res["curves"][q, who] = list(zip(*[(t, *measures(c, q == "blame")[:2]) for t, c in cms])) or ([], [], [])
    return res


def summary(recs, folder, title):
    """Print and save one table (two lines per run: each agent's answers, the team vote; plus their MEAN if there
    are several runs) and the plot."""
    results = [run_result(rec, live_jobs()) for rec in recs]
    w = max(len(rec["id"]) for rec in recs) + 4  # room for the ID and its marks

    def rows(first, seed, t_max, qmix, blame, scores):
        lines = []
        for i, who in enumerate(WHO):
            left = f"{first:<{w}}{seed:<6}{t_max:<12}{qmix:<8.3f}{blame:<9.3f}" if i == 0 else " " * (w + 35)
            cells = "".join(f"{'{:.3f}/{:.3f} {:.3f}/{:.3f}'.format(*scores[q, who]):<24}" for q, _ in QUESTIONS)
            lines.append(f"   {left}{who:<6}{cells}".rstrip())
        return lines

    out = [title, "",
           f"   {'run':<{w}}{'seed':<6}{'t_max':<12}{'qmix':<8}{'blame':<9}{'':<6}" + "".join(f"{q:<24}" for q, _ in QUESTIONS).rstrip(),
           f"   {'':<{w}}{'':<6}{'':<12}{'return':<8}{'return':<9}{'':<6}" + ("acc/maj     sens/spec   " * len(QUESTIONS)).rstrip()]
    for r in results:
        rec = r["rec"]
        out += rows(f"{rec['id']} {r['marks']}", rec["seed"], f"{rec['t_max']:,}", r["qmix"]["return"], r["blame"]["return"], r["scores"])
    if len(results) > 1:
        out += rows("MEAN", "", "", mean(r["qmix"]["return"] for r in results), mean(r["blame"]["return"] for r in results),
                    {key: tuple(mean(r["scores"][key][i] for r in results) for i in range(4)) for key in results[0]["scores"]})
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
    """Left: test return of both jobs. Right: a panel per question (columns) and who (rows), with accuracy (colour)
    and the majority baseline (grey dashed). Lines are the mean over the runs, bands their min-max range."""
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
    fig.suptitle(f"{title}  ({len(results)} run{'s' if len(results) != 1 else ''}: line = mean, band = min to max)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
