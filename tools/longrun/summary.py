"""summary: measures computed from the logged confusion counts, the table and the plot.

Words used in this file:
    metrics     everything a job logged, as jobs.job_metrics returns it:
                {metric name: {"steps": [...], "values": [...]}}
    series      one entry of metrics, {"steps": [...], "values": [...]}: a value at each of these steps
    question    what a head is asked: "whether", "who" or "combined" (LEGEND says what they mean)
    answerer    who gives the answer: "each" or "vote" (LEGEND again)
    matrix      a confusion matrix, as a list of rows: matrix[truth][answer] is how often the true answer
                was truth and the given answer was answer. Training logs every cell as a metric of its own,
                named cm_<set><question>_<answerer>_<truth>_<answer>, where <set> is "" for the training
                batches and "held_out_" for the held-out test episodes.
    score       {"accuracy": ..., "sensitivity": ..., "specificity": ...} computed from one matrix
    result      everything the table and the plot need to know about one run (see grade_run)
"""
import json
import math

import matplotlib

matplotlib.use("Agg")  # draw into a file; WSL usually has no screen
import matplotlib.pyplot as plt
import numpy as np

from .config import KINDS, ROOT, SUMMARIES_DIR
from .jobs import commit_text, job_metrics, job_state, live_jobs, load_runs, run_dir, unique_ids


# ---------------------------------------------------------------------------------------------------------
# Names, labels and colours
# ---------------------------------------------------------------------------------------------------------

# The questions, by the names they are logged with.
QUESTIONS = ["whether", "who", "combined"]

# The questions as shown in the table and the plot.
QUESTION_TITLE = {"whether": "whether", "who": "who", "combined": "blame"}

# The names of two questions in runs logged before the renaming.
OLD_QUESTION_NAME = {"whether": "blame", "who": "importance"}

# Who answered, as logged: every agent on its own, and the team's majority vote.
ANSWERERS = ["each", "vote"]

# The answerers as shown in the table and the plot.
ANSWERER_LABEL = {"each": "each", "vote": "maj"}

# The three numbers of a score.
MEASURES = ["accuracy", "sensitivity", "specificity"]

# What comes before the question in the name of a logged count: the set the answers were graded on.
TRAINING_PREFIX = ""
HELD_OUT_PREFIX = "held_out_"

# The plot: the line of each job in the return panel.
JOB_COLOUR = {"qmix": "#8a8984", "blame": "#2a78d6"}
JOB_LINE_STYLE = {"qmix": "--", "blame": "-"}
JOB_LABEL = {"qmix": "benchmark QMIX", "blame": "blame QMIX"}

# The plot: the agents' line in the panels of each question.
QUESTION_COLOUR = {"whether": "#e87ba4", "who": "#eda100", "combined": "#1baf7a"}

# The table: how many characters each column is wide. The column of the run's ID depends on the IDs.
SEED_WIDTH = 6
T_MAX_WIDTH = 12
QMIX_WIDTH = 8
BLAME_WIDTH = 9
ANSWERER_WIDTH = 18
QUESTION_WIDTH = {"whether": 20, "who": 12, "combined": 12}

# The table: the second heading line of each question's column.
QUESTION_SUBTITLE = {"whether": "acc    sens/spec", "who": "acc", "combined": "acc"}

# The table: the explanation printed below it.
LEGEND = [
    "   pooled over the second half of each job's steps",
    "   return     = test return (1.0 = all food collected)",
    ("   whether    = did the step go worse than predicted by more than the threshold (1) or not (0): "
     + "dL/dQ_tot above whether_threshold;  who = which agent has the largest mixer weight"),
    "   blame      = the final answer: the agent named by who if whether says yes, else nobody",
    "   each       = every agent's own answer;  maj = the team's majority vote",
    "                all graded on the held-out test episodes (epsilon 0.05), which nothing is trained on;",
    ("                (train) = graded on the training batches instead, "
     + "because a run in this summary has no held-out data"),
    "   acc        = accuracy: the share of answers that are right",
    ("   sens/spec  = sensitivity / specificity of the whether answer "
     + "(answer 1, worse than predicted by more than the threshold, is the positive)"),
    "   *  still running (not final)    !  stopped or crashed",
]


# ---------------------------------------------------------------------------------------------------------
# Means and scores
# ---------------------------------------------------------------------------------------------------------

def mean(values):
    """In: a list of numbers. Out: their mean without the missing values (nan); nan if nothing is left."""
    present = []
    for value in values:
        if not math.isnan(value):
            present.append(value)
    if len(present) == 0:
        return float("nan")
    # the built-in sum, not a loop of additions: the two can differ in a float's last digits
    return sum(present) / len(present)


def logged_series(metrics, name):
    """In: a job's metrics and a metric's name. Out: the metric's series, {"steps": [...], "values": [...]};
    an empty series if the job has not logged this metric."""
    if name in metrics:
        return metrics[name]
    return {"steps": [], "values": []}


def late_mean(metrics, name, first_step):
    """Mean of a metric's values from step first_step on (the last value if nothing was logged there yet).
    In: a job's metrics, a metric's name and a step. Out: a number; nan if the metric has no values."""
    series = logged_series(metrics, name)
    late_values = []
    for step, value in zip(series["steps"], series["values"]):
        if step >= first_step:
            late_values.append(value)
    if len(late_values) == 0 and len(series["values"]) > 0:
        late_values = [series["values"][-1]]
    return mean(late_values)


def empty_score():
    """Out: the score of an answerer with no answers: every measure is missing (nan)."""
    nan = float("nan")
    return {"accuracy": nan, "sensitivity": nan, "specificity": nan}


def score_of(matrix):
    """In: one confusion matrix; rows = truth, columns = answer. Out: its score.
    Accuracy is the share of answers on the diagonal, where the answer equals the truth.
    Sensitivity and specificity are for a yes/no question (a 2 x 2 matrix), with answer 1 as the positive;
    they are nan for the other questions."""
    score = empty_score()
    total = 0
    for row in matrix:
        total = total + sum(row)
    if total == 0:
        return score

    right = 0
    for position in range(len(matrix)):
        right = right + matrix[position][position]
    score["accuracy"] = right / total

    if len(matrix) == 2:
        truly_yes = sum(matrix[1])  # row 1: every case whose true answer is 1
        truly_no = sum(matrix[0])  # row 0: every case whose true answer is 0
        if truly_yes != 0:
            score["sensitivity"] = matrix[1][1] / truly_yes
        if truly_no != 0:
            score["specificity"] = matrix[0][0] / truly_no
    return score


# ---------------------------------------------------------------------------------------------------------
# Reading the logged confusion counts
# ---------------------------------------------------------------------------------------------------------

def names_starting_with(metrics, start):
    """In: a job's metrics and the start of a name. Out: the list of metric names that start like this."""
    names = []
    for name in metrics:
        if name.startswith(start):
            names.append(name)
    return names


def confusion_counts(metrics, question, answerer, set_prefix):
    """The confusion matrices of one question and one answerer, one matrix per log point.
    In: a job's metrics, the question, the answerer, and TRAINING_PREFIX or HELD_OUT_PREFIX.
    Out: (steps, matrices): the steps of the log points and the matrix logged at each of them. Both lists
    are empty if the job has not logged these counts."""
    name_start = f"cm_{set_prefix}{question}_{answerer}_"
    names = names_starting_with(metrics, name_start)
    if len(names) == 0 and question in OLD_QUESTION_NAME:
        # a run logged before the two questions were renamed
        old_name = OLD_QUESTION_NAME[question]
        name_start = f"cm_{set_prefix}{old_name}_{answerer}_"
        names = names_starting_with(metrics, name_start)
    if len(names) == 0:
        return [], []

    # every name ends in _<truth>_<answer>: the largest answer number tells how many answers there are
    size = 0
    for name in names:
        name_parts = name.split("_")
        answer = int(name_parts[-1])
        size = max(size, answer + 1)

    # training logs all cells of a matrix together: every cell has the same steps
    steps = metrics[names[0]]["steps"]
    matrices = []
    for position in range(len(steps)):
        matrix = []
        for truth in range(size):
            row = []
            for answer in range(size):
                cell = metrics[f"{name_start}{truth}_{answer}"]
                row.append(cell["values"][position])
            matrix.append(row)
        matrices.append(matrix)
    return steps, matrices


def has_counts(metrics, set_prefix):
    """In: a job's metrics, and TRAINING_PREFIX or HELD_OUT_PREFIX. Out: True if the job has logged counts
    on this set at one log point or more. The team's vote on whether stands for all of them."""
    steps, matrices = confusion_counts(metrics, "whether", "vote", set_prefix)
    return len(steps) > 0


def add_matrices(matrices):
    """In: a list of confusion matrices of the same size, at least one. Out: one matrix of that size; each
    of its cells is the sum of this cell over all the matrices."""
    size = len(matrices[0])
    pooled = []
    for truth in range(size):
        row = []
        for answer in range(size):
            total = 0
            for matrix in matrices:
                total = total + matrix[truth][answer]
            row.append(total)
        pooled.append(row)
    return pooled


# ---------------------------------------------------------------------------------------------------------
# Grading a run
# ---------------------------------------------------------------------------------------------------------

def grade_question(metrics, question, answerer, set_prefix, first_late_step):
    """In: the blame job's metrics, a question, an answerer, the prefix of the set to grade on, and the
    step at which the second half of the job begins.
    Out: (score, curve). score: from the counts of all log points in the second half, added up.
    curve: a series {"steps": [...], "values": [...]} with the accuracy at every log point."""
    steps, matrices = confusion_counts(metrics, question, answerer, set_prefix)

    late_matrices = []
    for step, matrix in zip(steps, matrices):
        if step >= first_late_step:
            late_matrices.append(matrix)
    if len(late_matrices) == 0 and len(matrices) > 0:
        # nothing was logged in the second half yet: take the last log point there is
        late_matrices = [matrices[-1]]
    if len(late_matrices) == 0:
        score = empty_score()
    else:
        score = score_of(add_matrices(late_matrices))

    accuracies = []
    for matrix in matrices:
        accuracies.append(score_of(matrix)["accuracy"])
    curve = {"steps": steps, "values": accuracies}
    return score, curve


def grade_job(rec, kind, alive):
    """In: a run's record, the job's kind and the result of live_jobs().
    Out: {"status": ..., "steps": steps reached, "metrics": ..., "return": ...} of this job, where the
    return is the mean test return over the second half of the job's steps."""
    status, steps = job_state(rec, kind, alive)
    metrics = job_metrics(rec, kind)
    test_return = late_mean(metrics, "test_return_mean", steps // 2)
    return {"status": status, "steps": steps, "metrics": metrics, "return": test_return}


def run_marks(statuses):
    """In: the list of the statuses of a run's jobs. Out: the marks shown after the run's ID: "*" if a job
    is still running and "!" if a job stopped or crashed (both if both)."""
    marks = ""
    if "running" in statuses:
        marks = marks + "*"
    if "stopped" in statuses or "crashed" in statuses:
        marks = marks + "!"
    return marks


def grade_run(rec, alive, held_out):
    """In: a run's record, the result of live_jobs(), and whether held-out data may be used.
    Out: the run's result, a dictionary:
        "rec"               the run's record
        "qmix", "blame"     what grade_job returns for each job
        "marks"             what run_marks returns
        "set"               the set the blame job's heads are graded on: "held-out" (the held-out test
                            episodes, when the run has them and held_out is True), else "training" (its
                            training batches); None while the job has logged no counts at all
        "scores"            scores[question][answerer] = the score pooled over the second half of the steps
        "curves"            curves[question][answerer] = a series with the accuracy at every log point"""
    result = {"rec": rec}
    statuses = []
    for kind in KINDS:
        result[kind] = grade_job(rec, kind, alive)
        statuses.append(result[kind]["status"])
    result["marks"] = run_marks(statuses)

    metrics = result["blame"]["metrics"]
    first_late_step = result["blame"]["steps"] // 2

    if held_out and has_counts(metrics, HELD_OUT_PREFIX):
        set_prefix = HELD_OUT_PREFIX
        result["set"] = "held-out"
    elif has_counts(metrics, TRAINING_PREFIX):
        set_prefix = TRAINING_PREFIX
        result["set"] = "training"
    else:
        # nothing logged yet: the prefix finds no counts, so every score below is empty
        set_prefix = TRAINING_PREFIX
        result["set"] = None

    result["scores"] = {}
    result["curves"] = {}
    for question in QUESTIONS:
        result["scores"][question] = {}
        result["curves"][question] = {}
        for answerer in ANSWERERS:
            score, curve = grade_question(metrics, question, answerer, set_prefix, first_late_step)
            result["scores"][question][answerer] = score
            result["curves"][question][answerer] = curve
    return result


def grade_runs(recs, held_out):
    """In: a list of run records, and whether held-out data may be used. Out: the list of their results."""
    results = []
    for rec in recs:
        results.append(grade_run(rec, live_jobs(), held_out))
    return results


# ---------------------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------------------

def left_columns(run_text, seed_text, t_max_text, qmix_text, blame_text, id_width):
    """In: the texts of the five columns left of the answers, and the width of the first one.
    Out: one string with every text filled up with spaces to its column's width."""
    text = run_text.ljust(id_width)
    text = text + seed_text.ljust(SEED_WIDTH)
    text = text + t_max_text.ljust(T_MAX_WIDTH)
    text = text + qmix_text.ljust(QMIX_WIDTH)
    text = text + blame_text.ljust(BLAME_WIDTH)
    return text


def header_lines(id_width):
    """In: the width of the ID column. Out: the two heading lines of the table, as a list of strings."""
    first = "   " + left_columns("run", "seed", "t_max", "qmix", "blame", id_width)
    second = "   " + left_columns("", "", "", "return", "return", id_width)
    first = first + " " * ANSWERER_WIDTH
    second = second + " " * ANSWERER_WIDTH
    for question in QUESTIONS:
        width = QUESTION_WIDTH[question]
        first = first + QUESTION_TITLE[question].ljust(width)
        second = second + QUESTION_SUBTITLE[question].ljust(width)
    return [first.rstrip(), second.rstrip()]


def score_cell(question, score):
    """In: a question and a score. Out: the score as text for the table: the accuracy for every question,
    and sensitivity/specificity as well for whether, the yes/no question."""
    accuracy = score["accuracy"]
    accuracy_text = f"{accuracy:.3f}"
    if question == "whether":
        # "0.812  " + "0.640/0.910" + "  " is 7 + 11 + 2 = 20 characters: QUESTION_WIDTH["whether"].
        # The cell is not filled up to 20: a missing value prints as the shorter "nan".
        sensitivity = score["sensitivity"]
        specificity = score["specificity"]
        return accuracy_text.ljust(7) + f"{sensitivity:.3f}/{specificity:.3f}" + "  "
    return accuracy_text.ljust(QUESTION_WIDTH[question])


def score_lines(left, id_width, scores, note):
    """The table lines of one run, or of the MEAN: one line per answerer.
    In: left = the text of the five left columns, which only the first line shows; the width of the ID
    column; scores[question][answerer]; note = the text added to every answerer's label.
    Out: a list of strings."""
    blank = left_columns("", "", "", "", "", id_width)
    lines = []
    for answerer in ANSWERERS:
        if len(lines) == 0:
            line = "   " + left
        else:
            line = "   " + blank
        answerer_text = ANSWERER_LABEL[answerer] + note
        line = line + answerer_text.ljust(ANSWERER_WIDTH)
        for question in QUESTIONS:
            line = line + score_cell(question, scores[question][answerer])
        lines.append(line.rstrip())
    return lines


def run_lines(result, id_width, note):
    """In: one run's result, the width of the ID column and the note for the answerers' labels.
    Out: the run's lines of the table, as a list of strings."""
    rec = result["rec"]
    run_id = rec["id"]
    marks = result["marks"]
    t_max = rec["t_max"]
    qmix_return = result["qmix"]["return"]
    blame_return = result["blame"]["return"]

    run_text = f"{run_id} {marks}"
    seed_text = str(rec["seed"])
    t_max_text = f"{t_max:,}"
    qmix_text = f"{qmix_return:.3f}"
    blame_text = f"{blame_return:.3f}"
    left = left_columns(run_text, seed_text, t_max_text, qmix_text, blame_text, id_width)
    return score_lines(left, id_width, result["scores"], note)


def mean_score(results, question, answerer):
    """In: the runs' results, a question and an answerer. Out: a score; each of its measures is the mean of
    this measure over the runs that have it."""
    score = {}
    for measure in MEASURES:
        values = []
        for result in results:
            values.append(result["scores"][question][answerer][measure])
        score[measure] = mean(values)
    return score


def mean_lines(results, id_width, note):
    """In: the runs' results, the width of the ID column and the note for the answerers' labels.
    Out: the MEAN lines of the table, as a list of strings."""
    qmix_returns = []
    blame_returns = []
    for result in results:
        qmix_returns.append(result["qmix"]["return"])
        blame_returns.append(result["blame"]["return"])
    qmix_return = mean(qmix_returns)
    blame_return = mean(blame_returns)

    scores = {}
    for question in QUESTIONS:
        scores[question] = {}
        for answerer in ANSWERERS:
            scores[question][answerer] = mean_score(results, question, answerer)

    qmix_text = f"{qmix_return:.3f}"
    blame_text = f"{blame_return:.3f}"
    left = left_columns("MEAN", "", "", qmix_text, blame_text, id_width)
    return score_lines(left, id_width, scores, note)


def table_lines(results, title, note):
    """In: the runs' results, the title and the note for the answerers' labels.
    Out: everything the summary prints, as a list of lines: the title, the table and the legend."""
    # the ID column has room for the longest ID and its marks
    id_width = 0
    for result in results:
        id_width = max(id_width, len(result["rec"]["id"]))
    id_width = id_width + 4

    lines = [title, ""]
    lines.extend(header_lines(id_width))
    for result in results:
        lines.extend(run_lines(result, id_width, note))
    if len(results) > 1:
        lines.extend(mean_lines(results, id_width, note))
    lines.append("")
    lines.extend(LEGEND)
    return lines


# ---------------------------------------------------------------------------------------------------------
# The plot
# ---------------------------------------------------------------------------------------------------------

def draw_band(panel, series_list, colour, style, label):
    """Draw the mean of several series as a line and their min-max range as a shaded band.
    The series are put on a common grid first (linear interpolation; a run contributes only where it has
    data).
    In: the panel to draw in, a list of series {"steps": [...], "values": [...]} (one per run), and the
    line's colour, style and label for the legend (None for a line without one). Draws nothing if every
    series is empty."""
    with_data = []
    for series in series_list:
        if len(series["steps"]) > 0:
            with_data.append(series)
    if len(with_data) == 0:
        return

    last_step = 0
    for series in with_data:
        last_step = max(last_step, series["steps"][-1])
    grid = np.linspace(0, last_step, 200)  # (grid points,)

    # one row per run: its values at the grid's steps, nan where the run has no data
    run_rows = []
    for series in with_data:
        run_rows.append(np.interp(grid, series["steps"], series["values"], left=np.nan, right=np.nan))
    rows = np.array(run_rows)  # (runs, grid points)

    # keep the grid points where at least one run has data
    no_run_has_data = np.isnan(rows).all(axis=0)  # (grid points,)
    some_run_has_data = np.logical_not(no_run_has_data)
    grid = grid[some_run_has_data]
    rows = rows[:, some_run_has_data]

    mean_line = np.nanmean(rows, axis=0)
    panel.plot(grid, mean_line, color=colour, linestyle=style, linewidth=1.5, label=label)
    if len(with_data) > 1:
        lowest = np.nanmin(rows, axis=0)
        highest = np.nanmax(rows, axis=0)
        panel.fill_between(grid, lowest, highest, color=colour, alpha=0.15, linewidth=0)


def panel_name(question, answerer):
    """Out: the name of the plot's panel that shows one question answered by one answerer."""
    return f"{question}_{answerer}"


def steps_in_millions(steps, position):
    """Text of a tick on the step axis: 1500000 becomes "1.5M". matplotlib calls this with the tick's
    value and its position on the axis; the position is not needed."""
    millions = steps / 1e6
    return f"{millions:g}M"


def draw_returns(panel, results):
    """In: the return panel and the runs' results. Draws the test return of both jobs and the panel's
    title."""
    for kind in KINDS:
        series_list = []
        for result in results:
            series_list.append(logged_series(result[kind]["metrics"], "test_return_mean"))
        draw_band(panel, series_list, JOB_COLOUR[kind], JOB_LINE_STYLE[kind], JOB_LABEL[kind])
    panel.set_title("test return")


def draw_accuracies(panel, results, question, answerer):
    """In: a panel, the runs' results, and the panel's question and answerer.
    Draws the accuracy of the agents' answers and the panel's title."""
    series_list = []
    for result in results:
        series_list.append(result["curves"][question][answerer])
    # None: the panel's title says what the line is, so it needs no legend
    draw_band(panel, series_list, QUESTION_COLOUR[question], "-", None)

    question_title = QUESTION_TITLE[question]
    answerer_label = ANSWERER_LABEL[answerer]
    panel.set_title(f"{question_title} accuracy: {answerer_label}")


def panel_layout():
    """Out: the grid of panel names for the plot, one list per row. The return panel is named in the
    first place of both rows, so it fills the left of the picture over both rows."""
    layout = []
    for answerer in ANSWERERS:
        layout_row = ["return"]
        for question in QUESTIONS:
            layout_row.append(panel_name(question, answerer))
        layout.append(layout_row)
    return layout


def plot(results, path, title, graded):
    """Left: test return of both jobs. Right: a panel per question (columns) and answerer (rows) with the
    agents' accuracy on the graded episodes.
    Lines are the mean over the runs, bands their min-max range.
    In: the runs' results, the path of the picture to write, the title, and the set the heads are graded on
    ("held-out" or "training")."""
    layout = panel_layout()
    # the return panel is twice as wide as the accuracy panel of a question
    width_ratios = [2]
    for question in QUESTIONS:
        width_ratios.append(1)
    figure, panels = plt.subplot_mosaic(layout, figsize=(17, 6.5), width_ratios=width_ratios, sharex=True)

    draw_returns(panels["return"], results)
    for question in QUESTIONS:
        for answerer in ANSWERERS:
            panel = panels[panel_name(question, answerer)]
            draw_accuracies(panel, results, question, answerer)
    # the panels share the step axis: this starts all of them at step 0
    panels["return"].set_xlim(left=0)

    for name in panels:
        panel = panels[name]
        panel.set_ylim(0, 1)
        panel.grid(alpha=0.3)
        panel.xaxis.set_major_formatter(steps_in_millions)

    # the panels share the step axis: only the bottom ones name it
    bottom_answerer = ANSWERERS[-1]
    panels["return"].set_xlabel("environment steps")
    for question in QUESTIONS:
        panels[panel_name(question, bottom_answerer)].set_xlabel("environment steps")

    panels["return"].legend(fontsize=8, frameon=False)

    run_count = len(results)
    if run_count == 1:
        runs_text = "1 run"
    else:
        runs_text = f"{run_count} runs"
    figure.suptitle(f"{title}  ({runs_text}: line = mean, band = min to max; "
                    f"heads graded on {graded} episodes)")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


# ---------------------------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------------------------

def summary(recs, folder, title):
    """Print and save one table (per run: each agent's answers and the team's majority vote, graded on the
    held-out test episodes; plus their MEAN if there are several runs) and the plot. If some of the runs
    have no held-out data, all of them are graded on their training batches instead, so that the MEAN and
    the plot compare like with like.
    In: a list of run records, the folder to save in, and the title. Out: the list of the runs' results."""
    results = grade_runs(recs, True)  # True: on held-out episodes where a run has them
    some_run_is_held_out = False
    some_run_is_training = False
    for result in results:
        if result["set"] == "held-out":
            some_run_is_held_out = True
        if result["set"] == "training":
            some_run_is_training = True
    if some_run_is_held_out and some_run_is_training:
        results = grade_runs(recs, False)  # False: every run on its training batches

    if some_run_is_held_out and not some_run_is_training:
        graded = "held-out"
        note = ""
    else:
        graded = "training"
        note = " (train)"

    text = "\n".join(table_lines(results, title, note))
    print(text)

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.txt").write_text(text + "\n")
    plot(results, folder / "summary.png", title, graded)
    relative_folder = folder.relative_to(ROOT)
    print(f"\nsaved in {relative_folder}/")
    return results


def cmd_summary(args):
    """Print and save the summary of the runs args.ids: one run in its own folder, together with its jobs'
    metrics; several runs in a folder of results/long_runs/summaries/."""
    recs = load_runs(unique_ids(args.ids))
    if len(recs) == 1:
        rec = recs[0]
        run_id = rec["id"]
        commit = commit_text(rec)
        note = rec["note"]
        title = f"run {run_id}  commit {commit}  {note}".rstrip()
        results = summary(recs, run_dir(run_id), title)
        for kind in KINDS:
            job_dir = run_dir(run_id) / kind
            job_dir.mkdir(exist_ok=True)
            (job_dir / "metrics.json").write_text(json.dumps(results[0][kind]["metrics"]))
    else:
        ids = []
        for rec in recs:
            ids.append(rec["id"])
        name = "+".join(ids)
        summary(recs, SUMMARIES_DIR / name, f"summary {name}")
