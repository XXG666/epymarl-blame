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

from longrun.commands import cmd_delete, cmd_list, cmd_resume, cmd_run, cmd_status, cmd_stop
from longrun.summary import cmd_summary


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
