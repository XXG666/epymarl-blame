r"""
Long runs: benchmark QMIX against blame-head QMIX on Foraging-2s-10x10-3p-3f-v3 (CPU).

A run has an ID and one seed, and trains two jobs with that seed:
    qmix    benchmark QMIX                         blame_head=False split_backward=False
    blame   benchmark QMIX + trained blame head    blame_head=True  split_backward=True
Benchmark settings: Papoudakis et al. 2021, Table 28 (QMIX, LBF).

COMMANDS
    run <id> --seed S [--t-max N] [--note TEXT]    start a run (it keeps going after the terminal closes)
    list                                           the runs that are training right now, and the state of their jobs
    status <id> ...                                progress bars, updated in place (Ctrl+C stops the display only)
    summary <id>                                   one run's results  -> results/long_runs/runs/<id>/
    summary <id> <id> ...                          several runs and their MEAN -> results/long_runs/summaries/<id>+<id>+.../
    delete <id> [--yes]                            remove the run's folder and its Sacred results

FILES in results/long_runs/runs/<id>/
    run.json                          seed, steps, commit, the jobs' command lines
    <job>.1.log                       output of the job
    <job>/models/                     checkpoints every 100k steps
    summary.txt, summary.png, <job>/metrics.json      written by summary
"""

import argparse

from longrun.commands import cmd_delete, cmd_list, cmd_run, cmd_status
from longrun.summary import cmd_summary


def build_parser():
    """Out: the parser of the command line: one sub-command per action, each with its own options.
    The text at the top of this file is what --help prints."""
    # the raw formatter prints the text as it is written, without re-wrapping its lines
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest="action", required=True)

    run_parser = actions.add_parser("run", help="start a run: benchmark QMIX and blame QMIX with one seed")
    run_parser.add_argument("id")
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--t-max", type=int, default=2_000_000)
    run_parser.add_argument("--note", default="")

    actions.add_parser("list", help="the runs that are training right now")

    status_parser = actions.add_parser("status")
    status_parser.add_argument("ids", nargs="+", metavar="id")

    summary_parser = actions.add_parser("summary", help="one run's results, or several runs and their MEAN")
    summary_parser.add_argument("ids", nargs="+", metavar="id")

    delete_parser = actions.add_parser("delete", help="remove a run's folder and its Sacred results")
    delete_parser.add_argument("id")
    delete_parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    return parser


def main():
    """Read the command line and call the function of the action it names."""
    args = build_parser().parse_args()
    if args.action == "run":
        cmd_run(args)
    elif args.action == "list":
        cmd_list(args)
    elif args.action == "status":
        cmd_status(args)
    elif args.action == "summary":
        cmd_summary(args)
    elif args.action == "delete":
        cmd_delete(args)


if __name__ == "__main__":
    main()
