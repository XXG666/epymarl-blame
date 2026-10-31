"""Where things are, and how the two jobs of a run are trained."""
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]  # the repository: tools/longrun/config.py -> ../..


ENV_KEY = "lbforaging:Foraging-2s-10x10-3p-3f-v3"


SACRED_DIR = ROOT / "results" / "sacred" / "qmix" / ENV_KEY


RUNS_DIR = ROOT / "results" / "long_runs" / "runs"


SUMMARIES_DIR = ROOT / "results" / "long_runs" / "summaries"


COMMON_SETTINGS = [
    "env_args.time_limit=50",
    f"env_args.key={ENV_KEY}",
    "use_cuda=False",
    "log_interval=10000",
    "runner_log_interval=10000",
    "learner_log_interval=10000",
    "test_interval=20000",
    "test_nepisode=20",
    "save_model=True",
    "save_model_interval=100000",  # checkpoints for resume
]


BENCH_SETTINGS = [
    "use_rnn=True",
    "lr=0.0003",
    "epsilon_anneal_time=200000",
    "target_update_interval_or_tau=0.01",
    "evaluation_epsilon=0.05",
]


KINDS = {
    "qmix": ["blame_head=False", "split_backward=False"],
    "blame": ["blame_head=True", "split_backward=True"],
}
