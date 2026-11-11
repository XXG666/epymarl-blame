"""Where things are, and how the two jobs of a run are trained."""
import pathlib


# The repository folder. This file is tools/longrun/config.py, so the repository is two folders up.
ROOT = pathlib.Path(__file__).resolve().parents[2]

# The environment every job is trained on.
ENV_KEY = "lbforaging:Foraging-2s-10x10-3p-3f-v3"

# Sacred (EPyMARL's logger) gives every training process a numbered folder in here.
SACRED_DIR = ROOT / "results" / "sacred" / "qmix" / ENV_KEY

# One folder per run: its record (run.json), the jobs' logs and their checkpoints.
RUNS_DIR = ROOT / "results" / "long_runs" / "runs"

# One folder per summary of several runs.
SUMMARIES_DIR = ROOT / "results" / "long_runs" / "summaries"

# Settings both jobs share that are not hyperparameters of the algorithm: the environment, the device,
# how often to log and test, and the checkpoints.
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

# The benchmark's hyperparameters of both jobs: Papoudakis et al. 2021, Table 28 (QMIX, LBF).
BENCH_SETTINGS = [
    "use_rnn=True",
    "lr=0.0003",
    "epsilon_anneal_time=200000",
    "target_update_interval_or_tau=0.01",
    "evaluation_epsilon=0.05",
]

# The two jobs of a run, and the settings in which they differ.
KINDS = {
    "qmix": ["blame_head=False", "split_backward=False"],
    "blame": ["blame_head=True", "split_backward=True"],
}
