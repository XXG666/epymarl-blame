"""The TD error delta = Q_tot - target of every step, recomputed from a run's saved checkpoints on fresh test
episodes (epsilon 0.05, like the held-out evaluation), at several training stages.

    scan   <run_id> <job> <episodes> <step> [<step> ...]    play episodes per checkpoint, save per-step records
    report <npz> [<npz> ...]                                 statistics of the saved records

The checkpoints hold no reward-standardisation statistics, so the running mean and variance of the reward
are rebuilt from the job's logged training returns (see reward_stats) and validated against the logged
held-out base rate: the fraction of steps with delta > 0 must match.  The target networks are not saved
either; the live networks stand in for them (tau = 0.01, so they trail the live ones by ~100 updates).
"""
import ast
import pathlib
import sys

import numpy as np
import torch as th
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]  # the repository: tools/delta_scan.py -> ..
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from types import SimpleNamespace as SN  # noqa: E402

from components.episode_buffer import ReplayBuffer  # noqa: E402
from components.transforms import OneHot  # noqa: E402
from controllers import REGISTRY as mac_REGISTRY  # noqa: E402
from learners import REGISTRY as le_REGISTRY  # noqa: E402
from runners import REGISTRY as r_REGISTRY  # noqa: E402
from utils.logging import Logger, get_logger  # noqa: E402
from longrun.config import BENCH_SETTINGS, COMMON_SETTINGS, KINDS  # noqa: E402
from longrun.jobs import job_metrics, load_run, run_dir  # noqa: E402

RHO_RANDOM = 0.2248  # E[r^2] / E[r] over pickups under a random policy (measured; the early training phase)


# ----------------------------------------------------------------------------------------------- config
def update(d, u):
    for k, v in u.items():
        d[k] = update(d.get(k, {}), v) if isinstance(v, dict) else v
    return d


def parse(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def build_args(job, seed):
    """The same config main.py assembles for a run_long job: default.yaml <- gymma.yaml <- qmix.yaml <- settings."""
    cfg = yaml.load((ROOT / "src/config/default.yaml").read_text(), Loader=yaml.FullLoader)
    cfg = update(cfg, yaml.load((ROOT / "src/config/envs/gymma.yaml").read_text(), Loader=yaml.FullLoader))
    cfg = update(cfg, yaml.load((ROOT / "src/config/algs/qmix.yaml").read_text(), Loader=yaml.FullLoader))
    for s in COMMON_SETTINGS + BENCH_SETTINGS + KINDS[job] + [f"seed={seed}"]:
        key, value = s.split("=", 1)
        d = cfg
        for part in key.split(".")[:-1]:
            d = d[part]
        d[key.split(".")[-1]] = parse(value)
    cfg["env_args"]["seed"] = seed
    args = SN(**cfg)
    args.device = "cpu"
    return args


# ----------------------------------------------------------------------------------------------- setup
def build(job, seed, n_episodes):
    """Runner, buffer (n_episodes slots), controller and learner, built exactly as run.py builds them."""
    args = build_args(job, seed)
    th.set_num_threads(1)
    np.random.seed(seed)
    th.manual_seed(seed)
    logger = Logger(get_logger())
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents, args.n_actions, args.state_shape = env_info["n_agents"], env_info["n_actions"], env_info["state_shape"]
    args.n_blame_outputs = args.n_agents + 1 if getattr(args, "blame_head", False) else 0
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,)},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    buffer = ReplayBuffer(scheme, groups, n_episodes, env_info["episode_limit"] + 1, preprocess=preprocess, device="cpu")
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    return args, runner, buffer, learner


def checkpoint(run_id, job, step):
    """The checkpoint folder of the job nearest to the step."""
    dirs = [d for d in (run_dir(run_id) / job / "models").glob("*/*") if d.name.isdigit()]
    best = min(dirs, key=lambda d: abs(int(d.name) - step))
    return best, int(best.name)


def play(runner, buffer, n_episodes):
    """n_episodes test episodes (epsilon 0.05) into the buffer; returns them as one batch, time-trimmed."""
    for _ in range(n_episodes):
        buffer.insert_episode_batch(runner.run(test_mode=True))
    batch = buffer[np.arange(n_episodes)]  # index with an array: a copy, not a view, so later stages cannot overwrite it
    return batch[:, : batch.max_t_filled()]


# ----------------------------------------------------------------------------------------------- reward statistics
def pickup_ratio(batch):
    """rho = E[r^2] / E[r] over the pickups (non-zero team rewards) of the batch."""
    r = batch["reward"][:, :-1, 0]
    r = r[(batch["filled"][:, :-1, 0] > 0) & (r > 0)]
    return (r ** 2).sum().item() / r.sum().item()


def reward_stats(run_id, job, rho_at, up_to=None):
    """(mean, variance) of the per-slot training reward as the learner's RunningMeanStd holds them at step
    up_to (default: the end of the run).  Every update adds a batch of 32 sampled episodes x 50 reward slots
    (padding counts as reward 0), so the statistics are those of the episodes sampled so far: the mean is the
    training return per 50 slots and E[r^2] is the return times rho (E[r^2]/E[r] over pickups).  An episode
    played at step u has been sampled about 32 * min(1, (s - u) / B) times by step s, B being the buffer's
    span in steps (5000 episodes x mean length), so the newest part of the history weighs less.
    rho_at: {step: rho} measured at the scanned stages; before the first stage the random-policy value is
    used.  Each 10k-step interval of the log holds 10k / mean length episodes."""
    m = job_metrics(load_run(run_id), job)
    steps, returns = m["return_mean"]["steps"], m["return_mean"]["values"]
    lengths = dict(zip(m["ep_length_mean"]["steps"], m["ep_length_mean"]["values"]))
    stages = sorted(rho_at)
    s = steps[-1] if up_to is None else up_to
    n = r1 = r2 = 0.0
    for t, ret in zip(steps, returns):
        if t >= s:
            break
        length = lengths.get(t, 50)
        episodes = 10000 / length * min(1.0, (s - t) / (5000 * length))
        rho = RHO_RANDOM if t < stages[0] else rho_at[min(stages, key=lambda x: abs(x - t))]
        n += episodes * 50
        r1 += episodes * ret
        r2 += episodes * ret * rho
    mean = r1 / n
    return mean, r2 / n - mean ** 2


def logged_base_rate(run_id, job, step, window=150000):
    """Fraction of held-out steps with blame truth 1 (delta > 0) in the job's log around the step."""
    m = job_metrics(load_run(run_id), job)
    key = lambda i, j: m.get(f"cm_held_out_blame_vote_{i}_{j}", {"steps": [], "values": []})
    tot = pos = 0.0
    for i in range(2):
        for j in range(2):
            s = key(i, j)
            for t, v in zip(s["steps"], s["values"]):
                if abs(t - step) <= window:
                    tot += v
                    pos += v if i == 1 else 0
    return pos / tot if tot else float("nan")


# ----------------------------------------------------------------------------------------------- delta
def deltas(learner, batch):
    """Per step: delta = Q_tot - target (the learner's td_error), dL/dQ_a, the real-step mask and the raw reward.
    The same arithmetic as QLearner.held_out, with the current networks, nothing updated."""
    args = learner.args
    rewards = batch["reward"][:, :-1]
    actions = batch["actions"][:, :-1]
    terminated = batch["terminated"][:, :-1].float()
    mask = batch["filled"][:, :-1].float()
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    avail_actions = batch["avail_actions"]
    raw = rewards[..., 0].clone()
    rewards = (rewards - learner.rew_ms.mean) / th.sqrt(learner.rew_ms.var)
    rewards = rewards.expand(-1, -1, args.n_agents)
    with th.no_grad():
        mac_out = learner._rollout(learner.mac, batch)
        q_out = mac_out[..., : args.n_actions]
        q_agents = th.gather(q_out[:, :-1], dim=3, index=actions).squeeze(3)
        target_mac_out = learner._rollout(learner.target_mac, batch)[:, 1:, :, : args.n_actions]
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        live = q_out.clone()
        live[avail_actions == 0] = -9999999
        target_max_qvals = th.gather(target_mac_out, 3, live[:, 1:].max(dim=3, keepdim=True)[1]).squeeze(3)
        targets = rewards + args.gamma * (1 - terminated) * learner.target_mixer(target_max_qvals, batch["state"][:, 1:])
    q_agents_cut = q_agents.detach().requires_grad_(True)
    td_error = learner.mixer(q_agents_cut, batch["state"][:, :-1]) - targets
    mask = mask.expand_as(td_error)
    loss = ((td_error * mask) ** 2).sum() / mask.sum()
    dL_dQa = th.autograd.grad(loss, q_agents_cut)[0]
    return td_error[..., 0].detach(), dL_dQa.detach(), mask[..., 0] > 0, raw, terminated[..., 0]


# ----------------------------------------------------------------------------------------------- commands
def scan(run_id, job, n_episodes, steps, out_dir):
    rec = load_run(run_id)
    args, runner, buffer, learner = build(job, rec["seed"], n_episodes)
    # pass 1: play every stage first, so that the pickup ratio of every stage is known before the reward
    # statistics (which need all of them) are rebuilt
    played = {}
    for step in steps:
        ckpt, at = checkpoint(run_id, job, step)
        learner.load_models(str(ckpt))
        played[at] = play(runner, buffer, n_episodes)
        print(f"{run_id}/{job} @ {at:>9,}: played {n_episodes} episodes, mean return "
              f"{played[at]['reward'][:, :-1, 0].sum(1).mean().item():.3f}, pickup rho {pickup_ratio(played[at]):.4f}")
    rho_at = {at: pickup_ratio(b) for at, b in played.items()}
    # pass 2: delta of every step with the reward statistics the learner held at that stage
    for at, batch in played.items():
        ckpt, _ = checkpoint(run_id, job, at)
        learner.load_models(str(ckpt))
        mean, var = reward_stats(run_id, job, rho_at, up_to=at)
        learner.rew_ms.mean[:] = mean
        learner.rew_ms.var[:] = var
        td, g, real, raw, term = deltas(learner, batch)
        n, T = td.shape
        ep = th.arange(n).unsqueeze(1).expand(n, T)
        t = th.arange(T).unsqueeze(0).expand(n, T)
        nxt = th.zeros_like(raw)
        nxt[:, :-1] = raw[:, 1:]
        sel = real
        out = out_dir / f"delta_{run_id}_{job}_{at}.npz"
        np.savez(out, step=at, delta=td[sel].numpy(), w=g[sel].numpy(), r=raw[sel].numpy(), r_next=nxt[sel].numpy(),
                 t=t[sel].numpy(), ep=ep[sel].numpy(), terminal=term[sel].numpy(), mean=mean, var=var,
                 logged_rate=logged_base_rate(run_id, job, at))
        print(f"  {at:>9,}: {int(sel.sum())} steps, delta > 0: {(td[sel] > 0).float().mean():.3f} "
              f"(logged held-out base rate {logged_base_rate(run_id, job, at):.3f}), mean |delta| {td[sel].abs().mean():.3f}"
              f"  -> {out.name}")
    runner.close_env()


def corrected(files):
    """The saved records with delta recomputed for the reward statistics the learner held at that stage.
    Only the standardised reward enters the target, so delta_new = delta_saved + r'_saved - r'_new."""
    out = []
    by_run = {}
    for f in files:
        parts = pathlib.Path(f).stem.split("_")  # delta_<run_id>_<job>_<step>; the run id may contain underscores
        by_run.setdefault(("_".join(parts[1:-2]), parts[-2]), []).append(f)
    for (run_id, job), fs in by_run.items():
        data = {int(np.load(f)["step"]): np.load(f) for f in fs}
        rho_at = {s: float((d["r"][d["r"] > 0] ** 2).sum() / d["r"][d["r"] > 0].sum()) for s, d in data.items()}
        for s, d in sorted(data.items()):
            mean, var = reward_stats(run_id, job, rho_at, up_to=s)
            r = d["r"]
            r_saved = (r - float(d["mean"])) / float(d["var"]) ** 0.5
            r_new = (r - mean) / var ** 0.5
            out.append((f"{run_id}/{job}", s, dict(d), d["delta"] + r_saved - r_new, mean, var))
    return out


def report(files):
    rows = []
    print(f"{'run':<12}{'step':>9} {'mean':>8} {'std':>8}   rebuilt reward statistics at that stage")
    for name, s, d, delta, mean, var in corrected(files):
        print(f"{name:<12}{s:>9,} {mean:>8.5f} {var ** 0.5:>8.5f}")
    print()
    for name, s, d, delta, mean, var in corrected(files):
        r, r_next = d["r"], d["r_next"]
        med = np.median(delta)
        scale = 1.4826 * np.median(np.abs(delta - med))  # robust std of the core
        q = np.quantile(delta, [0.5, 0.75, 0.9, 0.95, 0.99])
        rows.append((int(d["step"]), len(delta), float(d["logged_rate"]), (delta > 0).mean(), delta.mean(), med, delta.std(),
                     scale, q, (r > 0).mean(), delta[r > 0].mean() if (r > 0).any() else np.nan,
                     delta[(r == 0) & (r_next == 0)].mean(), (delta > med + 2 * scale).mean(), (delta > med + 3 * scale).mean(),
                     (delta > 2 * scale).mean(), (delta > 3 * scale).mean(), name))
    print(f"{'step':>9} {'n':>6} {'logged':>7} {'d>0':>6} {'mean':>7} {'median':>7} {'std':>6} {'scale':>6} "
          f"{'q50':>6} {'q75':>6} {'q90':>6} {'q95':>6} {'q99':>6} {'pick%':>6} {'d|pick':>7} {'d|quiet':>8} "
          f"{'>m+2s':>6} {'>m+3s':>6} {'>2s':>6} {'>3s':>6}")
    for step, n, logged, pos, mean, med, std, scale, q, pick, dpick, dquiet, m2, m3, a2, a3, name in rows:
        print(f"{step:>9,} {n:>6} {logged:>7.3f} {pos:>6.3f} {mean:>7.3f} {med:>7.3f} {std:>6.3f} {scale:>6.3f} "
              + " ".join(f"{x:>6.2f}" for x in q) + f" {pick:>6.3f} {dpick:>7.2f} {dquiet:>8.3f} {m2:>6.3f} {m3:>6.3f} {a2:>6.3f} {a3:>6.3f}")
    print("scale = 1.4826 * MAD (robust std of the core);  pick% = steps with a pickup;  d|pick, d|quiet = mean delta at pickup"
          " steps / at steps with no pickup now or next;  >m+ks = share of steps above median + k scale;  >ks = above k scale")


if __name__ == "__main__":
    out_dir = ROOT / "results" / "long_runs" / "deltas"
    out_dir.mkdir(exist_ok=True)
    if sys.argv[1] == "scan":
        scan(sys.argv[2], sys.argv[3], int(sys.argv[4]), [int(s) for s in sys.argv[5:]], out_dir)
    elif sys.argv[1] == "report":
        report(sys.argv[2:])
