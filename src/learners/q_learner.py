import copy

import torch as th
from torch.optim import Adam
import torch.nn.functional as F

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                assert args.common_reward, "VDN only supports common reward setting"
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                assert args.common_reward, "QMIX only supports common reward setting"
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # --- split the combined [ Q | blame | importance ] output ---
        n_blame = getattr(self.args, "n_blame_outputs", 0)
        if n_blame > 0:
            q_out = mac_out[..., :-n_blame]
            b_out = mac_out[..., -n_blame:]  
        else:
            q_out = mac_out

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(q_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        if n_blame > 0:
            target_mac_out = target_mac_out[..., :-n_blame]  # Q part only

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = q_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        """
        Mixer, keep a pointer on the per-agent Q-value BEFORE the mixer as 
        the mixer call below rebinds `chosen_action_qvals` to Q_tot
        """
        q_agents = chosen_action_qvals  # (batch, steps, n_agents)
        q_agents_cut = None
        if self.mixer is not None:
            if getattr(self.args, "split_backward", False):
                # THE CUT: a fresh leaf that will collect dL/dQ_a
                q_agents_cut = q_agents.detach().requires_grad_(True)
                """
                Detached tensor to feed forward to the mixer, the backward will be done 
                on this tensor, and the gradient will stop here since it is detached.
                """
                mixer_in = q_agents_cut
            else:
                mixer_in = q_agents
            chosen_action_qvals = self.mixer(mixer_in, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:]
            )

        if self.args.standardise_returns:
            target_max_qvals = (
                target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            )

        # Calculate 1-step Q-Learning targets
        targets = (
            rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()
        )

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        if q_agents_cut is not None:
            # loss -> mixer, stops at the cut
            loss.backward()
            dL_dQa = q_agents_cut.grad
            assert dL_dQa is not None, "ERROR: cut received no gradient - check the graph"

            # ------------------- INTERRUPT -------------------

            # resume, pushing dL/dQ_a into the agent networks using the pointer (q_agents) we create in forward ---
            if n_blame > 0:
                # --- build the expanded gradient explicitly ---
                full_grad = th.zeros_like(mac_out) # q + b

                """
                Episode, Step (0..49 within the episode), Agent, Action
                full_grad[e][t][a][ actions[e][t][a][0] ]  =  dL_dQa[e][t][a][0]
                          ▲  ▲  ▲    ▲
                          │  │  │    └── comes from the INDEX
                          └──┴──┴─────── come straight from the position
                """
                full_grad[:, :-1, :, : self.args.n_actions].scatter_(
                    3, actions, dL_dQa.unsqueeze(3) # "=" assign the value
)
                # --- blame and importance ---
                # the sub-heads judge step t at step t+1 (slot 1 onwards): there the agent's hidden state h_{t+1}
                # has the step's outcome in view.  Their labels come from the TD error and dL/dQ_a that the
                # interrupted backward pass just produced for this batch.
                b = mac_out[:, 1:, :, self.args.n_actions :]              # [ blame logit | importance logits ]
                real, blame_target = self._blame_target(td_error[..., 0].detach(), mask)
                defined, imp_target = self._importance_target(dL_dQa.detach(), mask)
                blame_loss = self._blame_loss(b[..., 0], blame_target, real)
                imp_loss = self._importance_loss(b[..., 1:], imp_target, defined)
                if t_env - self.log_stats_t >= self.args.learner_log_interval:
                    self.logger.log_stat("blame_loss", blame_loss.item(), t_env)
                    self.logger.log_stat("importance_loss", imp_loss.item(), t_env)
                    self._log_answers(b, blame_target, imp_target, real, defined, t_env)
                # resume: autograd adds the TD gradient (full_grad) and the two losses' gradient, one pass
                th.autograd.backward([mac_out, blame_loss + imp_loss], [full_grad, None])
            else:
                q_agents.backward(gradient=dL_dQa)
        else:
            loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    # ------------------------------------------------------------------------------------------------
    # Two sub-heads trained on the TD error delta = Q_tot - target and on the interrupted gradient
    # dL/dQ_a = (2/|T|) * delta * dQ_tot/dQ_a.  Every agent judges the same step from its own view, so all
    # agents share each target (batch, steps).
    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _window(x, real, n):
        """With n > 1, the sum of x over the n steps ending at each step, and whether all n of them are real; with
        n = 1, x and real unchanged.  x: (batch, steps) or (batch, steps, agents); real: (batch, steps).
        Padding is zeroed first so that nothing leaks into a window, and the first n - 1 steps of an episode have no
        full window, so they carry no label."""
        if n <= 1:
            return x, real
        x = x * (real.unsqueeze(-1) if x.dim() == 3 else real)
        c = th.cumsum(x, 1)
        s = c - th.cat([th.zeros_like(c[:, :n]), c[:, :-n]], 1)                # c_t - c_{t-n}: the sum over t-n+1 .. t
        cr = th.cumsum(real.float(), 1)
        full = (cr - th.cat([th.zeros_like(cr[:, :n]), cr[:, :-n]], 1)) == n   # n real steps in the window
        return s, full

    def _blame_target(self, delta, mask):
        """blame: did the team do worse than predicted at this step?  1 or 0, from delta (batch, steps), as delta > tau.
        With blame_window n > 1, delta is first summed over the n steps ending at the step: did the team fall short
        over that window.  blame_threshold > 0 fixes tau (standardised reward units).  Else blame_margin k > 0 sets it
        to the batch's median plus k robust standard deviations (1.4826 * MAD) of the batch's deltas, a clear miss by
        the batch's own scale.  Else tau = 0: any step worse than predicted.  Returns (steps that count, target)."""
        delta, real = self._window(delta, mask[..., 0] > 0, getattr(self.args, "blame_window", 1))
        tau = getattr(self.args, "blame_threshold", 0)
        k = getattr(self.args, "blame_margin", 0)
        assert not (tau > 0 and k > 0), "set blame_threshold or blame_margin, not both"
        if tau <= 0 and k > 0:
            d = delta[real]
            med = d.median()
            tau = med + k * 1.4826 * (d - med).abs().median()
        return real, (delta > tau).float()

    def _importance_target(self, dL_dQa, mask):
        """importance: which agent has the largest mixer weight dQ_tot/dQ_a?  Since delta is shared by the agents,
        that is the largest |dL/dQ_a|.  With importance_window n > 1, dL/dQ_a is first summed over the n steps ending
        at the step: which agent accumulated the most credit over that window.  That sum is decided mostly by the
        window's most surprising step, good or bad, so by default importance stays a one-step label even when blame
        uses a window.  Returns (steps that count = steps with a label where delta != 0, target index)."""
        g, real = self._window(dL_dQa, mask[..., 0] > 0, getattr(self.args, "importance_window", 1))
        defined = real & (g.abs().sum(-1) > 0)
        return defined, g.abs().argmax(-1)

    def _blame_loss(self, logit, target, steps):
        """BCE between the blame logits (batch, time, agent) and the shared 0/1 target over the counted steps:
        one term per agent, or with ensemble_loss one term per step on the team's mean logit."""
        if getattr(self.args, "ensemble_loss", False):
            return F.binary_cross_entropy_with_logits(logit.mean(2)[steps], target[steps])
        return F.binary_cross_entropy_with_logits(logit[steps], target.unsqueeze(2).expand_as(logit)[steps])

    def _importance_loss(self, logits, target, steps):
        """Cross-entropy between the importance logits (batch, time, agent, n_agents) and the shared target index:
        one term per agent, or with ensemble_loss one term per step on the team's mean logits."""
        n_agents = logits.size(-1)
        if getattr(self.args, "ensemble_loss", False):
            return F.cross_entropy(logits.mean(2)[steps], target[steps])
        return F.cross_entropy(logits[steps].reshape(-1, n_agents),
                               target.unsqueeze(2).expand(-1, -1, logits.size(2))[steps].reshape(-1))

    def _credit(self, batch, mac_out):
        """The TD error delta = Q_tot - target (batch, steps) and dL/dQ_a (batch, steps, agents) of a batch with
        the current networks, for labels only: the same arithmetic as train, but no parameter gradient is kept
        and nothing is learned (no optimiser step, no target update, no change to the reward statistics).
        mac_out: the agents' output on the batch from _rollout; only its Q part is read, without its graph.
        Also returns the mask (batch, steps, agents) of the real steps."""
        assert not self.args.standardise_returns, "_credit does not standardise returns"
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        if self.args.standardise_rewards:
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)  # current statistics, not updated
        rewards = rewards.expand(-1, -1, self.n_agents)

        with th.no_grad():  # the same forward passes and targets as train, just without a graph
            q_out = mac_out[..., : self.args.n_actions]
            q_agents = th.gather(q_out[:, :-1], dim=3, index=actions).squeeze(3)
            target_mac_out = self._rollout(self.target_mac, batch)[:, 1:, :, : self.args.n_actions]
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999
            if self.args.double_q:
                live = q_out.clone()
                live[avail_actions == 0] = -9999999
                target_max_qvals = th.gather(target_mac_out, 3, live[:, 1:].max(dim=3, keepdim=True)[1]).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]
            targets = rewards + self.args.gamma * (1 - terminated) * self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # the same cut as in train: only the mixer is in the graph, and only dL/dQ_a is read from it
        q_agents_cut = q_agents.detach().requires_grad_(True)
        td_error = self.mixer(q_agents_cut, batch["state"][:, :-1]) - targets
        mask = mask.expand_as(td_error)
        loss = ((td_error * mask) ** 2).sum() / mask.sum()
        dL_dQa = th.autograd.grad(loss, q_agents_cut)[0]
        return td_error[..., 0].detach(), dL_dQa, mask

    def held_out(self, batch: EpisodeBatch, t_env: int):
        """Grade the sub-heads on the runner's test episodes, which never enter the replay buffer.
        The labels are the same as in train (from the current networks) but nothing is learned.  The counts are
        logged as cm_held_out_<question>_..., next to the training counts cm_<question>_..."""
        if getattr(self.args, "n_blame_outputs", 0) == 0:
            return
        with th.no_grad():
            mac_out = self._rollout(self.mac, batch)
        delta, dL_dQa, mask = self._credit(batch, mac_out)
        b = mac_out[:, 1:, :, self.args.n_actions :]
        real, blame_target = self._blame_target(delta, mask)
        defined, imp_target = self._importance_target(dL_dQa, mask)
        with th.no_grad():
            self.logger.log_stat("held_out_blame_loss", self._blame_loss(b[..., 0], blame_target, real).item(), t_env)
            self.logger.log_stat("held_out_importance_loss", self._importance_loss(b[..., 1:], imp_target, defined).item(), t_env)
        self._log_answers(b, blame_target, imp_target, real, defined, t_env, where="held_out_")

    @staticmethod
    def _rollout(mac, batch):
        """Every agent's output at every step of the batch: (batch, time, agent, outputs)."""
        mac.init_hidden(batch.batch_size)
        return th.stack([mac.forward(batch, t=t) for t in range(batch.max_seq_length)], dim=1)

    # ------------------------------------------------------------------------------------------------
    # Logging: confusion counts of the three questions.  where = "" for training batches, "held_out_" for
    # the test episodes; the metric names become cm_<where><question>_<each|vote>_<truth>_<answer>.
    # The team's answer ("vote") is the majority vote, or with ensemble_loss the prediction from the team's
    # mean logits, which is what that loss trains.  For blame, the histogram of the probability per truth is
    # logged too (ph_...), so that the summary can grade a rare label without the 0.5 rule.
    # ------------------------------------------------------------------------------------------------
    P_BINS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def _log_answers(self, b, blame_target, imp_target, real, defined, t_env, where=""):
        """b (batch, time, agent, [ blame logit | importance logits ]) and the targets from _blame_target and
        _importance_target.  Turns the logits into probabilities and counts each question's answers."""
        with th.no_grad():
            p_blame, p_imp = th.sigmoid(b[..., 0]), th.softmax(b[..., 1:], dim=-1)
            team = None
            if getattr(self.args, "ensemble_loss", False):               # the team's probabilities from its mean logits
                team = th.sigmoid(b[..., 0].mean(2)), th.softmax(b[..., 1:].mean(2), dim=-1)
            self._log_blame(p_blame, blame_target, real, t_env, where, team)
            self._log_importance(p_imp, imp_target, defined, t_env, where, team)
            self._log_combined(p_blame, p_imp, blame_target, imp_target, real, t_env, where, team)

    def _log_blame(self, p_blame, target, steps, t_env, where="", team=None):
        """p_blame (batch, time, agent) = P(someone is to blame). Answer: 1 if P > 0.5.  The team's probability for
        the histogram is the ensemble's, or else the mean of the agents'."""
        p_team = team[0] if team is not None else p_blame.mean(2)
        vote = (p_team > 0.5).long() if team is not None else None
        self._log_counts(where + "blame", target.long(), (p_blame > 0.5).long(), th.stack([1 - p_blame, p_blame], -1), steps, t_env, vote)
        self._log_hist(where + "blame", target.long(), p_blame, p_team, steps, t_env)

    def _log_importance(self, p_imp, target, steps, t_env, where="", team=None):
        """p_imp (batch, time, agent, n_agents) = P(each agent is the most important). Answer: the most likely."""
        vote = team[1].argmax(-1) if team is not None else None
        self._log_counts(where + "importance", target, p_imp.argmax(-1), p_imp, steps, t_env, vote)

    def _log_combined(self, p_blame, p_imp, blame_target, imp_target, steps, t_env, where="", team=None):
        """The final answer: the most important agent if blame, else "nobody" (index n_agents).
        Its probabilities multiply: P(agent k) = P(blame) * P(k | importance), P(nobody) = 1 - P(blame).
        The truth is the same rule applied to the targets, i.e. the old n + 1 blame target."""
        nobody = p_imp.size(-1)
        answer = th.where(p_blame > 0.5, p_imp.argmax(-1), nobody)
        probs = th.cat([p_blame.unsqueeze(-1) * p_imp, (1 - p_blame).unsqueeze(-1)], dim=-1)
        truth = th.where(blame_target == 1, imp_target, nobody)
        vote = th.where(team[0] > 0.5, team[1].argmax(-1), nobody) if team is not None else None
        self._log_counts(where + "combined", truth, answer, probs, steps, t_env, vote)

    def _log_hist(self, name, truth, p_each, p_team, steps, t_env):
        """Histogram of the blame probability per truth: ph_<name>_<each|vote>_<truth>_<bin>, bin edges P_BINS (finer
        near 0, where a rare label lives).  The summary turns it into an AUC and into the sensitivity and specificity
        at a threshold matched to the label's rate, which the 0.5 rule cannot show for a rare label."""
        nb = len(self.P_BINS) - 1
        edges = th.tensor(self.P_BINS[1:-1], device=p_each.device)
        t_each = truth.unsqueeze(2).expand_as(p_each)[steps]
        for who, p, tru in (("each", p_each[steps], t_each), ("vote", p_team[steps], truth[steps])):
            counts = th.bincount(tru.reshape(-1) * nb + th.bucketize(p.reshape(-1), edges), minlength=2 * nb)
            for i in range(2):
                for j in range(nb):
                    self.logger.log_stat(f"ph_{name}_{who}_{i}_{j}", counts[i * nb + j].item(), t_env)

    def _log_counts(self, name, truth, answer, probs, steps, t_env, vote=None):
        """Confusion counts of each agent's answer and of the team vote: cm_<name>_<each|vote>_<truth>_<answer>.
        truth (batch, time); answer (batch, time, agent); probs (batch, time, agent, k); steps (batch, time);
        vote (batch, time): the team's answer, by default the majority vote with ties broken by summed probabilities."""
        k = probs.size(-1)
        if vote is None:
            vote = self._vote(answer, probs, k)
        t_each = truth.unsqueeze(2).expand_as(answer)[steps]              # one truth per (step, agent)
        for who, ans, tru in (("each", answer[steps], t_each), ("vote", vote[steps], truth[steps])):
            counts = th.bincount(tru.reshape(-1) * k + ans.reshape(-1), minlength=k * k)   # row = truth, col = answer
            for i in range(k):
                for j in range(k):
                    self.logger.log_stat(f"cm_{name}_{who}_{i}_{j}", counts[i * k + j].item(), t_env)

    @staticmethod
    def _vote(answers, probs, k):
        """Team vote per step: the answer most agents gave; ties broken by the agents' summed probabilities.
        answers: (batch, time, agent); probs: (batch, time, agent, k)."""
        votes = F.one_hot(answers, k).sum(2)                               # (batch, time, k)
        tied = votes == votes.max(-1, keepdim=True).values
        return th.where(tied, probs.sum(2), -1.0).argmax(-1)               # (batch, time)

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
            # TEMP (not committed): start the target mixer from the loaded mixer, like target_mac above,
            # instead of leaving it at its fresh initial weights after a resume
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
