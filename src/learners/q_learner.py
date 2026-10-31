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
        q_agents = chosen_action_qvals  # (batch, time, n_agents)
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
                Batch, Time, Agents, Actions
                full_grad[b][t][a][ actions[b][t][a][0] ]  =  dL_dQa[b][t][a][0]
                          ▲  ▲  ▲    ▲
                          │  │  │    └── comes from the INDEX
                          └──┴──┴─────── come straight from the position
                """
                full_grad[:, :-1, :, : self.args.n_actions].scatter_(
                    3, actions, dL_dQa.unsqueeze(3) # "=" assign the value
)
                # --- blame and importance, read from the interrupted signal ---
                # the sub-heads judge step t at step t+1 (mac_out[:, 1:]); there the agent's heads read
                # [ h_t | h_{t+1} ], the situation when it acted and the situation with the step's outcome in view
                b = mac_out[:, 1:, :, self.args.n_actions :]              # [ blame logit | importance logits ]
                g = dL_dQa.detach()
                real, blame_target = self._blame_target(g, mask)
                defined, imp_target = self._importance_target(g, mask)
                blame_loss = self._blame_loss(b[..., 0], blame_target, real)
                imp_loss = self._importance_loss(b[..., 1:], imp_target, defined)
                if t_env - self.log_stats_t >= self.args.learner_log_interval:
                    with th.no_grad():
                        p_blame, p_imp = th.sigmoid(b[..., 0]), th.softmax(b[..., 1:], dim=-1)
                        self._log_blame(p_blame, blame_target, real, blame_loss, t_env)
                        self._log_importance(p_imp, imp_target, defined, imp_loss, t_env)
                        self._log_combined(p_blame, p_imp, blame_target, imp_target, real, t_env)
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
    # Two sub-heads trained on the interrupted gradient dL/dQ_a = (2/|T|) * delta * dQ_tot/dQ_a.
    # Every agent judges the same step from its own view, so all agents share each target (batch, time).
    # ------------------------------------------------------------------------------------------------
    def _blame_target(self, dL_dQa, mask):
        """blame: did the team do worse than predicted at this step (dL/dQ_a > 0)?  1 or 0.
        Returns (steps that count = real steps, target)."""
        real = mask[..., 0] > 0
        return real, (dL_dQa.max(-1).values > 0).float()

    def _importance_target(self, dL_dQa, mask):
        """importance: which agent has the largest mixer weight dQ_tot/dQ_a?  Since delta is shared by the agents,
        that is the largest |dL/dQ_a|.  Returns (steps that count = real steps where delta != 0, target index)."""
        defined = (mask[..., 0] > 0) & (dL_dQa.abs().sum(-1) > 0)
        return defined, dL_dQa.abs().argmax(-1)

    def _blame_loss(self, logit, target, steps):
        """BCE between the blame logit (batch, time, agent) and the shared 0/1 target, over the counted steps."""
        return F.binary_cross_entropy_with_logits(logit[steps], target.unsqueeze(2).expand_as(logit)[steps])

    def _importance_loss(self, logits, target, steps):
        """Cross-entropy between the importance logits (batch, time, agent, n_agents) and the shared target index."""
        n_agents = logits.size(-1)
        return F.cross_entropy(logits[steps].reshape(-1, n_agents),
                               target.unsqueeze(2).expand(-1, -1, logits.size(2))[steps].reshape(-1))

    def _log_blame(self, p_blame, target, steps, loss, t_env):
        """p_blame (batch, time, agent) = P(someone is to blame). Answer: 1 if P > 0.5."""
        self.logger.log_stat("blame_loss", loss.item(), t_env)
        self._log_counts("blame", target.long(), (p_blame > 0.5).long(), th.stack([1 - p_blame, p_blame], -1), steps, t_env)

    def _log_importance(self, p_imp, target, steps, loss, t_env):
        """p_imp (batch, time, agent, n_agents) = P(each agent is the most important). Answer: the most likely."""
        self.logger.log_stat("importance_loss", loss.item(), t_env)
        self._log_counts("importance", target, p_imp.argmax(-1), p_imp, steps, t_env)

    def _log_combined(self, p_blame, p_imp, blame_target, imp_target, steps, t_env):
        """The final answer: the most important agent if blame, else "nobody" (index n_agents).
        Its probabilities multiply: P(agent k) = P(blame) * P(k | importance), P(nobody) = 1 - P(blame).
        The truth is the same rule applied to the targets, i.e. the old n + 1 blame target."""
        nobody = p_imp.size(-1)
        answer = th.where(p_blame > 0.5, p_imp.argmax(-1), nobody)
        probs = th.cat([p_blame.unsqueeze(-1) * p_imp, (1 - p_blame).unsqueeze(-1)], dim=-1)
        truth = th.where(blame_target == 1, imp_target, nobody)
        self._log_counts("combined", truth, answer, probs, steps, t_env)

    def _log_counts(self, name, truth, answer, probs, steps, t_env):
        """Confusion counts of each agent's answer and of the team vote: cm_<name>_<each|vote>_<truth>_<answer>.
        truth (batch, time); answer (batch, time, agent); probs (batch, time, agent, k); steps (batch, time)."""
        k = probs.size(-1)
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
