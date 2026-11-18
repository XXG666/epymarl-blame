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

        # --- split the combined [ Q | whether | who ] output ---
        # The Q part is the first n_actions outputs.  Without blame heads the output is the Q part alone,
        # and the slice below keeps all of it.
        n_actions = self.args.n_actions
        n_blame_outputs = self.args.n_blame_outputs
        q_out = mac_out[:, :, :, :n_actions]  # (batch, steps + 1, agents, n_actions)

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
        target_mac_out = target_mac_out[:, :, :, :n_actions]  # Q part only

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

        # Mix
        # Keep a pointer on the per-agent Q-values BEFORE the mixer, because the mixer call below rebinds
        # `chosen_action_qvals` to Q_tot.
        q_agents = chosen_action_qvals  # (batch, steps, agents)
        q_agents_cut = None
        if self.mixer is not None:
            if self.args.split_backward:
                # THE CUT: a fresh leaf that will collect dL/dQ_a.
                # This detached tensor is what is fed forward to the mixer.  The backward pass of the
                # loss runs back to this tensor and stops here, because it is detached from the agents.
                q_agents_cut = q_agents.detach()
                q_agents_cut.requires_grad_(True)
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
        # dL/dQ_tot  =  TD error × 2 / N 
        # Optimise
        self.optimiser.zero_grad()
        if q_agents_cut is not None:
            # First half of the backward pass: loss -> mixer.  It stops at the cut.
            loss.backward()
            dL_dQa = q_agents_cut.grad  # (batch, steps, agents)
            assert dL_dQa is not None, "ERROR: cut received no gradient - check the graph"

            # ------------------- INTERRUPT -------------------

            # Second half (resume): push dL/dQ_a into the agent networks.
            if n_blame_outputs > 0:
                # --- build the expanded gradient explicitly ---
                # full_grad has the shape of mac_out, (batch, steps + 1, agents, [ Q | whether | who ]).
                # It is zero everywhere except at the Q-value of the action each agent took:
                #
                #   Batch, Time, Agents, Actions
                #   full_grad[b][t][a][ actions[b][t][a][0] ]  =  dL_dQa[b][t][a]
                #             ▲  ▲  ▲    ▲
                #             │  │  │    └── comes from the INDEX
                #             └──┴──┴─────── come straight from the position
                full_grad = th.zeros_like(mac_out)
                # a view on the Q part of the steps that were acted in: writing into it fills full_grad
                full_grad_q = full_grad[:, :-1, :, :n_actions]
                full_grad_q.scatter_(3, actions, dL_dQa.unsqueeze(3))  # "=" assign the value

                # --- whether and who, read from the interrupted signal ---
                # the sub-heads judge step t at step t+1 (mac_out[:, 1:]), with the step's outcome in view
                blame_out = mac_out[:, 1:, :, n_actions:]  # (batch, steps, agents, [ whether | who ])
                whether_logit = blame_out[:, :, :, 0]  # (batch, steps, agents)
                who_logits = blame_out[:, :, :, 1:]  # (batch, steps, agents, n_agents)
                # td_error is (batch, steps, agents) with the same value for every agent, because the
                # common reward was repeated once per agent: take the first copy
                delta = td_error[:, :, 0].detach()  # (batch, steps)
                real, whether_target = self._whether_target(delta, mask)
                defined, who_target = self._who_target(dL_dQa.detach(), mask)
                whether_loss = self._whether_loss(whether_logit, whether_target, real)
                who_loss = self._who_loss(who_logits, who_target, defined)

                if t_env - self.log_stats_t >= self.args.learner_log_interval:
                    self.logger.log_stat("whether_loss", whether_loss.item(), t_env)
                    self.logger.log_stat("who_loss", who_loss.item(), t_env)
                    # prefix="" marks the counts of training batches (held_out uses "held_out_")
                    self._log_agents(blame_out, whether_target, who_target, real, defined, t_env, prefix="")

                # resume: autograd adds the TD gradient (full_grad) and the two losses' gradient, one pass
                blame_loss = whether_loss + who_loss
                th.autograd.backward([mac_out, blame_loss], [full_grad, None])
            else:
                # No blame heads: push dL/dQ_a straight into q_agents, the pointer that was kept before
                # the mixer.
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
    # Two sub-heads trained on the interrupted signal.  With delta = Q_tot - target, the TD error, and
    # |T|, the number of real steps in the batch:
    #     dL/dQ_tot = (2/|T|) * delta                    one value per step, the same for every agent
    #     dL/dQ_a   = dL/dQ_tot * dQ_tot/dQ_a            one value per step and agent
    # dQ_tot/dQ_a is the mixer weight of agent a, which is never negative.
    # whether is trained on dL/dQ_tot, who on dL/dQ_a.
    # Every agent judges the same step from its own view, so all agents share each target, which has
    # one entry per step: (batch, steps).
    # ------------------------------------------------------------------------------------------------
    def _whether_target(self, delta, mask):
        """whether: did the team do worse than predicted at this step, by more than a fixed threshold?

        delta: the TD error Q_tot - target, (batch, steps).  It is positive when the step went worse than
        predicted.  mask: (batch, steps, agents), larger than 0 on the real (not padded) steps and the
        same for every agent.

        A step is blamable when dL/dQ_tot = (2/|T|) * delta is positive and larger than a fixed threshold,
        the setting whether_threshold.  |T| is the number of real steps in the batch, so the same delta
        gives a smaller dL/dQ_tot in a batch with more real steps.

        Returns (real, target).  real: (batch, steps), True on the steps that count, the real steps.
        target: (batch, steps), 1.0 if the step is blamable, else 0.0."""
        real = (mask[:, :, 0] > 0)  # (batch, steps)
        n_real_steps = real.sum()  # |T|
        dL_dQtot = 2 * delta / n_real_steps  # (batch, steps)
        target = (dL_dQtot > self.args.whether_threshold).float()
        return real, target

    def _who_target(self, dL_dQa, mask):
        """who: which agent has the largest mixer weight dQ_tot/dQ_a?  Since delta is shared by the agents,
        that is the agent with the largest |dL/dQ_a|.

        dL_dQa: (batch, steps, agents).  mask: (batch, steps, agents), larger than 0 on the real steps.

        Returns (defined, target).  defined: (batch, steps), True on the steps that count, the real steps
        where delta is not zero (with delta = 0 every agent's gradient is 0 and there is no largest one).
        target: (batch, steps), the index of that agent."""
        magnitude = dL_dQa.abs()  # (batch, steps, agents)
        real = (mask[:, :, 0] > 0)  # (batch, steps)
        has_gradient = (magnitude.sum(dim=2) > 0)  # (batch, steps)
        defined = real & has_gradient
        target = magnitude.argmax(dim=2)
        return defined, target

    def _whether_loss(self, logit, target, counted):
        """Binary cross-entropy between every agent's whether logit and the shared 0/1 target.

        logit: (batch, steps, agents).  target: (batch, steps).  counted: (batch, steps), True on the
        steps that count.  Returns one number, the mean over the counted steps and the agents.

        The loss of the steps whose target is 1 is multiplied by the setting whether_pos_weight.  Above
        1.0 a missed 1 costs more than a wrongly given 1, so the head says 1 more often; 1.0 weighs both
        kinds of steps the same."""
        # every agent has the same target: repeat it once per agent
        target_per_agent = target.unsqueeze(2).expand_as(logit)  # (batch, steps, agents)
        # Indexing with the (batch, steps) True/False tensor keeps only the counted steps and lines them
        # up along one dimension: (batch, steps, agents) becomes (counted steps, agents).
        counted_logit = logit[counted]  # (counted steps, agents)
        counted_target = target_per_agent[counted]  # (counted steps, agents)
        # binary_cross_entropy_with_logits wants the weight as a tensor
        pos_weight = th.tensor(self.args.whether_pos_weight, device=counted_logit.device)
        return F.binary_cross_entropy_with_logits(counted_logit, counted_target, pos_weight=pos_weight)

    def _who_loss(self, logits, target, counted):
        """Cross-entropy between every agent's who logits and the shared target.

        logits: (batch, steps, agents, n_agents).  target: (batch, steps), the index of the right agent.
        counted: (batch, steps), True on the steps that count.
        Returns one number, the mean over the counted steps and the agents."""
        n_answerers = logits.size(2)
        n_choices = logits.size(3)
        # every agent has the same target: repeat it once per agent (-1 keeps a dimension as it is)
        target_per_agent = target.unsqueeze(2).expand(-1, -1, n_answerers)  # (batch, steps, agents)
        # keep the counted steps only, as in _whether_loss
        counted_logits = logits[counted]  # (counted steps, agents, n_agents)
        counted_target = target_per_agent[counted]  # (counted steps, agents)
        # cross_entropy wants one row of logits per answer: one row per (counted step, agent).
        # -1 lets PyTorch work out the number of rows.
        logit_rows = counted_logits.reshape(-1, n_choices)  # (counted steps * agents, n_agents)
        target_rows = counted_target.reshape(-1)  # (counted steps * agents,)
        # Write every target as a vector: 1.0 at the right agent and 0.0 at the others.
        # With 3 agents the index 2 becomes (0, 0, 1).
        target_vectors = F.one_hot(target_rows, n_choices).float()  # (counted steps * agents, n_agents)
        return F.cross_entropy(logit_rows, target_vectors)

    # ------------------------------------------------------------------------------------------------
    # Held-out grading: the same labels and counts on the runner's test episodes, which never enter the
    # replay buffer, so the agents are never trained on them.
    # ------------------------------------------------------------------------------------------------
    def held_out(self, batch: EpisodeBatch, t_env: int):
        """Grade the sub-heads on test episodes.

        batch: the test episodes.  t_env: the environment step the counts are logged at.

        Nothing is learned: no optimiser step, no target update, no change to the reward statistics.
        The counts are logged as cm_held_out_<question>_..., next to the training counts
        cm_<question>_..."""
        # Only where train trains the heads: blame heads present, a mixer, and the cut in the backward
        # pass.  Runs with standardised returns are left out as well, because _credit does not handle them.
        if self.args.n_blame_outputs == 0:
            return
        if self.mixer is None:
            return
        if not self.args.split_backward:
            return
        if self.args.standardise_returns:
            return

        with th.no_grad():
            mac_out = self._rollout(self.mac, batch)
        delta, dL_dQa, mask = self._credit(batch, mac_out)
        real, whether_target = self._whether_target(delta, mask)
        defined, who_target = self._who_target(dL_dQa, mask)

        # the sub-heads judge step t at step t+1, as in train
        blame_out = mac_out[:, 1:, :, self.args.n_actions:]  # (batch, steps, agents, [ whether | who ])
        self._log_agents(blame_out, whether_target, who_target, real, defined, t_env, prefix="held_out_")

    def _credit(self, batch, mac_out):
        """The TD error and dL/dQ_a of a batch with the current networks, for labels only: the same
        arithmetic as train, but no parameter gradient is kept and nothing is updated.

        batch: the episodes.  mac_out: the agents' output on the batch, from _rollout,
        (batch, steps + 1, agents, outputs).  Needs a mixer and no return standardisation, which held_out
        checks.

        Returns (delta, dL_dQa, mask).  delta: the TD error, (batch, steps).  dL_dQa: (batch, steps, agents).
        mask: (batch, steps, agents), 1.0 on the real steps and 0.0 on the padded ones."""
        # The batch holds steps + 1 entries per episode; [:, :-1] takes the steps that were acted in.
        rewards = batch["reward"][:, :-1]  # (batch, steps, 1): one common reward per step
        actions = batch["actions"][:, :-1]  # (batch, steps, agents, 1)
        terminated = batch["terminated"][:, :-1].float()  # (batch, steps, 1)
        mask = batch["filled"][:, :-1].float()  # (batch, steps, 1): 1.0 on real steps, 0.0 on padding
        # a step that comes after the episode has ended does not count either
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]  # (batch, steps + 1, agents, n_actions)
        if self.args.standardise_rewards:
            # with the current statistics, which are not updated here
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        # repeat the common reward once per agent (-1 keeps a dimension as it is): (batch, steps, agents)
        rewards = rewards.expand(-1, -1, self.n_agents)

        n_actions = self.args.n_actions
        # the same forward passes and targets as train, just without a graph
        with th.no_grad():
            q_out = mac_out[:, :, :, :n_actions]  # (batch, steps + 1, agents, n_actions)
            q_agents = th.gather(q_out[:, :-1], dim=3, index=actions).squeeze(3)  # (batch, steps, agents)

            target_mac_out = self._rollout(self.target_mac, batch)
            target_q_out = target_mac_out[:, 1:, :, :n_actions]  # (batch, steps, agents, n_actions)
            # unavailable actions get a very low value, so the max below never picks them
            target_q_out[avail_actions[:, 1:] == 0] = -9999999
            if self.args.double_q:
                # the live networks choose the next action, the target networks give its value
                live_q_out = q_out.clone()
                live_q_out[avail_actions == 0] = -9999999
                best_actions = live_q_out[:, 1:].max(dim=3, keepdim=True).indices
                target_max_qvals = th.gather(target_q_out, 3, best_actions).squeeze(3)
            else:
                target_max_qvals = target_q_out.max(dim=3).values
            target_q_tot = self.target_mixer(target_max_qvals, batch["state"][:, 1:])  # (batch, steps, 1)
            targets = rewards + self.args.gamma * (1 - terminated) * target_q_tot

        # the same cut as in train: only the mixer is in the graph, and only dL/dQ_a is read from it
        q_agents_cut = q_agents.detach()
        q_agents_cut.requires_grad_(True)
        q_tot = self.mixer(q_agents_cut, batch["state"][:, :-1])  # (batch, steps, 1)
        # (batch, steps, agents), the same value for every agent: the reward was repeated once per agent
        td_error = q_tot - targets
        # repeat the mask once per agent as well: (batch, steps, 1) becomes (batch, steps, agents)
        mask = mask.expand_as(td_error)
        loss = ((td_error * mask) ** 2).sum() / mask.sum()
        gradients = th.autograd.grad(loss, q_agents_cut)  # a tuple with one gradient per input
        dL_dQa = gradients[0]
        delta = td_error[:, :, 0].detach()  # (batch, steps): the first of the equal copies
        return delta, dL_dQa, mask

    def _rollout(self, mac, batch):
        """Run a controller over every step of the batch.

        mac: the controller (the live one or the target one).  batch: the episodes.

        Returns every agent's output at every step, (batch, steps + 1, agents, outputs)."""
        outputs = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            outputs.append(mac.forward(batch, t=t))
        return th.stack(outputs, dim=1)

    # ------------------------------------------------------------------------------------------------
    # Logging: confusion counts of the three questions, whether, who and combined.
    # prefix = "" for training batches, "held_out_" for the test episodes.  The metric names are
    # cm_<prefix><whether|who|combined>_<each|vote>_<truth>_<answer>.
    # ------------------------------------------------------------------------------------------------
    def _log_agents(self, blame_out, whether_target, who_target, real, defined, t_env, prefix):
        """The agents' answers to the three questions.  Turns the logits into probabilities and counts each
        question's answers, per agent ("each") and as a team vote ("vote").

        blame_out: the agents' sub-head outputs, (batch, steps, agents, [ whether logit | who logits ]).
        whether_target, who_target, real, defined: (batch, steps), from _whether_target and _who_target.
        t_env: the environment step to log at.  prefix: "" or "held_out_", put in front of the question
        in the metric names."""
        with th.no_grad():
            p_whether = th.sigmoid(blame_out[:, :, :, 0])  # (batch, steps, agents)
            p_who = th.softmax(blame_out[:, :, :, 1:], dim=3)  # (batch, steps, agents, n_agents)
            self._log_whether(p_whether, whether_target, real, t_env, prefix)
            self._log_who(p_who, who_target, defined, t_env, prefix)
            self._log_combined(p_whether, p_who, whether_target, who_target, real, t_env, prefix)

    def _log_whether(self, p_whether, target, counted, t_env, prefix):
        """Count the agents' answers to whether.  An agent answers 1 (yes) if its probability is above 0.5.

        p_whether: (batch, steps, agents), P(the step is blamable, see _whether_target).
        target: (batch, steps), 1.0 or 0.0.  counted: (batch, steps), True on the steps that count.
        t_env, prefix: as in _log_agents."""
        truth = target.long()
        answer = (p_whether > 0.5).long()
        # the probability of each of the two answers, 0 (no) and 1 (yes)
        probs = th.stack([1 - p_whether, p_whether], dim=3)  # (batch, steps, agents, 2)
        self._log_each_and_vote(prefix + "whether", truth, answer, probs, counted, t_env)

    def _log_who(self, p_who, target, counted, t_env, prefix):
        """Count the agents' answers to who.  An agent answers with the agent it finds most likely.

        p_who: (batch, steps, agents, n_agents), P(each agent is the most important).
        target: (batch, steps), an agent index.  counted: (batch, steps), True on the steps that count.
        t_env, prefix: as in _log_agents."""
        answer = p_who.argmax(dim=3)  # (batch, steps, agents)
        self._log_each_and_vote(prefix + "who", target, answer, p_who, counted, t_env)

    def _log_combined(self, p_whether, p_who, whether_target, who_target, counted, t_env, prefix):
        """Count the agents' final answers: the agent named by who if whether says yes, else "nobody"
        (index n_agents).  The truth is the same rule applied to the two targets.

        p_whether: (batch, steps, agents).  p_who: (batch, steps, agents, n_agents).
        whether_target, who_target: (batch, steps).  counted: (batch, steps), True on the steps that
        count.  t_env, prefix: as in _log_agents."""
        n_agents = p_who.size(3)
        nobody = n_agents  # the extra answer "nobody" gets the index after the last agent
        says_yes = (p_whether > 0.5)  # (batch, steps, agents)
        named_agent = p_who.argmax(dim=3)  # (batch, steps, agents)
        # th.where(condition, a, b) takes a where the condition is True and b elsewhere
        answer = th.where(says_yes, named_agent, nobody)  # (batch, steps, agents)
        # The probabilities multiply: P(agent k) = P(whether) * P(k | who), P(nobody) = 1 - P(whether).
        p_each_agent = p_whether.unsqueeze(3) * p_who  # (batch, steps, agents, n_agents)
        p_nobody = (1 - p_whether).unsqueeze(3)  # (batch, steps, agents, 1)
        probs = th.cat([p_each_agent, p_nobody], dim=3)  # (batch, steps, agents, n_agents + 1)
        truth = th.where(whether_target == 1, who_target, nobody)  # (batch, steps)
        self._log_each_and_vote(prefix + "combined", truth, answer, probs, counted, t_env)

    def _log_each_and_vote(self, question, truth, answer, probs, counted, t_env):
        """Confusion counts of each agent's answer and of the team vote, logged as
        cm_<question>_each_<truth>_<answer> and cm_<question>_vote_<truth>_<answer>.

        question: the question's name with its prefix, for example "held_out_who".
        truth: (batch, steps).  answer: (batch, steps, agents).  probs: (batch, steps, agents, n_answers),
        the probability an agent gives to each possible answer.  counted: (batch, steps), True on the
        steps that count.  t_env: the environment step to log at."""
        n_answers = probs.size(3)
        vote = self._vote(answer, probs, n_answers)  # (batch, steps)
        # every agent is graded against the same truth: repeat it once per agent
        truth_per_agent = truth.unsqueeze(2).expand_as(answer)  # (batch, steps, agents)
        # keep the counted steps only, as in _whether_loss
        each_truth = truth_per_agent[counted]  # (counted steps, agents)
        each_answer = answer[counted]  # (counted steps, agents)
        self._log_confusion_matrix(question, "each", each_truth, each_answer, n_answers, t_env)
        vote_truth = truth[counted]  # (counted steps,)
        vote_answer = vote[counted]  # (counted steps,)
        self._log_confusion_matrix(question, "vote", vote_truth, vote_answer, n_answers, t_env)

    def _vote(self, answers, probs, n_answers):
        """Team vote per step: the answer most agents gave.  Ties are broken by the agents' summed
        probabilities.

        answers: (batch, steps, agents), each one of 0 .. n_answers - 1.
        probs: (batch, steps, agents, n_answers).  Returns (batch, steps), the winning answer."""
        ballots = F.one_hot(answers, n_answers)  # (batch, steps, agents, n_answers): 1 at the agent's answer
        votes = ballots.sum(dim=2)  # (batch, steps, n_answers): how many agents gave each answer
        most_votes = votes.max(dim=2, keepdim=True).values  # (batch, steps, 1)
        # (batch, steps, n_answers): True for every answer with the most votes
        tied = (votes == most_votes)
        summed_probs = probs.sum(dim=2)  # (batch, steps, n_answers)
        # Among the answers with the most votes, take the one with the largest summed probability.
        # The other answers get -1, which is below any probability, so they cannot win.
        score = th.where(tied, summed_probs, -1.0)
        return score.argmax(dim=2)

    def _log_confusion_matrix(self, question, answerer, truth, answer, n_answers, t_env):
        """Log one n_answers x n_answers confusion matrix, one number per cell, as
        cm_<question>_<answerer>_<truth>_<answer>.

        question: the question's name with its prefix, for example "held_out_who".  answerer: "each"
        or "vote".  truth, answer: tensors of the same shape with whole numbers 0 .. n_answers - 1,
        one pair per graded answer.  t_env: the environment step to log at.
        Each logged number says how often that truth met that answer."""
        # Give every (truth, answer) pair the number truth * n_answers + answer and count each number.
        # reshape(-1) lays a tensor out as one long row.
        pair = truth.reshape(-1) * n_answers + answer.reshape(-1)
        counts = th.bincount(pair, minlength=n_answers * n_answers)
        for truth_index in range(n_answers):
            for answer_index in range(n_answers):
                count = counts[truth_index * n_answers + answer_index].item()
                metric_name = f"cm_{question}_{answerer}_{truth_index}_{answer_index}"
                self.logger.log_stat(metric_name, count, t_env)

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
            # start the target mixer from the loaded mixer, like target_mac above, instead of leaving
            # it at its fresh initial weights after a resume
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
