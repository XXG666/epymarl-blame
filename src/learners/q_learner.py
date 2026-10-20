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

        # --- Stage 0: split the combined [ Q | B ] output ---
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
            target_mac_out = target_mac_out[..., :-n_blame]  # Stage 0: Q half only

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
                # --- Stage 0: build the expanded gradient explicitly ---
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
                # --- Stage 1: who is to blame, read from the interrupted signal ---
                blame_target = self._blame_target(dL_dQa.detach())
                # the head judges step t AFTER seeing its outcome: its output at step t+1 (mac_out[:, 1:]) is trained on step t's blame
                blame_loss = self._blame_loss(mac_out[:, 1:, :, self.args.n_actions :], blame_target, mask, t_env)
                # resume: autograd adds the TD gradient (full_grad) and the blame loss's gradient, one pass
                th.autograd.backward([mac_out, blame_loss], [full_grad, None])
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

    def _blame_target(self, dL_dQa):
        """
        Stage 1 target, one choice among n + 1 per step: [agent 1, ..., agent n, nobody].
        If the team did worse than predicted (Q_tot > y, so dL/dQ_a > 0), the agent with the
        largest dL/dQ_a gets the 1; otherwise "nobody" gets it (also at padded steps).
        """
        agents = F.one_hot(dL_dQa.argmax(-1), dL_dQa.size(-1)).float() * (dL_dQa > 0)   # (batch, time, n_agents)
        nobody = 1 - agents.sum(-1, keepdim=True)                                          # (batch, time, 1)
        return th.cat([agents, nobody], dim=-1)                                            # (batch, time, n_agents + 1)

    def _blame_loss(self, b, blame_target, mask, t_env):
        """Stage 1: one choice among n + 1 (each agent, or nobody): cross-entropy between B (logits) and the one-hot target, over real steps."""
        real = mask[..., 0] > 0                                  # (batch, time): True = real step
        target = blame_target.unsqueeze(2).expand_as(b)          # every agent guesses the same one-hot vector
        n_choices = b.size(-1)                                   # n agents + "nobody"
        blame_loss = F.cross_entropy(b[real].reshape(-1, n_choices), target[real].reshape(-1, n_choices))

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            with th.no_grad():
                p = th.softmax(b[real], dim=-1)                   # (real steps, guessing agent, n_choices): probabilities
                guess = p.argmax(-1)                              # (real steps, guessing agent): each agent's choice
                truth = blame_target[real].argmax(-1)             # (real steps,): the correct choice
                nobody = n_choices - 1                            # the last choice is "nobody"

                # how often each answer occurs in this batch (for the lazy loss)
                freq = th.bincount(truth, minlength=n_choices).float() / truth.numel()

                # team vote: the choice with the most votes; ties broken by the agents' summed probabilities
                votes = F.one_hot(guess, n_choices).sum(1)                     # (real steps, n_choices)
                tied = votes == votes.max(-1, keepdim=True).values             # the choices with the most votes
                vote = th.where(tied, p.sum(1), -1.0).argmax(-1)               # among those, the most believed

                is_blame = truth != nobody
                share = is_blame.float().mean()

                log = self.logger.log_stat
                log("blame_loss", blame_loss.item(), t_env)
                log("blame_loss_const", -(freq[freq > 0] * freq[freq > 0].log()).sum().item(), t_env) # loss of a lazy head: one that knows only how often each answer occurs, and gives those same probabilities at every step.
                log("blame_step_rate", share.item(), t_env) # how often the blame happens over real steps

                # per question: accuracy (how often exactly right), and sensitivity / specificity
                # (a head that ignores the situation always has sens + spec = 1)
                # each and vote: the 4-way answer (each agent's own, and the team vote), averaged over the n + 1 answers
                for name, answer, correct in (("each", guess, truth.unsqueeze(1).expand_as(guess)), ("vote", vote, truth)):
                    sens, spec = self._sens_spec(answer, correct, n_choices)
                    log(f"blame_{name}_acc", (answer == correct).float().mean().item(), t_env)
                    log(f"blame_{name}_sens", sens, t_env)
                    log(f"blame_{name}_spec", spec, t_env)

                # whether, from each agent's actual answer: "someone" if it names an agent, else "nobody"
                says_someone = guess != nobody                                 # (real steps, guessing agent)
                at_blame = is_blame.unsqueeze(1).expand_as(says_someone)
                log("blame_whether_acc", (says_someone == at_blame).float().mean().item(), t_env)
                if is_blame.any():
                    log("blame_whether_sens", says_someone[at_blame].float().mean().item(), t_env)       # blame steps flagged
                if (~is_blame).any():
                    log("blame_whether_spec", (~says_someone[~at_blame]).float().mean().item(), t_env)   # nobody steps left alone

                # who: at blame steps, the most likely agent (ignoring "nobody"), averaged over the n agents
                if is_blame.any():
                    who = p[is_blame][..., :nobody].argmax(-1)                 # (blame steps, guessing agent)
                    blamed = truth[is_blame].unsqueeze(1).expand_as(who)
                    sens, spec = self._sens_spec(who, blamed, nobody)
                    log("blame_who_acc", (who == blamed).float().mean().item(), t_env)
                    log("blame_who_sens", sens, t_env)
                    log("blame_who_spec", spec, t_env)
        return blame_loss

    @staticmethod
    def _sens_spec(answer, correct, k):
        """Sensitivity and specificity, each averaged over the k possible answers. For answer c: sensitivity =
        when the correct answer is c, how often c was given; specificity = when it is not c, how often c was not given."""
        sens, spec = [], []
        for c in range(k):
            is_c, said_c = correct == c, answer == c
            if is_c.any():
                sens.append(said_c[is_c].float().mean().item())
            if (~is_c).any():
                spec.append((~said_c[~is_c]).float().mean().item())
        return sum(sens) / len(sens), sum(spec) / len(spec)

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
