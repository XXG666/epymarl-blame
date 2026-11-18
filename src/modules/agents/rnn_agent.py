# code adapted from https://github.com/wendelinboehmer/dcg
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)  # the Q head, exactly as in plain QMIX

        # Output per step: [ Q (n_actions) | whether (1) | who (n_agents) ]
        # The two blame sub-heads read the hidden state like the Q head.  The learner trains their output
        # at step t+1 on step t, so they judge a step with its outcome in view.
        #   whether : one logit, "the team did worse than predicted at step t by more than a fixed threshold"
        #             (dL/dQ_tot is above the setting whether_threshold)
        #   who     : one logit per agent, "this agent has the largest mixer weight at step t"
        self.n_blame_outputs = args.n_blame_outputs
        if self.n_blame_outputs > 0:
            # Built after fc2 from a bookmarked RNG state, which is then restored, so that everything
            # built later (the mixer) and every exploration draw are identical to plain QMIX.
            rng_state = th.get_rng_state()
            self.whether = nn.Linear(args.hidden_dim, 1)
            self.who = nn.Linear(args.hidden_dim, args.n_agents)
            th.set_rng_state(rng_state)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        if self.n_blame_outputs > 0:
            whether_logit = self.whether(h)  # (batch * agents, 1)
            who_logits = self.who(h)  # (batch * agents, n_agents)
            # from here on q is the whole output row of an agent: [ Q | whether | who ]
            q = th.cat([q, whether_logit, who_logits], dim=1)
        return q, h

