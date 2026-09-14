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
        # Stage 0: [ Q (n_actions) | B (n_blame_outputs) ]
        self.n_blame_outputs = getattr(args, "n_blame_outputs", 0)
        n_out = args.n_actions + self.n_blame_outputs

        if self.n_blame_outputs > 0 and getattr(args, "blame_head_rng_match", True):
            """ 
            P = where fc2 starts (after fc1 and rnn are built)
            STEP 1   bookmark P
            STEP 2   draw 650  →  now at P + 650     B rows get values
            STEP 3   rewind    →  back at P          the 650 are "un-spent"
            STEP 4   draw 390  →  now at P + 390     exactly the baseline's draw
            END      at P + 390  ←  where the baseline would be
            """
            rng_state = th.get_rng_state()
            self.fc2 = nn.Linear(args.hidden_dim, n_out)
            th.set_rng_state(rng_state)
            ref = nn.Linear(args.hidden_dim, args.n_actions)
            with th.no_grad():
                self.fc2.weight[: args.n_actions] = ref.weight
                self.fc2.bias[: args.n_actions] = ref.bias
        else:
            self.fc2 = nn.Linear(args.hidden_dim, n_out)

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
        if self.n_blame_outputs > 0:
            # fix float32 rounding 
            n = self.args.n_actions
            q = F.linear(h, self.fc2.weight[:n], self.fc2.bias[:n])
            b = F.linear(h, self.fc2.weight[n:], self.fc2.bias[n:])
            q = th.cat([q, b], dim=-1)
        else:
            q = self.fc2(h)
        return q, h

