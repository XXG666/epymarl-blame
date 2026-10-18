import torch as th

th.manual_seed(0)
B, T, N, A, NB = 2, 3, 4, 6, 4          # batch, time, agents, n_actions, blame width

net = th.nn.Linear(5, A + NB)           # the Q+B head
x = th.randn(B, T + 1, N, 5)
actions = th.randint(0, A, (B, T, N, 1))

def run(explicit):
    for p in net.parameters():
        p.grad = None
    out = net(x)                                    # (B, T+1, N, A+NB)
    q_out = out[..., :A]                            # THE SPLIT
    q_taken = th.gather(q_out[:, :-1], 3, actions).squeeze(3)
    cut = q_taken.detach().requires_grad_(True)     # the M2 cut
    ((cut ** 2).sum()).backward()                   # stage 1
    g = cut.grad                                    # dL/dQ_a
    if explicit:                                    # STAGE 0: build it by hand
        full = th.zeros_like(out)
        full[:, :-1, :, :A].scatter_(3, actions, g.unsqueeze(3))
        out.backward(gradient=full)                 # stage 2
    else:
        q_taken.backward(gradient=g)                # let autograd pad
    return [p.grad.clone() for p in net.parameters()]

auto, expl = run(False), run(True)
for a, b in zip(auto, expl):
    print(f"  max|diff| = {(a - b).abs().max():.3e}   shape={tuple(a.shape)}")
print("explicit == automatic       :", all(th.equal(a, b) for a, b in zip(auto, expl)))
print("B-block weight grad all zero:", bool((expl[0][A:] == 0).all()))
print("B-block bias   grad all zero:", bool((expl[1][A:] == 0).all()))
print("Q-block weight grad nonzero :", bool((expl[0][:A] != 0).any()))