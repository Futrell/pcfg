import random
import operator

from collections import namedtuple

import torch
import einops
import tqdm
import rfutils

INF = float('inf')
EPSILON = 2. ** -10
VERY_NEGATIVE = -2.**50

# grammar is a matrix of terminal rules: LHS x RHS
# and a matrix of nonterminal rules: LHS x RHS x RHS

def identity(x):
    return x

Semiring = namedtuple("Semiring", "add mul sum prod matmul zero one to_log to_exp softmax".split())

PSPACE = Semiring(operator.add, operator.mul, torch.sum, torch.prod, torch.matmul, 0, 1, torch.log, identity, torch.softmax)
LOGSPACE = Semiring(None, operator.add, torch.logsumexp, torch.sum, None, VERY_NEGATIVE, 0, identity, torch.exp, torch.log_softmax)

class PCFG:
    def __init__(self, nonterminal_matrix, terminal_matrix, semiring=PSPACE):
        *batch1, N, T = terminal_matrix.shape
        *batch2, A, B, C = nonterminal_matrix.shape
        assert N == A == B == C
        assert batch1 == batch2
        self.nonterminal_matrix = nonterminal_matrix
        self.terminal_matrix = terminal_matrix
        self.semiring = semiring


    @classmethod
    def from_rules(self, rules, ps, semiring=PSPACE):
        rules = list(rules)
        nonterminals = set()
        terminals = set()
        for lhs, rhs in rules:
            nonterminals.add(lhs)
            if len(rhs) == 1:
                terminals.add(rhs[0])
            elif len(rhs) == 2:
                nonterminals.update(rhs)
            else:
                raise ValueError("Grammar must be binary")
            
        T = len(terminals)
        N = len(nonterminals)
        nt_matrix = torch.ones(N, N, N) * semiring.zero
        t_matrix = torch.ones(N, T) * semiring.zero

        nt_indices = {nt:i for i, nt in enumerate(nonterminals)}
        t_indices = {t:i for i, t in enumerate(terminals)}
        for (lhs, rhs), p in zip(rules, ps):
            lhs_index = nt_indices[lhs]
            if len(rhs) == 1:
                t_matrix[lhs_index, t_indices[rhs[0]]] = p
            else:
                a, b = rhs
                nt_matrix[lhs_index, nt_indices[a], nt_indices[b]] = p
        return cls(nt_matrix, t_matrix, semiring=semiring), nt_indices, t_indices

    @property
    def num_nonterminals(self):
        return self.terminal_matrix.shape[-2]

    def inside(self, sentence):
        """ Probability to generate the sentence """
        n = len(sentence)
        chart = torch.ones(n, n, self.num_nonterminals) * self.semiring.zero # shape N x N x G
        chart[0, :, :] = self.terminal_matrix[:, sentence].T
        for start, length in inclusive_spans(n):
            #charts = chart.repeat(length, 1, 1, 1)
            span_vals = torch.stack([
                span_val(self.nonterminal_matrix, chart, length, split, start, semiring=self.semiring)
                for split in range(length)
            ])
            chart = chart.clone()
            chart[length, start, :] = self.semiring.sum(span_vals, dim=0)
        return chart[-1,0,0] # longest length; first word; nonterminal 0

def span_val(nt_matrix, chart, length, split, start, semiring=PSPACE):
    B = chart[split, start, :] # shape NT
    C = chart[length - split - 1, start + split + 1, :] # shape NT
    outer = semiring.mul(B.unsqueeze(-1), C.unsqueeze(-2)) # shape NT x NT
    probs = semiring.mul(nt_matrix, outer.unsqueeze(-3)) # shape NT x NT x NT
    return semiring.sum(probs, dim=(-1, -2)) # shape NT

def inclusive_spans(n):
    assert n >= 0
    for length in range(1, n):
        for start in range(n):
            if start + length < n:
                yield start, length

def test_inclusive_spans():
    assert list(inclusive_spans(2)) == [(0,1)]
    assert list(inclusive_spans(3)) == [(0, 1), (1, 1), (0, 2)]
    assert len(list(inclusive_spans(4))) == 6
    assert list(inclusive_spans(1)) == []

def is_close(x, y, eps=10**-7):
    return abs(x-y) < eps

def test_pcfg_score():
    # rules
    # S -> SS p
    # S -> AB 1-p
    # A -> a 1
    # B -> b 1
    terminal_rules = torch.Tensor([
        #a  b
        [0, 0], # S -> a 0, S -> b 0
        [1, 0], # A -> a 1, B -> b 0
        [0, 1], # B -> a 0, B -> b 1
    ])
    zero_block = torch.zeros(3,3)

    for p in [.2, .5, .9]:
        nonterminal_rules = torch.stack([
            torch.Tensor(
                [[p, 0, 0], # S -> SS, SA, SB
                [0, 0, 1-p], # S -> AS, AA, AB
                [0, 0, 0]]), # S -> BS, BA, BB
            zero_block, zero_block])

        g = PCFG(nonterminal_rules, terminal_rules)
        
        assert is_close(g.inside((0, 1, 0, 1)).item(), p*(1-p)*(1-p))
        assert is_close(g.inside((0, 1)).item(), 1-p)
        assert is_close(g.inside((0, 1, 0, 1, 0, 1)).item(), 2*p**2*(1-p)**3)

    # rules
    # S -> AB 1-p
    # S -> AX p
    # X -> SB 1
    terminal_rules = torch.Tensor([
        [0,0], 
        [0,0], 
        [1,0],
        [0,1]
    ])
    zero_block = torch.zeros(4,4)
    for p in [.2, .5, .9]:
        nonterminal_rules = torch.Tensor([
            [[0, 0, 0, 0], # S -> SS, SX, SA, SB
            [0, 0, 0, 0], # S -> XS, XX, XA, XB
            [0, p, 0, 1-p], # S -> AS, AX, AA, AB
            [0, 0, 0, 0]],
            [[0, 0, 0, 1], # X -> SS, SX, SA, SB
            [0, 0, 0, 0], # X -> XS, XX, XA, XB
            [0, 0, 0, 0], # X -> BS, BX, BA, BB
            [0, 0, 0, 0]],         
            zero_block, zero_block])

        g = PCFG(nonterminal_rules, terminal_rules)
        assert is_close(g.inside((0, 1)), 1-p)
        assert is_close(g.inside((0, 0, 1, 1)), p * (1-p))
        assert is_close(g.inside((0, 0, 0, 1, 1, 1)), p**2 * (1-p))

    nonterminal_rules = torch.Tensor([
        [[0., 0., 0.], 
         [0., 0., 0.], # S -> BS
         [1., 0., 0.]],

        [[0., 0., 1.],
         [0., 0., 0.], # A -> SB
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.], # B -> SA
         [0., 1., 0.]]])

    terminal_rules = torch.Tensor([
        [0, 1],
        [0, 1],
        [1, 0],
    ])

    # for example,
    # S -> BS -> SAS -> BSAS -> BSSBS -> SASSSAS -> ... diverges.

    # Current setup: Rules N -> N N and N -> T
    # Some N's are preterminal, others are not.
    # need "preterminals"?

    # N -> {N,P} x {N,P}
    # P -> distro on T
    # If something appears on the left hand side of NT,
    # then it does not appear on the left hand side of T.
    # so NT has shape N x (N+P) x (N+P)
    # and T has shape P x T
    # Why not N x (N+T) x (N+T)?
    # Or, just choose P subset of N to blank out...
        

def softmax2(x, semiring=PSPACE):
    *initials, a, b = x.shape
    coalesced = initials + [a*b]
    return semiring.softmax(x.reshape(coalesced), -1).reshape(x.shape)

def example_gradient_descent_pcfg():
    data = [(0, 1), (0, 0, 1, 1), (0, 0, 0, 1, 1, 1), (0, 0, 0, 0, 1, 1, 1, 1)]
    pcfg = gradient_descent_pcfg(data, 4, 2, 2)
    return pcfg

def gradient_descent_pcfg(data, num_nt, num_t, num_pt, init_temperature=1, device=None, num_epochs=1000, batch_size=None, print_every=10, semiring=PSPACE, **kwds):
    assert 0 < num_pt < num_nt
    
    NT_energy = init_temperature * torch.randn(num_nt, num_nt, num_nt)
    T_energy = init_temperature * torch.randn(num_nt, num_t)

    NT_mask = torch.ones(num_nt, num_nt, num_nt) * semiring.one
    NT_mask[-num_pt:, :, :] = semiring.zero
    T_mask = torch.ones(num_nt, num_t) * semiring.one
    T_mask[:num_pt, :] = semiring.zero

    NT_energy = NT_energy.detach().to(device).requires_grad_(True)
    T_energy = T_energy.detach().to(device).requires_grad_(True)

    opt = torch.optim.Adam(params=[NT_energy, T_energy], **kwds)
    for i in range(num_epochs):
        opt.zero_grad()
        
        # Initialize PCFG
        NT = semiring.mul(softmax2(NT_energy, semiring=semiring), NT_mask)
        T = semiring.mul(semiring.softmax(T_energy, -1), T_mask)
        pcfg = PCFG(NT, T, semiring=semiring)
        

        # Get training data
        if batch_size is None:
            sample = data
        else:
            sample = random.sample(data, batch_size)

        # Calculate NLL
        nll = -sum(semiring.to_log(pcfg.inside(sentence)) for sentence in sample)
        loss = nll
        loss.backward()
        opt.step()

        if print_every is not None and i % print_every == 0:
            print(loss.item())

    return PCFG(NT.detach(), T.detach())

if __name__ == '__main__':
    import nose
    nose.runmodule()
    
    
