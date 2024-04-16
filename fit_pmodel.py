import itertools

import torch
import opt_einsum

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PFSA:
    def __init__(self, E, T, device=DEVICE):
        self.E = E
        self.T = T
        Q, S, Q2 = self.T.shape
        Q3, S2 = self.E.shape
        assert Q == Q2 == Q3
        assert S == S2
        self.device = device
        self.init = torch.eye(Q, device=self.device)[0]

    @classmethod
    def init_logits(cls, Q=10, S=2, device=DEVICE):
        T = torch.randn(Q, S, Q, requires_grad=True, device=device)
        E = torch.randn(Q, S, requires_grad=True, device=device)
        return E, T

    @classmethod
    def from_logits(cls, E_logit, T_logit, device=DEVICE):
        T = torch.softmax(T_logit, -1)
        E = torch.softmax(E_logit, -1)
        return cls(E, T, device=device)

    def score(self, xs):
        B, T = xs.shape
        state = torch.stack([self.init for _ in range(B)])
        score = torch.zeros(B).to(self.device)
        for x in xs.T:
            y = opt_einsum.contract("qx, Bq -> Bx", self.E, state)
            score += torch.gather(y, -1, x.unsqueeze(-1)).squeeze(-1).log()
            state = opt_einsum.contract("qBr, Bq -> Br", self.T[:, x], state)
        return score

class SSM:
    def __init__(self, A, B, C, bias=None, device=DEVICE):
        self.A = A
        self.B = B
        self.C = C
        X, X2 = self.A.shape
        X3, Y = self.C.shape
        U, X4 = self.B.shape
        assert X == X2 == X3 == X4
        if bias is None:
            self.bias = torch.zeros(Y)
        else:
            self.bias = bias
        self.device = device
        self.init = torch.eye(X, device=self.device)[0]

    @classmethod
    def init_logits(cls, X=10, S=2, device=DEVICE):
        A = torch.randn(X, X, device=device, requires_grad=True)
        B = torch.randn(S, X, device=device, requires_grad=True)
        C = torch.randn(X, S, device=device, requires_grad=True)
        bias = torch.randn(S, device=device, requires_grad=True)
        return A, B, C, bias

    @classmethod
    def from_logits(cls, A, B, C, bias, device=DEVICE):
        return cls(A, B, C, bias, device=device)

    def score(self, us):
        B, T = us.shape
        x = torch.stack([self.init for _ in range(B)])
        score = torch.zeros(B)
        for u in us.T:
            y = opt_einsum.contract("xy, Bx -> By", self.C, x) + self.bias
            score += torch.gather(torch.log_softmax(y, -1), -1, u.unsqueeze(-1)).squeeze(-1)
            x = opt_einsum.contract("xy, Bx -> By", self.A, x) + self.B[u]
        return score

class PCFG:
    """ PCFG in Chomsky Normal Form. """
    def __init__(self, R, omega, device=DEVICE):
        self.R = R
        self.omega = omega
        self.V, self.S = self.omega.shape
        self.device = device
        
        X, Y, Z = R.shape
        W, S = omega.shape
        assert X == Y == Z == W

    @classmethod
    def init_logits(cls, V=12, S=2, device=DEVICE):
        R_logit = torch.randn(V, V, V, requires_grad=True, device=device)
        omega_logit = torch.randn(V, S, requires_grad=True, device=device)
        return R_logit, omega_logit

    @classmethod
    def from_logits(cls, R_logit, omega_logit, device=DEVICE):
        lnZ = torch.logaddexp(omega_logit.logsumexp(-1), R_logit.logsumexp(dim=(-1, -2)))
        R = (R_logit - lnZ[:, None, None]).exp()
        omega = (omega_logit - lnZ[:, None]).exp()
        return cls(R, omega, device=device)

    def score(self, xs):
        B, T = xs.shape
        chart = torch.zeros(B, T, T, self.V).to(self.device)
        chart[:, 0] = self.omega[:, xs].permute(1,2,0) # batch x span size x start index x label
        for s in range(1, T): # span lengths
            t = torch.arange(T-s)
            for t in range(T-s): # start indices
                # w_A(t,s) = \sum_{k=1}^{s-1} \sum_B \sum_C w_{ABC} * w_B(t, k) * w_C(t+k+1, k-s-1)
                k = torch.arange(s) # split indices
                left = chart[:, k, t]
                right = chart[:, s-k-1, t+k+1]
                chart[:, s, t] = opt_einsum.contract("abc, Bkb, Bkc -> Ba", self.R, left, right)
        return chart[:, -1, 0, 0].log()

def bitstrings(K, V=2):
    return torch.LongTensor(list(itertools.product(range(V), repeat=K)))

def fit_model(forms,
              p,
              model_cls=PCFG,
              num_iter=10000,
              print_every=1000,
              normalize=False,
              device=DEVICE,
              lr=.1,
              **kwds):
    """ Assumes all forms are the same length. """
    params = model_cls.init_logits()
    target_entropy = -torch.xlogy(p, p).sum().item()
    opt = torch.optim.AdamW(params=params, lr=lr, **kwds)
    for i in range(num_iter):
        opt.zero_grad()
        model = model_cls.from_logits(*params, device=device)
        lnq = model.score(forms)
        if normalize:
            lnq = lnq.log_softmax(-1)
        loss = -p @ lnq
        loss.backward()
        if i % print_every == 0:
            print(loss.item() - target_entropy)
        opt.step()
    return model_cls.from_logits(*params, device=device)

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
        
        assert is_close(g.score(torch.LongTensor([[0, 1, 0, 1]])).item(), p*(1-p)*(1-p))
        assert is_close(g.score(torch.LongTensor([[0, 1]])).item(), 1-p)
        assert is_close(g.score(torch.LongTensor([[0, 1, 0, 1, 0, 1]])).item(), 2*p**2*(1-p)**3)

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
        assert is_close(g.score(torch.LongTensor([[0, 1]])), 1-p)
        assert is_close(g.score(torch.LongTensor([[0, 0, 1, 1]])), p * (1-p))
        assert is_close(g.score(torch.LongTensor([[0, 0, 0, 1, 1, 1]])), p**2 * (1-p))

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
        



    
    
    
