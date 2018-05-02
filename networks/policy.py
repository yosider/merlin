# coding: utf-8
import numpy as np
from chainer import links as L
from chainer import functions as F
from chainer import Chain
from networks.deeplstm import DeepLSTM
from networks.constants import *

class Policy(Chain):
    def __init__(self):
        super(Policy, self).__init__(
            policy = DeepLSTM(Z_DIM+M_DIM*Krp, Hp_DIM),
            reader = L.Linear(Hp_DIM, Krp*(2*Z_DIM+1)),
            pi1 = L.Linear(Z_DIM+Hp_DIM+M_DIM*Krp, 200),
            pi2 = L.Linear(200, A_DIM),
        )

    def reset(self):
        self.policy.reset_state()
        self.h = np.zeros((1, Hp_DIM), dtype=np.float32)
        self.m = np.zeros((1, M_DIM*Krp), dtype=np.float32)

    def __call__(self, z):
        state = F.concat((z, self.m))
        self.h = self.policy(state)
        i = self.reader(self.h)
        k = i[:, :2*Z_DIM*Krp]
        sc = i[:, 2*Z_DIM*Krp:]
        b = F.softplus(sc, 1)
        return k, b

    def get_action(self, z, m):
        assert m.shape == (1, M_DIM * Krp)
        self.m = m
        state = F.concat((z.data, self.h, m))   # Stop gradients wrt z.
        state = F.tanh(self.pi1(state))
        log_pi = F.log_softmax(self.pi2(state))  # log_softmax may be more stable.
        a = np.random.multinomial(1, F.exp(log_pi)[0].data).astype(np.float32) # onehot
        return log_pi, a
