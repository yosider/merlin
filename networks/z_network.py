# coding: utf-8
import numpy as np
from chainer import links as L
from chainer import functions as F
from chainer import Chain
from networks.constants import *

class Z_network(Chain):
    def __init__(self):
        super(Z_network, self).__init__(
            o_encoder1 = L.Linear(O_DIM, 12),
            o_encoder2 = L.Linear(12, 6),
            prior1 = L.Linear(H_DIM+M_DIM*Kr, 2*Z_DIM),
            prior2 = L.Linear(2*Z_DIM, 2*Z_DIM),
            prior3 = L.Linear(2*Z_DIM, 2*Z_DIM),
            f_post1 = L.Linear(6+A_DIM+1+H_DIM+M_DIM*Kr+2*Z_DIM, 2*Z_DIM),
            f_post2 = L.Linear(2*Z_DIM, 2*Z_DIM),
            f_post3 = L.Linear(2*Z_DIM, 2*Z_DIM),
        )

    def __call__(self, o, a, r, h, m):
        o_encode = F.relu(self.o_encoder1(o))
        o_encode = F.relu(self.o_encoder2(o_encode))
        e = F.concat((o_encode, a, r))

        state = F.concat((h, m))
        prior = F.tanh(self.prior1(state))
        prior = F.tanh(self.prior2(prior))
        prior = self.prior3(prior)

        n = F.concat((e, h, m, prior))
        f_post = F.tanh(self.f_post1(n))
        f_post = F.tanh(self.f_post2(f_post))
        f_post = self.f_post3(f_post)
        posterior = prior + f_post

        gaussian = np.random.normal(size=(1,Z_DIM)).astype(np.float32)
        z = posterior[:, :Z_DIM] + F.exp(posterior[:, Z_DIM:]) * gaussian
        return z
