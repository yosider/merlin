# coding: utf-8
import numpy as np
from chainer import links as L
from chainer import functions as F
from chainer import Chain
from networks.deeplstm import DeepLSTM
from networks.constants import *

class Predictor(Chain):
    def __init__(self):
        super(Predictor, self).__init__(
            predictor = DeepLSTM(Z_DIM+A_DIM+M_DIM*Kr, H_DIM),
            reader = L.Linear(H_DIM, Kr*(2*Z_DIM+1)),
        )

    def reset(self):
        self.predictor.reset_state()

    def __call__(self, z, a, m):
        state = F.concat((z, a, m))
        h = self.predictor(state)
        i = self.reader(h)
        k = i[:, :2*Z_DIM*Kr]
        sc = i[:, 2*Z_DIM*Kr:]
        b = F.softplus(sc)
        return h, k, b
