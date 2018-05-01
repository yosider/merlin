# coding: utf-8
import numpy as np
from chainer import links as L 
from chainer import functions as F
from chainer import Chain
from networks.constants import *


class Decoder(Chain):
    def __init__(self):
        super(Decoder, self).__init__(
            decoder1 = L.Linear(Z_DIM, 2*Z_DIM),
            decoder2 = L.Linear(2*Z_DIM, 2*Z_DIM),
            decoder3 = L.Linear(2*Z_DIM, 6+A_DIM+1+H_DIM+M_DIM+2*Z_DIM),
            decoder4 = L.Linear(6+A_DIM+1+H_DIM+M_DIM+2*Z_DIM, 6+A_DIM+1),
            o_decoder1 = L.Linear(6, 12),
            o_decoder2 = L.Linear(12, O_DIM),
            a_decoder = L.Linear(A_DIM, A_DIM),

            value1 = L.Linear(Z_DIM+A_DIM, 200),
            value2 = L.Linear(200, 1),
            advantage1 = L.Linear(Z_DIM+A_DIM, 50),
            advantage2 = L.Linear(50, 1),
        )

    def __call__(self, z, log_pi, a): #TODO
        decode = F.relu(self.decoder1(z))
        decode = F.relu(self.decoder2(decode))
        decode = F.relu(self.decoder3(decode))
        decode = self.decoder4(decode)
        o_decode = F.relu(self.o_decoder1(decode[:, :6]))
        o_decode = self.o_decoder2(o_decode)
        a_decode = self.a_decoder(decode[:, 6:6+A_DIM]) # softmax or onehoten? →loss計算時にやる
        r_decode = decode[:, -1:]

        state = F.concat((z, log_pi))
        V = F.tanh(self.value1(state))
        V = self.value2(V)

        state = F.concat((z, a))
        A = F.tanh(self.advantage1(state))
        A = F.tanh(self.advantage2(A))

        R = V.data + A  # stop gradient wrt V
        return o_decode, a_decode, r_decode, V, R

