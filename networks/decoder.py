# coding: utf-8
import numpy as np
from chainer import links as L 
from chainer import functions as F
from chainer import Chain
from networks.constants import *


class Decoder(Chain):
    def __init__(self):
        #TODO: search optimal size
        self.o_in_size = 3*O_DIM
        self.o_hid_size = 3*O_DIM
        self.value_hid_size = 200
        self.advantage_hid_size = 50

        super(Decoder, self).__init__(
            decoder1 = L.Linear(Z_DIM, 2*Z_DIM),
            decoder2 = L.Linear(2*Z_DIM, 2*Z_DIM),
            decoder3 = L.Linear(2*Z_DIM, self.o_in_size+A_DIM+1+H_DIM+M_DIM+2*Z_DIM),
            decoder4 = L.Linear(self.o_in_size+A_DIM+1+H_DIM+M_DIM+2*Z_DIM, self.o_in_size+A_DIM+1),
            o_decoder1 = L.Linear(self.o_in_size, self.o_hid_size),
            o_decoder2 = L.Linear(self.o_hid_size, O_DIM),
            a_decoder = L.Linear(A_DIM, A_DIM),

            value1 = L.Linear(Z_DIM+A_DIM, self.value_hid_size),
            value2 = L.Linear(self.value_hid_size, 1),
            advantage1 = L.Linear(Z_DIM+A_DIM, self.advantage_hid_size),
            advantage2 = L.Linear(self.advantage_hid_size, 1),
        )

    def __call__(self, z, log_pi, a): #TODO
        decode = F.relu(self.decoder1(z))
        decode = F.relu(self.decoder2(decode))
        decode = F.relu(self.decoder3(decode))
        decode = self.decoder4(decode)
        o_decode = F.relu(self.o_decoder1(decode[:, :self.o_in_size]))
        o_decode = self.o_decoder2(o_decode)
        a_decode = self.a_decoder(decode[:, self.o_in_size:self.o_in_size+A_DIM]) # softmax or onehoten? →loss計算時にやる
        r_decode = decode[:, -1:]

        state = F.concat((z, log_pi))
        V = F.tanh(self.value1(state))
        V = self.value2(V)

        state = F.concat((z, a))
        A = F.tanh(self.advantage1(state))
        A = F.tanh(self.advantage2(A))

        R = V.data + A  # stop gradient wrt V
        return o_decode, a_decode, r_decode, V, R

