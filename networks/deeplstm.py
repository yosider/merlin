# coding: utf-8
import numpy as np
from chainer import links as L 
from chainer import functions as F
from chainer import Chain
from networks.constants import *


class DeepLSTM(Chain): # too simple?
    def __init__(self, d_in, d_out):
        super(DeepLSTM, self).__init__(
            l1 = L.LSTM(d_in, d_out),
            l2 = L.Linear(d_out, d_out),
            )
    def __call__(self, x):
        self.x = x
        self.y = self.l2(self.l1(self.x))
        return self.y
    def reset_state(self):
        self.l1.reset_state()