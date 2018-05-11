# coding: utf-8
import numpy as np
from chainer import links as L
from chainer import functions as F
from chainer import Chain, Variable
from networks.constants import *

class Memory(Chain):
    def __init__(self):
        super(Memory, self).__init__()
        self.size = 1
        self.M = Variable(np.zeros((N_mem, M_DIM), dtype=np.float32))
        self.W_predictor = None
        self.W_policy = None
        self.v_wr = Variable(np.zeros((N_mem, 1), dtype=np.float32))
        self.v_ret = Variable(np.zeros((N_mem, 1), dtype=np.float32))
        self.u = Variable(np.zeros((1, N_mem), dtype=np.float32))

    def read(self, k, b):
        # k: (1, M_DIM*kr), b: (1, kr)
        kr = b.shape[1]
        K = k.reshape(kr, M_DIM)
        C = F.matmul(F.normalize(K), F.transpose(F.normalize(self.M)))  # FIXME: error when F.normalize
        B = F.repeat(b, N_mem).reshape(kr, N_mem)  # beta
        if kr == Kr:
            self.W_predictor = F.softmax(B*C)  # B*C: elementwise multiplication
            M = F.matmul(self.W_predictor, self.M)
        elif kr == Krp:
            self.W_policy = F.softmax(B*C)
            M = F.matmul(self.W_policy, self.M)
        else:
            raise(ValueError)
        return M.reshape((1, -1))

    def write(self, z, time):
        # update usage indicator
        self.u += F.matmul(Variable(np.ones((1, Kr), dtype=np.float32)), self.W_predictor)

        # update writing weights
        prev_v_wr = self.v_wr
        v_wr = np.zeros((N_mem, 1), dtype=np.float32)
        if time < N_mem:
            v_wr[time][0] = 1
        else:
            waste_index = int(F.argmin(self.u).data)
            v_wr[waste_index][0] = 1
        self.v_wr = Variable(v_wr)

        # writing
        # z: (1, Z_DIM)
        if USE_RETROACTIVE:
            # update retroactive weights
            self.v_ret = GAMMA*self.v_ret + (1-GAMMA)*prev_v_wr
            z_wr = F.concat((z, Variable(np.zeros((1, Z_DIM), dtype=np.float32))))
            z_ret = F.concat((Variable(np.zeros((1, Z_DIM), dtype=np.float32)), z))
            self.M += F.matmul(self.v_wr, z_wr) + F.matmul(self.v_ret, z_ret)
        else:
            self.M += F.matmul(self.v_wr, z)
