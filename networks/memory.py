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
        self.M = np.zeros((N_mem, M_DIM), dtype=np.float32)
    def read(self, k, b):
        # k: (1, M_DIM*Kr), b: (1, Kr)
        Kr = b.shape[1]
        for i in range(Kr):
            ki = k[:, M_DIM*i:M_DIM*(i+1)]
            print(ki.reshape(-1))
            print(self.M[0])
            
            ci = [F.matmul(ki.reshape(-1), Constant(self.M[j])) \
                    / F.batch_l2_norm_squared(ki)[0] \
                    / (np.linalg.norm(self.M[j]) + EPSILON) \
                    for j in range(N_mem)]

            #print(F.softmax(ci))
            #print(Variable(self.M[0]))
            #print(ci)
            exit()

            
            

    def write(self, z):
        pass