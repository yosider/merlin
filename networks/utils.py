# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers, Chain, Variable
from networks.constants import *

def make_batch(*xs):
    """ xs:     not batched variables.
        return: list of batched variables.
    """
    return [x.reshape(1,-1) if type(x)==Variable else np.array(x, dtype=np.float32).reshape(1,-1) for x in xs]

def softmax(xs):
    """ xs: list of scalar variables. 
        return: """
    pass

def visualize_log(**logs):
    for name, array in logs.items():
        train_times = np.arange(len(array))
        steps = TRAIN_INTERVAL * train_times 
        plt.plot(steps, array)
        plt.title(name)
        plt.xlabel('steps')
        plt.savefig(LOGDIR + name + '.png')
        plt.show()
