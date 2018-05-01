# coding: utf-8
import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers, Chain, Variable

def make_batch(*xs):
    """ xs:     not batched variables.
        return: list of batched variables.
    """
    return [x.reshape(1,-1) if type(x)==Variable else np.array(x, dtype=np.float32).reshape(1,-1) for x in xs]

def softmax(xs):
    """ xs: list of scalar variables. 
        return: """
    pass