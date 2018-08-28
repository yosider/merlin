# coding: utf-8
import unittest 
from chainer import Variable

from merlin import Merlin 
from networks.constants import *
from networks.utils import *

from networks.z_network import Z_network
from networks.decoder import Decoder
from networks.deeplstm import DeepLSTM
from networks.predictor import Predictor
from networks.policy import Policy
from networks.memory import Memory

class TestMain(unittest.TestCase):
    pass

class TestZnetwork(unittest.TestCase):
    def test_forward(self):
        net = Z_network()
        o = make_sample_input(1, O_DIM)
        a = make_sample_input(1, A_DIM)
        r = make_sample_input(1, 1)
        h = make_sample_input(1, H_DIM)
        m = make_sample_input(1, M_DIM*Kr)
        net(o, a, r, h, m)

class TestPredictor(unittest.TestCase):
    def test_forward(self):
        net = Predictor()
        z = make_sample_input(1, Z_DIM)
        a = make_sample_input(1, A_DIM)
        m = make_sample_input(1, M_DIM*Kr)
        net(z, a, m)

class TestDecoder(unittest.TestCase):
    def test_forward(self):
        net = Decoder()
        z = make_sample_input(1, Z_DIM)
        log_pi = make_sample_input(1, A_DIM)
        a = make_sample_input(1, A_DIM)
        net(z, log_pi, a)

class TestPolicy(unittest.TestCase):
    def test_readvec(self):
        net = Policy()
        net.reset()
        z = make_sample_input(1, Z_DIM)
        net(z)
    
    def test_get_action(self):
        net = Policy()
        net.reset()
        z = Variable(make_sample_input(1, Z_DIM).astype(np.float32))
        m = make_sample_input(1, M_DIM*Krp)
        net.get_action(z, m)

class TestMemory(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()