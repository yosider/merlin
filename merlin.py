# coding: utf-8
import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers, Chain, Variable

from networks.constants import *
from networks.z_network import Z_network
from networks.decoder import Decoder
from networks.predictor import Predictor
from networks.policy import Policy
from networks.memory import Memory
from networks.utils import *

class Merlin(Chain):
    def __init__(self):
        # Merlin performs cross-module processings．
        super(Merlin, self).__init__(
            z_network = Z_network(),
            predictor = Predictor(), #TODO: hは2層？
            decoder = Decoder(),
            policy = Policy(),
            memory = Memory(),
            )
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self)
        self.loss_log = []

    def reset(self, done=True):
        # Merlin holds variables shared between modules.
        self.rewards = []   # reward history
        self.log_pies = []  # action distribution (logarithmic) history
        self.V_preds = []   # value prediction history
        self.R_preds = []   # accumlated reward prediction history
        self.loss = 0       # loss

        if done:
            # episode finished
            self.predictor.reset()
            self.policy.reset()
            self.h = np.zeros((1,H_DIM), dtype=np.float32)
            self.m = np.zeros((1,M_DIM*Kr), dtype=np.float32)
            self.actions = [np.zeros(A_DIM, dtype=np.float32)]
        else:
            # episode goes on
            self.actions = [self.actions[-1]]

    def step(self, o, r, time):
        # state variables
        o_, prev_a_, r_, = make_batch(o, self.actions[-1], r)    # add batch_size dim
        z = self.z_network(o_, prev_a_, r_, self.h, self.m)

        # policy process
        kp, bp = self.policy(z)
        mp = self.memory.read(kp, bp)
        log_pi, a = self.policy.get_action(z, mp)

        # MBP (predictor) process
        a_, = make_batch(a)
        self.h, k, b = self.predictor(z, a_, self.m)
        self.m = self.memory.read(k, b)

        # memory writing
        self.memory.write(z, time)

        # decoding
        o_dec, a_dec, r_dec, V_pred, R_pred = self.decoder(z, log_pi, a_)
        self.V_preds.append(V_pred.reshape(-1))
        self.R_preds.append(R_pred.reshape(-1))
        
        self.loss += self._decode_loss(o_dec, o_, a_dec, prev_a_, r_dec, r_)
        self.rewards.append(r)
        self.actions.append(a)
        self.log_pies.append(log_pi.reshape(-1))

        # return action index
        action = np.where(a==1)[0][0]
        # for graph cisualization
        #self.tmp = [log_pi, self.m, mp, o_dec, a_dec, r_dec, V_pred, R_pred]

        return action

    def _decode_loss(self, o_dec, o, a_dec, a, r_dec, r):
        # WARNING: not taking mean in the paper.
        o_loss = F.mean_squared_error(o_dec, o)
        a_loss = self._bernoulli_softmax_crossentropy(a_dec, a)
        r_loss = F.mean_squared_error(r_dec, r) / 2
        return ALPHA_OBS*o_loss + ALPHA_ACTION*a_loss + ALPHA_REWARD*r_loss

    def _bernoulli_softmax_crossentropy(self, x, y):
        """ x: prediction. unnormalized distribution.
            y: teacher
        """
        return -F.sum(y * F.log_softmax(x) + (1-y) * F.log(1-F.softmax(x)+EPS))

    def update(self, done):
        if done:
            # without bootstrap
            R_rev = [Variable(np.zeros(1, dtype=np.float32))]
            A_rev = [Variable(np.zeros(1, dtype=np.float32))]
            self.V_preds.append(0)
        else:
            # with bootstrap
            R_rev = [self.V_preds[-1]]
            A_rev = [Variable(np.zeros(1, dtype=np.float32))]
            self.R_preds = self.R_preds[:-1]  # delete bootstrap element
            self.rewards = self.rewards[:-1]

        # accumulated rewards
        r_rev = self.rewards[::-1]
        for r in r_rev:
            R_rev.append(r + GAMMA*R_rev[-1])
        R = F.stack(R_rev[1:][::-1], axis=-1)   # (1, TRAIN_INTERVAL)

        # advantages
        N = len(r_rev)
        assert len(self.V_preds) == N+1
        for i in range(N):
            delta = r_rev[i] + GAMMA*self.V_preds[N-i] - self.V_preds[N-i-1]
            A_rev.append(delta + GAMMA*LAMBDA*A_rev[-1])
        A = F.stack(A_rev[1:][::-1], axis=-1)

        # MBP loss
        V_preds = F.stack(self.V_preds[:-1], axis=-1)
        R_preds = F.stack(self.R_preds, axis=-1)
        assert len(R) == len(R_preds) == len(V_preds) == len(A)
        R_loss = (F.sum(F.square(V_preds - R)) + F.sum(F.square(R_preds - R))) / 2
        self.loss += ALPHA_RETURN * R_loss
        self.loss *= ETA_MBP

        # Policy gradient
        A_ = 0
        H = 0
        self.actions = self.actions[1:]  # delete initial action
        for i in range(N):
            log_pi = self.log_pies[i]
            A_ += A[0][i] * log_pi[self.actions[i]==1][0]
            H += -F.matmul(F.exp(log_pi), log_pi)
        self.loss -= ETA_POLICY * (A_ + ALPHA_ENTROPY*H)     # gradient ascend
        #print(self.loss)

        # update
        self.loss_log.append(self.loss.data)
        self.cleargrads()
        self.loss.backward()
        self.optimizer.update()
        self.loss.unchain_backward()    # not needed?
