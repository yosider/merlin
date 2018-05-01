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
from utils import *

class Merlin(Chain):
    def __init__(self):
        super(Merlin, self).__init__(
            z_network = Z_network(),
            predictor = Predictor(), #TODO: hは2層？
            decoder = Decoder(),
            policy = Policy(),
            memory = Memory(),
            )
        # module間の処理はmerlinがやる．memory以外のmodule間で共有する変数はmerlinが持つ．
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self)

    def reset(self, done=True):
        self.rewards = []   # 報酬
        self.log_pies = []  # 行動分布(対数)
        self.V_preds = []   # 予測Value
        self.R_preds = []   # 予測累積報酬
        self.loss = 0
        self.delta = 0

        if done:
            # episode終了
            self.predictor.reset()
            self.policy.reset()
            self.h = np.zeros((1,H_DIM), dtype=np.float32)
            self.m = np.zeros((1,M_DIM), dtype=np.float32)
            self.actions = [np.zeros(A_DIM, dtype=np.float32)]
        else:
            # episode継続
            self.actions = [self.actions[-1]]

    def step(self, o, r):
        o_, a_, r_, = make_batch(o, self.actions[-1], r)
        z = self.z_network(o_, a_, r_, self.h, self.m)

        kp, bp = self.policy(z)

        mp = self.memory.read(kp, bp)
        log_pi, a = self.policy.get_action(z, mp)
        #print(log_pi)

        next_a_, = make_batch(a)
        self.h, k, b = self.predictor(z, next_a_, self.m)
        #print(k)
        #print(self.h)
        #print(self.m)
        self.m = self.memory.read(k, b)
        self.memory.write(z)

        o_dec, a_dec, r_dec, V_pred, R_pred = self.decoder(z, log_pi, next_a_)
        self.V_preds.append(V_pred.reshape(-1))
        self.R_preds.append(R_pred.reshape(-1))

        self.loss += self._decode_loss(o_dec, o_, a_dec, a_, r_dec, r_)
        self.rewards.append(r)
        self.actions.append(a)
        self.log_pies.append(log_pi.reshape(-1))

        action = np.where(a==1)[0][0]
        return action

    def _decode_loss(self, o_dec, o, a_dec, a, r_dec, r):
        # WARNING: 論文ではmeanとってない
        o_loss = F.mean_squared_error(o_dec, o)
        a_loss = self._bernoulli_softmax_crossentropy(a_dec, a)
        r_loss = F.mean_squared_error(r_dec, r) / 2
        return ALPHA_OBS*o_loss + ALPHA_ACTION*a_loss + ALPHA_REWARD*r_loss

    def _bernoulli_softmax_crossentropy(self, x, y):
        """ x: prediction. unnormalized distribution.
            y: teacher
        """
        return -F.sum(y * F.log_softmax(x) + (1-y) * F.log(1 - F.softmax(x)))

    def update(self, done):
        if done:
            # bootstrap なし
            R_rev = [0]
            A_rev = [0]
            self.V_preds.append(0)
        else:
            # bootstrap あり
            R_rev = [self.V_preds[-1]]
            A_rev = [0]
            self.R_preds = self.R_preds[:-1]  # bootstrap分削除
            self.rewards = self.rewards[:-1]

        # 累積報酬計算
        r_rev = self.rewards[::-1]
        #print(r_rev)
        for r in r_rev:
            R_rev.append(r + GAMMA*R_rev[-1])
        R = np.array(R_rev[1:][::-1], dtype=Variable)

        # Advantage計算
        N = len(r_rev)
        assert len(self.V_preds) == N+1
        for i in range(N):
            delta = r_rev[i] + GAMMA*self.V_preds[N-i] - self.V_preds[N-i-1]
            A_rev.append(delta + GAMMA*LAMBDA*A_rev[-1])
        A = np.array(A_rev[1:][::-1])
        #print(R)
        #print(A)
        #exit()

        # MBP loss
        V_preds = np.array(self.V_preds[:-1])
        R_preds = np.array(self.R_preds)
        #print(V_preds.shape)
        #print(R_preds.shape)
        #print(R.shape)
        #print(A.shape)
        #print(V_preds)
        #print(R_preds)
        #print(R)
        assert len(R) == len(R_preds) == len(V_preds) == len(A)
        R_loss = (np.sum((V_preds - R)**2) + np.sum((R_preds - R)**2)) / 2
        self.loss += ALPHA_RETURN * R_loss

        # Policy 勾配
        A_ = 0
        H = 0
        self.actions = self.actions[1:]  # 初期値分削除
        for i in range(N):
            log_pi = self.log_pies[i]
            #print(log_pi)
            #print(self.actions[i])
            #print(A[i])
            A_ += A[i][0] * log_pi[self.actions[i]==1][0]
            #print(F.exp(log_pi))
            #print(log_pi)
            #print(np.dot(F.exp(log_pi), log_pi))
            H += -ALPHA_ENTROPY * np.dot(F.exp(log_pi), log_pi)
        self.loss -= A_ + H  # gradient ascend

        # update
        print("  loss: {}".format(self.loss))
        self.cleargrads()
        self.loss.backward()
        self.optimizer.update()
        self.loss.unchain_backward() # 新しくlossを作り直しているのでいらん？
