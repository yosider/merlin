import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers, Chain, Variable
from constants import *


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

class Z_network(Chain):
    def __init__(self):
        super(Z_network, self).__init__(
            o_encoder1 = L.Linear(O_DIM, 12),
            o_encoder2 = L.Linear(12, 6),
            prior1 = L.Linear(H_DIM+M_DIM, 2*Z_DIM),
            prior2 = L.Linear(2*Z_DIM, 2*Z_DIM),
            prior3 = L.Linear(2*Z_DIM, 2*Z_DIM),
            f_post1 = L.Linear(6+A_DIM+1+H_DIM+M_DIM+2*Z_DIM, 2*Z_DIM),
            f_post2 = L.Linear(2*Z_DIM, 2*Z_DIM),
            f_post3 = L.Linear(2*Z_DIM, 2*Z_DIM),
        )

    def __call__(self, o, a, r, h, m):
        o_encode = F.relu(self.o_encoder1(o))
        o_encode = F.relu(self.o_encoder2(o_encode))
        #print(o_encode, a, r)
        e = F.concat((o_encode, a, r))

        state = F.concat((h, m))
        prior = F.tanh(self.prior1(state))
        prior = F.tanh(self.prior2(prior))
        prior = self.prior3(prior)

        n = F.concat((e, h, m, prior))
        f_post = F.tanh(self.f_post1(n))
        f_post = F.tanh(self.f_post2(f_post))
        f_post = self.f_post3(f_post)
        posterior = prior + f_post

        gaussian = np.random.normal(size=(1,Z_DIM)).astype(np.float32)
        z = posterior[:, :Z_DIM] + F.exp(posterior[:, Z_DIM:]) * gaussian
        return z

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

class Predictor(Chain):
    def __init__(self):
        super(Predictor, self).__init__(
            predictor = DeepLSTM(Z_DIM+A_DIM+M_DIM, H_DIM),
            reader = L.Linear(H_DIM, Kr*(2*Z_DIM+1)),
        )

    def reset(self):
        self.predictor.reset_state()

    def __call__(self, z, a, m):
        state = F.concat((z, a, m))
        h = self.predictor(state)
        i = self.reader(h)
        k = i[:2*Z_DIM*Kr]
        sc = i[2*Z_DIM*Kr:]
        b = F.softplus(sc)
        return h, k, b

class Policy(Chain):
    def __init__(self):
        super(Policy, self).__init__(
            policy = DeepLSTM(Z_DIM+M_DIM, Hp_DIM),
            reader = L.Linear(Hp_DIM, Krp*(2*Z_DIM+1)),
            pi1 = L.Linear(Z_DIM+Hp_DIM+M_DIM, 200),
            pi2 = L.Linear(200, A_DIM),
        )

    def reset(self):
        self.policy.reset_state()
        self.h = np.zeros((1, Hp_DIM), dtype=np.float32)
        self.m = np.zeros((1, M_DIM), dtype=np.float32)

    def __call__(self, z):
        state = F.concat((z, self.m))
        self.h = self.policy(state)
        i = self.reader(self.h)
        k = i[:2*Z_DIM*Krp]
        sc = i[2*Z_DIM*Krp:]
        b = F.softplus(sc, 1)
        return k, b

    def get_action(self, z, m):
        # z はstopgradient.
        assert m.shape == (1, M_DIM)
        self.m = m
        state = F.concat((z.data, self.h, m))
        state = F.tanh(self.pi1(state))
        log_pi = F.log_softmax(self.pi2(state)) # 常にlogの形で出力 log_softmaxは安定らしい
        a = np.random.multinomial(1, F.exp(log_pi)[0].data).astype(np.float32) # onehot
        return log_pi, a

class Memory(Chain):
    def __init__(self):
        super(Memory, self).__init__()
    def read(self, k, b):
        return np.zeros((1,M_DIM), dtype=np.float32)
    def write(self, z):
        pass

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
        o_, a_, r_, = self._make_batch(o, self.actions[-1], r)
        z = self.z_network(o_, a_, r_, self.h, self.m)

        kp, bp = self.policy(z)

        mp = self.memory.read(kp, bp)
        log_pi, a = self.policy.get_action(z, mp)
        #print(log_pi)

        next_a_, = self._make_batch(a)
        self.h, k, b = self.predictor(z, next_a_, self.m)
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

    def _make_batch(self, *xs):
        return [x.reshape(1,-1) if type(x)==Variable else np.array(x, dtype=np.float32).reshape(1,-1) for x in xs]

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
        self.cleargrads()
        self.loss.backward()
        self.optimizer.update()
        self.loss.unchain_backward() # 新しくlossを作り直しているのでいらん？
