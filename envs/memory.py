# coding: utf-8
import sys
import numpy as np
import gym
import gym.spaces

class Memory(gym.Env):
    def __init__(self):
        super().__init__()
        self.ROW = 3
        self.COL = 2
        self.NUM_CARD = self.COL*self.ROW    # number of cards
        self.NUM_TYPE = self.COL*self.ROW//2  # number of card types (== number of card pairs)
        self.NUM_EP_STEP = self.COL*self.ROW*3//2  # Optimal agent can clear all cards within this.

        self.action_space = gym.spaces.Discrete(self.NUM_CARD)
        self.observation_space = gym.spaces.Box(low=0, high=self.NUM_TYPE, shape=(self.NUM_TYPE,))
        self.reward_range = [0., 2.]
        self._reset()

    def _reset(self):
        # self.cards: (NUM_CARD, NUM_TYPE) 2-D array
        order = np.repeat(np.arange(self.NUM_TYPE), 2)   # two for each card type
        np.random.shuffle(order)
        self.cards = np.eye(self.NUM_TYPE)[order]

        self.done = False
        self.steps = 0
        self.prev_action = -1   # turned card position at the previous step
        self.prev_card = np.zeros(self.NUM_TYPE)    # turned card (onehot) at the previous step
        return self.prev_card   # the initial observed card is blank

    def _step(self, action):
        if not 0 <= action < self.NUM_CARD:
            raise ValueError("Action must be 0 ~ {}".format(self.NUM_CARD-1))

        card = self.cards[action]
        if action != self.prev_action and (card == self.prev_card).all() and len(np.nonzero(card)[0]) > 0:
            # clear the pair
            self.cards[action] = np.zeros(self.NUM_TYPE)
            self.cards[self.prev_action] = np.zeros(self.NUM_TYPE)
            reward = 1.
        else:
            reward = 0.

        self.prev_action = action
        self.prev_card = card
        self.steps += 1

        self.done = self._is_done()
        if self.done and self._is_complete():
            reward = 2.

        return card, reward, self.done, {}

    def _render(self, mode='human', close=False):
        pass
        #cards = np.where(self.cards == 1)[1].reshape(self.ROW, self.COL)
        #outfile = StringIO() if mode == 'ansi' else sys.stdout
        #outfile.write('\n'.join(' '.join(str(cards) for elem in row) for row in cards) + '\n')
        #return outfile

    def _close(self):
        pass

    def _seed(self):
        pass

    def _is_done(self):
        if self.steps >= self.NUM_EP_STEP:
            return True
        else:
            return False

    def _is_complete(self):
        if len(np.nonzero(self.cards)[0]) == 0:
            return True
        else:
            return False
