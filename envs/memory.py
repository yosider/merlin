# coding: utf-8
import sys
import numpy as np
import gym
import gym.spaces

class Memory(gym.Env):
    def __init__(self):
        super().__init__()
        self.ROW = 2
        self.COL = 2
        self.NUM_CARD = self.COL*self.ROW    # number of cards
        self.NUM_TYPE = self.COL*self.ROW//2  # number of card types (== number of card pairs)
        self.NUM_EP_STEP = self.COL*self.ROW*3//2  # Optimal agent can clear all cards within this.

        self.action_space = gym.spaces.Discrete(self.NUM_CARD)
        self.observation_space = gym.spaces.Box(low=0, high=self.NUM_TYPE, shape=(self.NUM_TYPE,))
        self.reward_range = [-1., 10.]
        self._reset()

    def _reset(self):
        # self.cards: (NUM_CARD, NUM_TYPE) 2-D array
        self.cards = np.repeat(np.arange(self.NUM_TYPE), 2)   # two for each card type
        np.random.shuffle(self.cards)

        self.done = False
        self.steps = 0
        self.prev_action = -1   # turned card position at the previous step
        self.prev_card = -1    # turned card type at the previous step
        return self._onehot(self.prev_card)   # the initial observed card is blank

    def _step(self, action):
        if self.done:
            raise Warning("Memory env has already done.")

        if not 0 <= action < self.NUM_CARD:
            raise ValueError("Action must be 0 ~ {}".format(self.NUM_CARD-1))

        card = self.cards[action]
        if card > -1 and card == self.prev_card and action != self.prev_action:
            # not cleared, same type but not the same one
            # -> clear the pair
            self.cards[action] = -1
            self.cards[self.prev_action] = -1
            reward = 5.
        else:
            reward = -1.

        self.prev_action = action
        self.prev_card = card
        self.steps += 1

        t, c = self._is_timeup(), self._is_complete()
        self.done = t or c
        reward += int(c) * 5.

        return self._onehot(card), reward, self.done, {}

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

    def _is_timeup(self):
        if self.steps >= self.NUM_EP_STEP:
            return True
        else:
            return False

    def _is_complete(self):
        if (self.cards == -1).all():
            return True
        else:
            return False

    def _onehot(self, n):
        x = np.zeros(self.NUM_TYPE)
        if n < 0:
            return x
        else:
            x[n] = 1
            return x