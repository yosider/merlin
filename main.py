# coding: utf-8
from networks.constants import *
from merlin import Merlin

def main():
    agent = Merlin()
    T = 0
    reward_log = []

    for ep in range(NUM_EP):
        s, r = ENV.reset(), 0
        agent.reset()
        ep_time = 0
        ep_reward = 0

        for t in range(1, NUM_EP_STEP+1):
            a = agent.step(s, r)
            s, r, done, info = ENV.step(a)

            ep_reward += r
            ep_time += 1

            if done:
                agent.update(done)
                break
            elif t % TRAIN_INTERVAL == 0:
                # run additional step for bootstrap
                a = agent.step(s, r)
                s, r, done, info = ENV.step(a)
                agent.update(done)
                agent.reset(done=False)


        print('Episode:', ep, 'Reward:', ep_reward)
        reward_log.append(ep_reward)

if __name__ == '__main__':
    main()
