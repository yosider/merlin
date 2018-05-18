# coding: utf-8
from chainer import serializers 

from merlin import Merlin
from networks.constants import *
from networks.utils import visualize_log

def main():
    T = 0

    for ep in range(NUM_EP):
        s, r = ENV.reset(), 0
        agent.reset()
        ep_reward = 0

        for ep_time in range(NUM_EP_STEP):
            a = agent.step(s, r, ep_time)
            s, r, done, info = ENV.step(a)

            ep_reward += r
            T += 1

            if done:
                agent.update(done)
                break
            elif (ep_time+1) % TRAIN_INTERVAL == 0:
                # run additional step for bootstrap
                a = agent.step(s, r, ep_time)
                s, r, done, info = ENV.step(a)
                agent.update(False)    # enable bootstrap regardless of done
                agent.reset(done)
                if done:    # episode sometimes finishes at the bootstrap step
                    break

        #print('Episode:', ep, 'Step:', T, 'Reward:', ep_reward)
        reward_log.append(ep_reward)


if __name__ == '__main__':
    agent = Merlin()
    reward_log = []
    try:
        main()
    except:
        import traceback
        traceback.print_exc()

    # visualize learning history
    if LOGGING:
        visualize_log(reward=reward_log, MBP_loss=agent.mbp_loss_log, policy_loss=agent.policy_loss_log)

    # save the model
    if SAVE_MODEL:
        serializers.save_npz(LOGDIR+'model.npz', agent)
