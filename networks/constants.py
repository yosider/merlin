import os
from datetime import datetime 
import gym

# --- Flags
LOGGING = True 
SAVE_MODEL = True

# --- Environment parameter
ENV_NAME = 'CartPole-v0' # discrete action space
ENV = gym.make(ENV_NAME)
NUM_EP = 10000000
NUM_EP_STEP = 200

# --- Logfile Directory
LOGROOT = './logs/' + ENV_NAME + '/'
LOGDIR = LOGROOT + datetime.now().isoformat()[:16] + '/'
if LOGGING and not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

# --- Dimensions
O_DIM = ENV.observation_space.shape[0] # observation dimension: 4
A_DIM = ENV.action_space.n # action dimension: 2
Z_DIM = O_DIM // 2 # state variable dimension: 2
H_DIM = 16
Hp_DIM = 16
M_DIM = 2*Z_DIM
Kr = 2
Krp = 2
N_mem = 100

# --- Learning parameter
GAMMA = 0.9
LAMBDA = 0.2
TRAIN_INTERVAL = 20
ALPHA_OBS = 0.01
ALPHA_RETURN = 0.01
ALPHA_REWARD = 0.01
ALPHA_ACTION = 0.01
ALPHA_ENTROPY = 0.01

# misc
EPSILON = 1e-5
