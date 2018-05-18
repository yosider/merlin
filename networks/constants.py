from datetime import datetime 
import os
import gym
import envs

# --- Flags
LOGGING = True
SAVE_MODEL = True
USE_RETROACTIVE = False

# --- Environment parameter
ENV_NAME = 'Memory-v0' # discrete action space
ENV = gym.make(ENV_NAME)
NUM_EP = 1000000000
NUM_EP_STEP = ENV.NUM_EP_STEP

# --- Logfile Directory
LOGROOT = './logs/' + ENV_NAME + '/'
LOGDIR = LOGROOT + datetime.now().isoformat()[:16] + '/'
if LOGGING and not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

# --- Dimensions
O_DIM = ENV.observation_space.shape[0] # observation dimension
A_DIM = ENV.action_space.n # action dimension
Z_DIM = 8 # state variable dimension
H_DIM = 12
Hp_DIM = 12
M_DIM = 2*Z_DIM if USE_RETROACTIVE else Z_DIM
Kr = 3
Krp = 1
N_mem = 10

# --- Learning parameter
ETA_MBP = 1e-2
ETA_POLICY = 1e-1
GAMMA = 1.0
LAMBDA = 0.8
TRAIN_INTERVAL = NUM_EP_STEP
ALPHA_OBS = 1.0
ALPHA_RETURN = 1/NUM_EP_STEP
ALPHA_REWARD = 1.0
ALPHA_ACTION = 1.0
ALPHA_ENTROPY = 1.#0.01

# misc
EPS = 1e-6
