from gym.envs.registration import register

register(
    id='Memory-v0',
    entry_point='envs.memory:Memory'
)
