# %%

import gym
import balance_bot
from stable_baselines.common.vec_env import VecEnv

env = gym.make("balancebot-v0", render=True)

if not isinstance(env, VecEnv):
    print("reset will happen")
else: print("no reset")
# %%
print("yeah")
# %%
