# %%
# imports ...
import gym
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import balance_bot

load = False
learn = True
save = True
render = False

mode = "demo"

if mode == "learn":
    load = False
    learn = True
    save = True
    render = False
if mode == "continue":
    load = True
    learn = True
    save = True
    render = False
elif mode == "demo":
    load = True
    learn = False
    save = False
    render = True

# %%
# training ...
def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = sum(lcl['episode_rewards'][-101:-1]) / 100 >= 275
    return not is_solved

env = gym.make("balancebot-v0", render=render)

if load:
    model = DQN.load("balance.pkl", env=env, policy=MlpPolicy)
else:
    model = DQN(MlpPolicy,
                env,
                verbose=1,
                #q_func=model,
                learning_rate=1e-3,
                #max_timesteps=100000,
                buffer_size=100000,
                exploration_initial_eps=0.5,
                exploration_fraction=0.01,
                prioritized_replay=True,
                #learning_starts=1,
                #exploration_final_eps=0.02,
                #train_freq=10,
                #callback=callback
                )

if learn:
    model.learn(
        total_timesteps=1000000,
        #q_func=model,
        #lr=1e-3,
        #max_timesteps=100000,
        #buffer_size=100000,
        #exploration_fraction=0.1,
        # exploration_final_eps=0.02,
        log_interval=10,
        callback=callback
    )
else:
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

if save:
    print("Saving model to balance.pkl")
    model.save("balance.pkl")
