import gym
import gym_flappy_bird

import numpy as np

from env_wrapper import make_flappy_env

import gym
import gym_flappy_bird

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines import ACER
from stable_baselines.common.atari_wrappers import MaxAndSkipEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv, FrameStack


best_mean_reward, n_steps = -np.inf, 0

ENV_ID = 'flappy-bird-v0'

env = gym.make(ENV_ID)
env = MaxAndSkipEnv(env, skip=4)
env = env_wrapper.wrap_deepmind(env, frame_stack = True)

#env = make_flappy_env(ENV_ID, num_env=1, seed=0)
# Frame-stacking with 4 frames
#env = VecFrameStack(env, n_stack=4)
env = DummyVecEnv([lambda: env])

#model = PPO2(MlpPolicy, env, verbose=1)
#model = DQN(MlpPolicy, env, verbose=1)
model = ACER(MlpPolicy, env, verbose=1)

print('loading model')
#model = PPO2.load('acer_2300k_test')
#model.learn(total_timesteps=10000, callback = callback)
#model.learn(total_timesteps=10000)
import time
print('model loaded')
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #print(obs.shape)
    time.sleep(0.08)
    #import pdb; pdb.set_trace()
    env.render()



