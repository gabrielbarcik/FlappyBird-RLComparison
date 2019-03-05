import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import gym_flappy_bird

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import ACER

# multiprocess environment
n_cpu = 16
env = SubprocVecEnv([lambda: gym.make('flappy-bird-v0') for i in range(n_cpu)])

model = ACER(MlpPolicy, env, verbose=1, tensorboard_log='./acer/flappy_bird1million/')
model.learn(total_timesteps=10000)
model.save("acer_1kkk_flappy_bird")

print('Finished')

