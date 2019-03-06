import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import gym_flappy_bird
import datetime

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack

from env_wrapper import make_flappy_env
from stable_baselines import ACER

ENV_ID = 'flappy-bird-v0'

env = make_flappy_env(ENV_ID, num_env=1, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

model = ACER(CnnPolicy, env, verbose=1, tensorboard_log='./acer/acer_1kk_cnn/')

start_time = datetime.datetime.now()

model.learn(total_timesteps=1000000)

print(datetime.datetime.now() - start_time)
model.save("acer_1kk_cnn")

print('Finished')

