import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import gym_flappy_bird

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2

from env_wrapper import make_flappy_env


# multiprocess environment
# n_cpu = 16
# env = SubprocVecEnv([lambda: gym.make('flappy-bird-v0') for i in range(n_cpu)])

# env = gym.make('flappy-bird-v0')
# env = DummyVecEnv([lambda: env])

ENV_ID = 'flappy-bird-v0'

env = make_flappy_env(ENV_ID, num_env=16, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log='./tmp/flappy_bird_cnn_test/')
model.learn(total_timesteps=10000)
model.save("ppo2_flappy_bird_cnn_test")

print('Finished')

