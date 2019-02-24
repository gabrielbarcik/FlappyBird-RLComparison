import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import gym_flappy_bird

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

# multiprocess environment
# n_cpu = 16
# env = SubprocVecEnv([lambda: gym.make('flappy-bird-v0') for i in range(n_cpu)])

env = gym.make('flappy-bird-v0')
env = DummyVecEnv([lambda: env])

model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log='./tmp/flappy_bird_cnn/')
model.learn(total_timesteps=3000000)
model.save("ppo2_flappy_bird_cnn")

print('Finished')

