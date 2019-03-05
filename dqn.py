import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import gym_flappy_bird

from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines import DQN

# multiprocess environment
#n_cpu = 16
#env = SubprocVecEnv([lambda: gym.make('flappy-bird-v0') for i in range(n_cpu)])
env = gym.make('flappy-bird-v0')
env = DummyVecEnv([lambda: env])

model = DQN(CnnPolicy, env, verbose=1, tensorboard_log='./dqn/flappy_bird10million/')
model.learn(total_timesteps=10000000)
model.save("dqn_10kkk_flappy_bird")

print('Finished')

