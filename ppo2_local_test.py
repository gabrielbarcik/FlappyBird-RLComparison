import os

#os.environ['SDL_AUDIODRIVER'] = 'dsp'
#os.putenv('SDL_VIDEODRIVER', 'fbcon')
#os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import gym_flappy_bird

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2
from stable_baselines.common.atari_wrappers import MaxAndSkipEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv, FrameStack
import env_wrapper



# multiprocess environment
# n_cpu = 16
# env = SubprocVecEnv([lambda: gym.make('flappy-bird-v0') for i in range(n_cpu)])

# env = gym.make('flappy-bird-v0')
# env = DummyVecEnv([lambda: env])


ENV_ID = 'flappy-bird-v0'

env = gym.make(ENV_ID)
env = MaxAndSkipEnv(env, skip=4)
env = env_wrapper.wrap_deepmind(env, frame_stack = True)

#env = make_flappy_env(ENV_ID, num_env=1, seed=0)
# Frame-stacking with 4 frames
#env = VecFrameStack(env, n_stack=4)
env = DummyVecEnv([lambda: env])

model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log='./tmp/flappy_bird_cnn_test/')
model.learn(total_timesteps=10000)
model.save("ppo2_flappy_bird_cnn_test_dummy_test")

print('Finished')

