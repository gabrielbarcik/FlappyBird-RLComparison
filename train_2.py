import gym
import gym_flappy_bird

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines import ACER
import numpy as np



from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2

from env_wrapper import make_flappy_env

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



best_mean_reward, n_steps = -np.inf, 0
'''
def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True
'''

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
#model = ACER(MlpPolicy, env, verbose=1)

print('loading model')
model = PPO2.load('ppo2_flappy_bird_cnn_test_dummy_test')
#model.learn(total_timesteps=10000, callback = callback)
#model.learn(total_timesteps=10000)
import time
print('model loaded')
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #print(obs.shape)
    time.sleep(0.03)
    #import pdb; pdb.set_trace()
    env.render()



