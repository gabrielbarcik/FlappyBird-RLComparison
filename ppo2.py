import gym
import gym_flappy_bird

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

# multiprocess environment
n_cpu = 1
env = SubprocVecEnv([lambda: gym.make('flappy-bird-v0') for i in range(n_cpu)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo2_flappy_bird")

print('Finished')

