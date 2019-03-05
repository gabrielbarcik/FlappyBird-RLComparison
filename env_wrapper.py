import os

from mpi4py import MPI
import gym
from gym.wrappers import FlattenDictWrapper

from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common.atari_wrappers import MaxAndSkipEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv, FrameStack
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def make_flappy(env_id):
    """
    Create a wrapped atari-like envrionment
    :param env_id: (str) the environment ID
    :return: (Gym Environment) the wrapped atari-like environment
    """
    env = gym.make(env_id)
    env = MaxAndSkipEnv(env, skip=4)
    return env

def make_flappy_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, allow_early_resets=True):
    """
    Create a wrapped, monitored SubprocVecEnv for FlappyBird game.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The gym environment
    """
    
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    
    def make_env(rank):
        def _thunk():
            env = make_flappy(env_id)
            env.seed(seed + rank)
#             env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
#                           allow_early_resets=allow_early_resets)
            return wrap_deepmind(env)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def wrap_deepmind(env, clip_rewards=True, frame_stack=False, scale=True):
    """
    Configure environment for DeepMind-style Atari.
    :param env: (Gym Environment) the atari environment
    :param episode_life: (bool) wrap the episode life wrapper
    :param clip_rewards: (bool) wrap the reward clipping wrapper
    :param frame_stack: (bool) wrap the frame stacking wrapper
    :param scale: (bool) wrap the scaling observation wrapper
    :return: (Gym Environment) the wrapped atari environment
    """
    
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env
