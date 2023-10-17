from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from viper.wrappers import TreeWrapper, SB3Wrapper, ShieldOracleWrapper


def load_oracle(path, model_type):
    if model_type == 'ppo':
        oracle = PPO.load(path)
    elif model_type == 'dqn':
        raise NotImplementedError('dqn not supported yet')
    elif model_type == 'qtree':
        try:
            from stratetrees.models import QTree
            oracle = SB3Wrapper(QTree(path))
        except ImportError:
            raise ValueError("'qtree' models require the 'stratetrees' " /
                             "package to be installed")
    elif model_type == 'shield':
        try:
            from stratetrees.models import DecisionTree
            action_map = [(0,1), (0,), (1,), ()]
            oracle = ShieldOracleWrapper(DecisionTree.load_from_file(path), action_map)
        except ImportError:
            raise ValueError("'shield' models require the 'stratetrees' " /
                             "package to be installed")

    return oracle


def make_env(env_id, n_envs=1, **kwargs):
    if 'BouncingBall' in env_id:
        try:
            import bouncing_ball
        except ImportError:
            raise ValueError(f"the {env_id} env requires the 'bouncing_ball' " \
                             "package to be installed")

    if 'RandomWalk' in env_id:
        try:
            import random_walk
        except ImportError:
            raise ValueError(f"the {env_id} env requires the 'random_walk' " \
                             "package to be installed")

    return make_vec_env(
        env_id,
        n_envs=n_envs,
        env_kwargs={'render_mode': None},
        **kwargs
    )
