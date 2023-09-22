import pathlib
import bouncing_ball

from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from viper.utils import load_oracle, make_env
from viper.viper import viper


def train_oracle(env_id, total_timesteps=100_000, n_envs=4, verbose=0):
    vec_env = make_env(env_id, n_envs=n_envs)

    model = PPO('MlpPolicy', vec_env, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)

    env_name = env_id.replace('/', '-')
    model.save(f'./models/{env_name}_ppo')


def train_viper(
    oracle_path, oracle_type, env_id,
    **kwargs
    # max_depth=8, n_iter=80, n_rollouts=100,
    # n_eval_episodes=100, max_samples=80_000
):
    oracle = load_oracle(oracle_path, oracle_type)

    pol = viper(
        oracle,
        env_id,
        **kwargs
        # max_depth,
        # n_iter,
        # n_rollouts,
        # n_eval_episodes,
        # max_samples
    )

    env_name = env_id.replace('/', '-')
    out_dir = f'outputs/{env_name}'

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    pol.save(f'{out_dir}/{env_name}.pk')
    pol.save_graph(f'{out_dir}/{env_name}_graph.png')
    rew, std = evaluate_policy(pol, make_env(env_id))
    print(f'final policy:\n\trew: {rew} (+/- {std})\nsize: {pol.tree.tree_.n_leaves}')
