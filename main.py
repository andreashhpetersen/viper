import os
import argparse

from viper.commands import train_oracle, train_viper


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='actions', dest='command')

    parser_train = subparsers.add_parser(
        'train-oracle',
        help='Train an oracle strategy (PPO)'
    )
    parser_train.add_argument(
        'ENV_ID', type=str, default='CartPole-v1',
        help='The env to train on'
    )
    parser_train.add_argument(
        '--total-timesteps', '-t', type=int, default=100_000,
        help='Number of timesteps in training'
    )
    parser_train.add_argument(
        '--n-envs', type=int, default=4,
        help='Number of envs in vectorized environment'
    )
    parser_train.add_argument(
        '--verbosity', '-v', type=int, default=1,
        help='Verbosity level (0 for no output)'
    )

    parser_viper = subparsers.add_parser(
        'train-viper',
        help='Train viper'
    )
    parser_viper.add_argument(
        'ORACLE_PATH',
        help='Path to oracle strategy'
    )
    parser_viper.add_argument(
        'ENV_ID',
        help='Name of the environment (eg. CartPole-v1)'
    )
    parser_viper.add_argument(
        '--oracle-type', '-m',
        type=str, default='ppo', choices=['ppo','dqn','qtree','shield'],
        help='The model type of the oracle'
    )
    parser_viper.add_argument(
        '--max-depth', '-d', type=int, default=8,
        help='Max depth of the student tree'
    )
    parser_viper.add_argument(
        '--n-iter', '-n', type=int, default=20,
        help='Number of training iterations'
    )
    parser_viper.add_argument(
        '--n-rollouts', '-r', type=int, default=100,
        help='Number of training iterations'
    )
    parser_viper.add_argument(
        '--n-eval-episodes', '-e', type=int, default=100,
        help='Number of episodes used in evaluation'
    )
    parser_viper.add_argument(
        '--max-samples', '-s', type=int, default=80000,
        help='The maximum number of samples to maintain'
    )
    parser_viper.add_argument(
        '--criterion', '-c',
        type=str, default='gini', choices=['gini', 'entropy'],
        help='The criterion to use for CART'
    )


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.command == 'train-oracle':
        train_oracle(
            args.ENV_ID,
            args.total_timesteps,
            args.n_envs,
            args.verbosity
        )

    elif args.command == 'train-viper':

        train_viper(
            args.ORACLE_PATH,
            args.oracle_type,
            args.ENV_ID,
            max_depth=args.max_depth,
            n_iter=args.n_iter,
            n_rollouts=args.n_rollouts,
            n_eval_episodes=args.n_eval_episodes,
            max_samples=args.max_samples,
            criterion=args.criterion
        )
