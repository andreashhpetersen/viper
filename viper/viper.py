import torch
import numpy as np

from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from viper.utils import make_env
from viper.wrappers import TreeWrapper


def loss(policy, obs):

    if isinstance(policy, PPO):
        device = policy.device
        obs = torch.from_numpy(obs).to(device)

        probs = []
        for action in np.arange(policy.action_space.n):
            actions = torch.tensor([action]).repeat(obs.shape[0]).to(device)
            _, log_probs, _ = policy.policy.evaluate_actions(obs, actions)
            probs.append(log_probs.detach().cpu().numpy())

        probs = np.array(probs).T
        return probs.max(axis=1) - probs.min(axis=1)

    else:
        qs = policy.predict_qs(obs)
        return qs.max(axis=1) - qs.min(axis=1)


def get_rollout(policy, env_id):
    env = make_env(env_id, n_envs=1)

    obss, actss, rews = [], [], []

    obs, done = env.reset(), False
    while not done:
        actions, _states = policy.predict(obs)
        nobs, rewards, done, info = env.step(actions)

        obss.append(obs[0])
        actss.append(actions)
        rews.append(rewards.item())

        obs = nobs

    return np.array(obss), np.array(actss), np.array([sum(rews)])


def get_rollouts(policy, env_id, n_rollouts):
    obss, actss, rewss = [], [], []
    for _ in range(n_rollouts):
        obs, acts, rews = get_rollout(policy, env_id)
        obss.append(obs)
        actss.append(acts)
        rewss.append(rews)

    return np.concatenate(obss), np.concatenate(actss), np.concatenate(rewss)


def sample_idxs(weights, max_samples=80000):
    """
    Return an array of indicies sampled from `weights` (at max `max_samples`)
    """
    return np.random.choice(
        len(weights),
        size=(min(weights.shape[0], max_samples),),
        p=weights / weights.sum()
    )


def get_best_policy(policies, env_id):
    while len(policies) > 1:
        policies = sorted(policies, key=lambda x: -x[-1])

        new_policies = []
        for policy, rew in policies[:int((len(policies) + 1) / 2)]:
            rew, std = evaluate_policy(policy, make_env(env_id))
            new_policies.append((policy, rew))

        policies = new_policies

    return policies[0][0]


def viper(
    oracle, env_id,
    max_depth=8, n_iter=80, n_rollouts=100,
    n_eval_episodes=100, max_samples=80_000, criterion='gini'
):

    obs, acts, _ = get_rollouts(oracle, env_id, n_rollouts)
    weights = loss(oracle, obs)

    policies = []

    for i in tqdm(range(n_iter)):
        idxs = sample_idxs(weights, max_samples=max_samples)
        obs, acts, weights = obs[idxs], acts[idxs], weights[idxs]

        policy = TreeWrapper(
            criterion=criterion,
            ccp_alpha=0.0001,
            max_depth=max_depth
        )
        policy.fit(obs, acts)

        mean_rew, std_rew = evaluate_policy(
            policy,
            make_env(env_id),
            n_eval_episodes=n_eval_episodes
        )
        tqdm.write(f'policy {i}: {mean_rew:0.2f} (+/- {std_rew:0.2f})')

        nobs, _, rews = get_rollouts(policy, env_id, n_rollouts)
        nacts = oracle.predict(nobs, deterministic=True)[0]
        nweights = loss(oracle, nobs)

        obs = np.vstack((obs, nobs))
        acts = np.vstack((acts, np.array([nacts]).T))
        weights = np.concatenate((weights, nweights))

        policies.append((policy, rews.mean()))

    return get_best_policy(policies, env_id)
