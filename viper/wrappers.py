import torch
import pydotplus
import numpy as np
import pickle as pk

from typing import Tuple
from stable_baselines3.common.policies import ActorCriticPolicy
from sklearn.tree import DecisionTreeClassifier, export_graphviz


class TreeWrapper:
    def __init__(self, *args, **kwargs):
        self.tree = DecisionTreeClassifier(*args, **kwargs)

    def predict(self, obs, *args, **kwargs):
        return self.tree.predict(obs), None

    def fit(self, obs, acts, **kwargs):
        self.tree.fit(obs, acts, **kwargs)

    def save(self, path):
        with open(path, 'wb') as f:
            pk.dump(self.tree, f)

    def save_graph(self, path):
        dot = export_graphviz(
            self.tree,
            class_names=True,
            filled=True,
            impurity=False,
            rounded=True
        )
        graph = pydotplus.graph_from_dot_data(dot)
        graph.write_png(path)


class SB3Wrapper:
    def __init__(self, model):
        self.device = 'cpu'
        self.model = model

    def predict(self, obs, *args, **kwargs):
        res = self.model.predict(obs)
        if not isinstance(res, tuple):
            return res, None
        return res

    def predict_qs(self, obs, *args, **kwargs):
        return self.model.predict_qs(obs)


class ShieldedMlp(ActorCriticPolicy):
    def __init__(self, *args, shield=None, action_map=None,  **kwargs):
        super().__init__(*args, **kwargs)
        # self.model = model
        self.shield = shield
        self.action_map = action_map

        self.predictions_total = 0
        self.shield_corrections = 0
        self.unsafe_states = 0

    def allowed_actions(self, state):
        return [self.action_map[self.shield.predict(s)] for s in state]
        # return self.action_map[self.shield.predict(state)]

    def enforce_shield(self, obs, actions):
        self.predictions_total += 1
        allowed_actions = self.allowed_actions(obs.cpu().numpy())
        for i in range(len(actions)):
            if actions[i].item() not in allowed_actions[i]:
                self.shield_corrections += 1
                n = len(allowed_actions[i])
                if n == 0:
                    self.unsafe_states += 1
                else:
                    actions[i] = allowed_actions[i][np.random.randint(n)]
                    # actions = torch.tensor(actions).cuda()

        return actions


    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)

        actions = self.enforce_shield(obs, actions)

        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def _predict(self, obs, *args, **kwargs):
        self.predictions_total += 1
        actions = super()._predict(obs, *args, **kwargs)
        return self.enforce_shield(obs, actions)

        # if actions.item() not in allowed_actions:
        #     self.shield_corrections += 1
        #     n = len(allowed_actions)
        #     if n == 0:
        #         self.unsafe_states += 1
        #     else:
        #         actions[0] = allowed_actions[np.random.randint(n)]
        return actions

    # def __call__(self, *args):
    #     import ipdb; ipdb.set_trace()
    #     return super().__call__(*args)

    # def predict_qs(self, obs, *args, **kwargs):
    #     return self.model.predict_qs
