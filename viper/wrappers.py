import pydotplus
import numpy as np
import pickle as pk

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


class ShieldOracleWrapper:
    def __init__(self, shield, action_map):
        self.shield = shield
        self.act_map = action_map
        self.num_actions = len(action_map[0])

    def get_action(self, allowed_actions):
        n = len(allowed_actions)
        if n == 0:
            return self.act_map[0][np.random.randint(self.num_actions)]
        return allowed_actions[np.random.randint(n)]

    def get_allowed_actions(self, obs):
        obs = np.array(obs)
        if len(obs.shape) == 1:
            obs = np.array([obs])

        predictions = [self.shield.predict(s) for s in obs]
        return [self.act_map[p] for p in predictions]

    def predict(self, obs, *args, **kwargs):
        allowed_actions = self.get_allowed_actions(obs)
        return [self.get_action(acts) for acts in allowed_actions], {}

    def predict_qs(self, obs, *args, **kwargs):
        allowed_actions = self.get_allowed_actions(obs)

        qs = np.random.sample((len(allowed_actions,), self.num_actions))
        for i in range(qs.shape[0]):
            acts = allowed_actions[i]
            if len(acts) > 0:
                qs[i,list(acts)] += 10

        return qs
