import attr
import numpy as np

from ... import nested
from ...situation import Action, Situation
from .buffer import ReplayRecord


@attr.s(slots=True)
class MCTSNode:
    """
    One node in the MCTS tree with evaluated policy and value.

    Value is a numpy array (value for every player), policy is
    a (non-normalized) distribution on actions (and may be a nested structure).
    """
    situation = attr.ib(type=Situation)
    v_sum = attr.ib(type=np.ndarray)
    policy = attr.ib()

    v_count = attr.ib(default=1, init=False)
    child_nodes = attr.ib(type=list, init=False)
    child_visits = attr.ib(type=list, init=False)

    @classmethod
    def new(cls, situation: Situation, v, policy):
        n = len(situation.actions)
        assert len(policy) == n
        return cls(situation, np.array(v), policy, child_nodes=[None] * n, child_visits=[0] * n)

    def update_value(self, v):
        self.v_sum += v
        self.v_count += 1

    @property
    def value(self):
        return self.v_sum / self.v_count


class MCTSearch:
    """
    Build MCTSNode tree from a given game situation.

    Needs to be given `features` to extract features from a situation, and
    `estimator` to get `(value, policy)` estimate for features.

    After `search()` you can get both the best move and get estimator update
    as `ReplayRecord`.
    """

    def __init__(self, situation: Situation, feature_extractor, estimator):
        self.root = None
        self.iterations = 0
        self.situation = situation
        self.feature_extractor = feature_extractor
        self.estimator = estimator

    def _new_node(self, situation):
        if situation.is_terminal():
            return MCTSNode.new(situation, situation.payoff, policy, child_nodes=[None] * n, child_visits=[0] * n)

        v, policy = estimator()

    def search(self, iterations):
        """
        Run given number of simulations, expanding (at most) one node on each iteration.
        """
        for i in range(iterations):
            self.iterations += 1
            if self.root is None:
                self.features = feature_extractor(self.situation)
                v, policy = estimator(self.features)
                self.root = MCTSNode.new(self.situation, v, policy)
            else:
                self._single_search(self.root)

    def _single_search(self, node: MCTSNode) -> float:
        """
        Recursively run the simulation from the given node,
        returning the value of the (likely new) reached leaf node.
        """
        s = node.situation
        if s.is_terminal():
            return 

    def choose_action(self, exploration: float = 0.0) -> Action:
        raise NotImplementedError

    def best_action(self) -> Action:
        return self.choose_action(exploration=0.0)

    def get_replay_record(self):
        return ReplayRecord(self.features, self.root.v, self.TODO)
