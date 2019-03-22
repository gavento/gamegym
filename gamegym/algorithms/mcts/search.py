import attr
import numpy as np
from typing import Iterable

from ... import nested
from ...situation import Action, Situation
from .buffer import ReplayRecord
from ...estimator import EstimatorAdaptor
from ...utils import Distribution


@attr.s(slots=True)
class MCTSNode:
    """
    One node in the MCTS tree with evaluated policy and value.

    Value is a numpy array (value for every player), policy is
    a normalized Distribution on valid actions.
    """
    situation = attr.ib(type=Situation)
    v_sum = attr.ib(type=np.ndarray)
    policy = attr.ib(type=Distribution)

    v_count = attr.ib(default=1, init=False)
    child_nodes = attr.ib(type=list, init=False)  # TODO: or dict?
    child_visits = attr.ib(type=list, init=False)  # TODO: or dict?

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

    Needs to be given `adaptor` to extract features from a situation, and
    `estimator` to get `(values, policy)` estimate for features.

    After `search()` you can get both the best move, explorative move and get estimator update
    as `ReplayRecord`.
    """

    def __init__(self, situation: Situation, adaptor: EstimatorAdaptor, estimator):
        self.root = None
        self.iterations = 0
        self.situation = situation
        self.adaptor = adaptor
        self.estimator = estimator

    def _new_node(self, situation) -> MCTSNode:
        if situation.is_terminal():
            return MCTSNode.new(situation, situation.payoff, None)
        features = self.adaptor.state_features(situation)
        v, probs = self.estimator(features)
        policy = self.adaptor.action_policy(situation, probs)
        return MCTSNode.new(situation, v, policy)

    def search(self, iterations) -> None:
        """
        Run given number of simulations, expanding (at most) one node on each iteration.
        """
        for i in range(iterations):
            self.iterations += 1
            if self.root is None:
                self.root = self._new_node(self.situation)
            else:
                self._single_search(self.root)

    def _single_search(self, node: MCTSNode) -> Iterable[float]:
        """
        Recursively run the simulation from the given node,
        returning the value of the (likely new) reached leaf node.
        """
        s = node.situation
        if s.is_terminal():
            return s.payoff
        # TODO: sample successor, create if non-existent, otherwise recurse
        raise NotImplementedError

    def choose_action(self, exploration: float = 0.0) -> Action:
        raise NotImplementedError
        # TODO: choose action based on formula with visits, values, exploration, ...

    def best_action(self) -> Action:
        # TODO: Select most visited action, play it
        raise NotImplementedError

    def get_replay_record(self):
        return ReplayRecord(self.features, self.root.v, self.TODO)
        # TODO: policy based on child vsits
