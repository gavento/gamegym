import attr
import numpy as np
from scipy.special import softmax
import math
from typing import Iterable, Tuple, Union

from ... import nested
from ...situation import Action, Situation
from .buffer import ReplayRecord
from ...estimator import EstimatorAdapter
from ...utils import Distribution


#@attr.s(slots=True)
class MctsNode:
    """
    One node in the MCTS tree with evaluated policy and value.

    Value is a numpy array (value for every player), policy is
    a normalized Distribution on valid actions.
    """

    __slots__ = ("situation", "children", "policy", "value_sum", "visit_count")

    def __init__(self, situation: Situation, policy: Union[Distribution, None]):
        assert policy is None or len(policy) == len(situation.actions)
        self.situation = situation
        self.value_sum = np.zeros(situation.game.players)
        self.visit_count = 0
        self.policy = policy
        self.children = {}

    def value_for_player(self, player):
        return self.value_sum[player] / self.visit_count

    @property
    def value(self):
        return self.value_sum / self.visit_count

    def update_value(self, value):
        self.value_sum += value
        self.visit_count += 1


class MctSearch:
    """
    Build MCTSNode tree from a given game situation.

    Needs to be given `Adapter` to extract features from a situation, and
    `estimator` to get `(values, policy)` estimate for features.

    After `search()` you can get both the best move, explorative move and get estimator update
    as `ReplayRecord`.
    """

    # UCB formula
    PB_C_BASE = 19652
    PB_C_INIT = 1.25

    def __init__(self, situation: Situation, estimator):
        self.root = None
        self.iterations = 0
        self.situation = situation
        self.estimator = estimator
        self.player = situation.player

        assert not situation.is_terminal()
        #self.root = MctsNode(situation, 0)
        self.root, _ = self._create_node(situation)

    def _create_node(self, situation) -> Tuple[MctsNode, float]:
        if situation.is_terminal():
            return MctsNode(situation, None), situation.payoff
        value, policy = self.estimator(situation)
        return MctsNode(situation, policy), value

    def create_dot(self):
        def helper(node):
            label = "val: {}\nvis: {}".format(node.value, node.visit_count)
            lines.append("n{} [label=\"{}\"]".format(id(node), label))
            for action, child in node.children.items():
                helper(child)
                label = str(action)
                lines.append("n{} -> n{} [label=\"{}\"]".format(id(node), id(child), label))
        lines = []
        lines.append("digraph Search {")
        helper(self.root)
        lines.append("}")
        return "\n".join(lines)

    def write_dot(self, filename):
        with open(filename, "w") as f:
            f.write(self.create_dot())

    def search(self, iterations: int) -> None:
        """
        Run given number of simulations, expanding (at most) one node on each iteration.
        """
        for _ in range(iterations):
            self._single_search()
        self.iterations += iterations

    def _select_child_action(self, node):
        if node.policy is None:
            return None  # We are in terminal node

        pb_c = math.log((node.visit_count + self.PB_C_BASE + 1) / self.PB_C_BASE) + self.PB_C_INIT
        pb_c *= math.sqrt(node.visit_count)

        best_score = None
        best_action = None

        player = node.situation.player

        for a, p in node.policy.items():
            child = node.children.get(a)
            if child:
                score = pb_c / (child.visit_count + 1) * p + child.value_for_player(player)
            else:
                score = pb_c * p
            if best_score is None or best_score < score:
                best_score = score
                best_action = a
        return best_action

    def _single_search(self):
        node = self.root
        action = self._select_child_action(node)
        child = node.children.get(action)
        search_path = [node]

        while child:
            search_path.append(child)
            node = child
            action = self._select_child_action(node)
            if action is None:
                break
            child = node.children.get(action)

        if action:  # Nonterminal node
            situation = node.situation.play(action)
            child, value = self._create_node(situation)
            node.children[action] = child
            search_path.append(child)
        else:  # Terminal node
            value = child.situation.payoff
        for node in search_path:
            node.update_value(value)

    def best_action_max_visits(self):
        return max(self.root.children.items(), key=lambda p: p[1].visit_count)[0]

    def best_action_softmax(self):
        actions = list(self.root.children)
        visit_counts = np.array([n.visit_count for n in self.root.children.values()])
        probs = softmax(visit_counts)
        return Distribution(actions, probs).sample()