from ..game import ObservationSequenceGame, Action
from ..situation import Situation, StateInfo
from ..utils import uniform


class DicePoker(ObservationSequenceGame):
    r"""
        chance
         |
         p1
        /| \
       F C  R
         |   \
         p2   p2
        /|\   /\
       F C R  F C
           |
           p1
           /\
           F C

    F - fold
    C - continue
    R - raise (double value)
    """
    ACTIONS1 = ("continue", "raise", "fold")
    ACTIONS2 = ("continue", "fold")

    def __init__(self, dice_size: int = 6, fold_cost: float = 2.0):
        super().__init__(2, self.ACTIONS1)
        self.dice_size = dice_size
        self.fold_cost = float(fold_cost)
        dice_combinations = []
        for i in range(dice_size):
            for j in range(dice_size):
                dice_combinations.append((i, j))
        self.dice_combinations = dice_combinations
        self.dice_distribution = uniform(len(self.dice_combinations))

    def initial_state(self) -> StateInfo:
        return StateInfo.new_chance(None, self.dice_combinations, self.dice_distribution)

    def _player(self, state: None, action: Action) -> int:
        h = state.history
        s = len(h)
        if action == "fold":
            return StateInfo.TERMINAL
        if s == 0:
            return 0
        if s == 1:
            return 1
        if action == "raise" and h[-1] == "continue":
            return 0
        return StateInfo.TERMINAL

    def update_state(self, situation: Situation, action: Action):
        h = situation.history
        s = len(h)
        pair = h[0] if h else action

        if action == "raise":
            actions = self.ACTIONS2
        else:
            actions = self.ACTIONS1
        player = self._player(situation, action)

        if player >= 0:
            if s == 0:
                obs = (action[0], action[1], None)
            else:
                obs = (action, ) * 3
            return StateInfo.new_player(None, player, actions, observations=obs)

        if action == "fold":
            v = self.fold_cost
            if len(h) != 2:
                v = -v
        else:
            v = float(pair[0] - pair[1])

        if h[-1] == "raise":
            v *= 2
        return StateInfo.new_terminal(None, (v, -v), observations=(action, ) * 3)

    def __repr__(self):
        return "<Dicepoker({}, {})>".format(self.dice_size, self.fold_cost)
