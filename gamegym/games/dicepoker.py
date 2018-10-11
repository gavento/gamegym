from ..game import Game, GameState
from ..distribution import Uniform


class DicePoker(Game):

    """
        chance
         |
         p1
        /|\
       F C R
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

    def __init__(self, dice_size=6, fold_cost=2):
        self.dice_size = dice_size
        self.fold_cost = fold_cost
        dice_combinations = []
        for i in range(dice_size):
            for j in range(dice_size):
                dice_combinations.append((i, j))
        self.dice_combinations = dice_combinations
        self.dice_distribution = Uniform(len(dice_combinations))

    def initial_state(self):
        return DicePokerState(None, None, game=self)

    def players(self):
        return 2


class DicePokerState(GameState):

    ACTIONS1 = ("continue", "raise", "fold")
    ACTIONS2 = ("continue", "fold")

    def player(self):
        h = self.history
        s = len(h)
        if s == 0:
            return self.P_CHANCE
        if h[-1] == "fold":
            return self.P_TERMINAL
        if s == 1:
            return 0
        if s == 2:
            return 1
        if h[-1] == "raise" and h[-2] == "continue":
            return 0
        return self.P_TERMINAL

    def actions(self):
        size = len(self.history)
        if size == 0:
            return self.game.dice_combinations
        if self.history[-1] == "raise":
            return self.ACTIONS2
        return self.ACTIONS1

    def chance_distribution(self):
        return self.game.dice_distribution

    def values(self):
        h = self.history
        if h[-1] == "fold":
            v = self.game.fold_cost
            if len(h) != 3:
                v = -v
        else:
            pair = self.game.dice_combinations[h[0]]
            v = pair[0] - pair[1]

        if h[-2] == "raise":
            v *= 2
        return (v, -v)

    def player_information(self, player):
        pair = self.game.dice_combinations[self.history[0]]
        return (pair[player], self.history[1:])