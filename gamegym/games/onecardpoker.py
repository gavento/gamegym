from ..game import Game, Situation

# TODO: Update OCP to new Game API

from ..game import ObservationSequenceGame, Action
from ..situation import Situation, StateInfo
from ..utils import uniform


import itertools


class OneCardPoker(ObservationSequenceGame):
    r"""
        chance
         |
         p1
         |\
         K R
         |   \
         p2   p2
         |\   /\
         K R  C F
           |
           p1
           /\
           F C

    F - fold
    K - check
    C - call
    R - raise
    """

    ACTIONS1 = ("raise", "check")
    ACTIONS2 = ("call", "fold")

    def __init__(self, n_cards: int = 3):
        super().__init__(2, self.ACTIONS1 + self.ACTIONS2)
        self.n_cards = n_cards
        card_combinations = []
        card_combinations = [
            p for p in itertools.product(range(n_cards), range(n_cards))
            if p[0] != p[1]
        ]
        self.card_combinations = card_combinations
        self.card_distribution = uniform(len(self.card_combinations))

    def initial_state(self) -> StateInfo:
        return StateInfo.new_chance(None, self.card_combinations, self.card_distribution)

    def _player(self, state: None, action: Action) -> int:
        h = self.history
        s = len(h)
        if s == 0:
            return self.P_TERMINAL
        if s == 1:
            return 0
        if s == 2:
            return 1
        if h[-1] == "raise":
            return 0
        return self.P_TERMINAL

    def _values(self, cards, bet):
        if cards[0] < cards[1]:
            return (-bet, bet)
        else:
            return (bet, -bet)

    def update_state(self, situation: Situation, action: Action):
        h = situation.history
        s = len(h)

        if s == 0:
            return StateInfo.new_player(None, 0, self.ACTIONS1,
                                        observations=(action[0], action[1], None))
        if s == 1:
            if action == "check":
                return StateInfo.new_player(None, 1, self.ACTIONS1,
                                            observations=(action,) * 3)
            elif action == "raise":
                return StateInfo.new_player(None, 1, self.ACTIONS2,
                                            observations=(action,) * 3)
            else:
                assert 0

        if s == 2:
            if action == "fold":
                return StateInfo.new_terminal(None, (1, -1),
                                              observations=(action,) * 3)
            elif action == "call":
                return StateInfo.new_terminal(None, self._values(h[0], 2),
                                              observations=(action,) * 3)
            elif action == "check":
                return StateInfo.new_terminal(None, self._values(h[0], 1),
                                              observations=(action,) * 3)
            elif action == "raise":
                return StateInfo.new_player(None, 0, self.ACTIONS2,
                                            observations=(action,) * 3)
            else:
                assert 0

        if s == 3:
            if action == "fold":
                return StateInfo.new_terminal(None, (-1, 1),
                                              observations=(action,) * 3)
            elif action == "call":
                return StateInfo.new_terminal(None, self._values(h[0], 2),
                                              observations=(action,) * 3)
            else:
                assert 0

    def __repr__(self):
        return "<OneCardPoker({})>".format(self.n_cards)



# class OneCardPoker(Game):
#     r"""
#         chance
#          |
#          p1
#          |\
#          C R
#          |   \
#          p2   p2
#          |\   /\
#          C R  C F
#            |
#            p1
#            /\
#            F C

#     F - fold
#     C - call
#     R - raise
#     """

#     def __init__(self, n_cards=3):
#         self.n_cards = n_cards
#         card_combinations = []
#         for i in range(n_cards):
#             for j in range(n_cards):
#                 card_combinations.append((i, j))
#         self.card_combinations = card_combinations
#         self.card_distribution = Uniform(len(card_combinations))

#     def initial_state(self):
#         return OneHandPokerState(None, None, game=self)

#     def players(self):
#         return 2


# class OneHandPokerState(Situation):

#     ACTIONS1 = ("raise", "check")
#     ACTIONS2 = ("call", "fold")

#     def player(self):
#         h = self.history
#         s = len(h)
#         if s == 0:
#             return self.P_CHANCE
#         if s == 1:
#             return 0
#         if s == 2:
#             return 1
#         if h[-1] == "raise":
#             return 0
#         return self.P_TERMINAL

#     def actions(self):
#         size = len(self.history)
#         if size == 0:
#             return self.game.card_distribution.values()
#         if self.history[-1] == "raise":
#             return self.ACTIONS2
#         return self.ACTIONS1

#     def chance_distribution(self):
#         return self.game.card_distribution

#     def values(self):
#         h = self.history

#         if h[-1] == "fold":
#             if len(h) == 3:
#                 return (1, -1)
#             else:
#                 return (-1, 1)

#         if h[-1] == "call":
#             bet = 2
#         else:
#             bet = 1

#         c1, c2 = self.game.card_combinations[h[0]]
#         if c1 == c2:
#             return (0, 0)
#         if c1 > c2:
#             return (bet, -bet)
#         else:
#             return (-bet, bet)

#     def player_information(self, player):
#         pair = self.game.card_combinations[self.history[0]]
#         return (pair[player], self.history[1:])
