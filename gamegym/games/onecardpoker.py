from ..game import Game, Situation

# TODO: Update OCP to new Game API


class OneCardPoker(Game):
    r"""
        chance
         |
         p1
         |\
         C R
         |   \
         p2   p2
         |\   /\
         C R  C F
           |
           p1
           /\
           F C

    F - fold
    C - call
    R - raise
    """

    def __init__(self, n_cards=3):
        self.n_cards = n_cards
        card_combinations = []
        for i in range(n_cards):
            for j in range(n_cards):
                card_combinations.append((i, j))
        self.card_combinations = card_combinations
        self.card_distribution = Uniform(len(card_combinations))

    def initial_state(self):
        return OneHandPokerState(None, None, game=self)

    def players(self):
        return 2


class OneHandPokerState(Situation):

    ACTIONS1 = ("raise", "check")
    ACTIONS2 = ("call", "fold")

    def player(self):
        h = self.history
        s = len(h)
        if s == 0:
            return self.P_CHANCE
        if s == 1:
            return 0
        if s == 2:
            return 1
        if h[-1] == "raise":
            return 0
        return self.P_TERMINAL

    def actions(self):
        size = len(self.history)
        if size == 0:
            return self.game.card_distribution.values()
        if self.history[-1] == "raise":
            return self.ACTIONS2
        return self.ACTIONS1

    def chance_distribution(self):
        return self.game.card_distribution

    def values(self):
        h = self.history

        if h[-1] == "fold":
            if len(h) == 3:
                return (1, -1)
            else:
                return (-1, 1)

        if h[-1] == "call":
            bet = 2
        else:
            bet = 1

        c1, c2 = self.game.card_combinations[h[0]]
        if c1 == c2:
            return (0, 0)
        if c1 > c2:
            return (bet, -bet)
        else:
            return (-bet, bet)

    def player_information(self, player):
        pair = self.game.card_combinations[self.history[0]]
        return (pair[player], self.history[1:])
