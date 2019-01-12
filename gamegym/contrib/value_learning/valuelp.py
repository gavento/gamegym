from ..game import Game, Situation
from ..strategy import Strategy
from ..utils import get_rng
from ..algorithms.infosets import InformationSetSampler
import numpy as np
import scipy.optimize
import scipy as sp

# TODO: Update this to new Game API


class LPZeroSumValueLearning:
    EPS = 1e-6
    POS_NEG_WEIGHT = 1e-12

    def __init__(self, game, infosetsampler, feature_extractor, strategies, flex_weight=1.0):
        """
        Initializes the value learning instance with given game, information set sampling
        helper, feature extractor (e.g. `matrix_zerosum_features`) and strategy set.
        If only one strategy is given, it is used for both players.
        """
        self.game = game
        self.infosetsampler = infosetsampler
        self.feature_extractor = feature_extractor
        if isinstance(strategies, Strategy):
            strategies = (strategies, strategies)
        self.strategies = tuple(strategies)
        self.flex_weight = flex_weight

        # Zero feature array and feature indices
        self.feature_0 = feature_extractor(self.game.initial_state())
        self.features = tuple(np.ndindex(*self.feature_0.shape))

        # Weights for variables and variable types
        self.var_weights = {}
        self.value_variables = set()
        self.nonneg_variables = set()
        self.flex_variables = set()

        # Result components
        self.opt = None
        self.result = None  # Dict: var -> value
        self.flex_sum = None
        # dict: variable -> coefficient
        self.conds_eq = []
        self.conds_le = []
        # right side constants
        self.conds_eq_right = []
        self.conds_le_right = []

        assert game.players() == 2
        self._construct_lp()

    def compute(self, sparse=True, method='interior-point', options=None):
        """
        Construct the linear program and run the LP solver.

        Return the computed feature values in a feature shape array.
        """
        var_list = sorted(
            set().union(*[set(c.keys()) for c in self.conds_eq],
                        *[set(c.keys()) for c in self.conds_le]),
            key=str)
        var_index = {v: i for i, v in enumerate(var_list)}
        # Weights
        weights = np.zeros(len(var_list))
        for v, w in self.var_weights.items():
            weights[var_index[v]] = w

        # Conditions matrices
        def create_matrix(conds):
            if sparse:
                A = sp.sparse.dok_matrix((len(conds), len(var_list)))
            else:
                A = np.zeros((len(conds), len(var_list)))
            for i, coeffs in enumerate(conds):
                for var, c in coeffs.items():
                    A[i, var_index[var]] = c
            if sparse:
                A = A.tocsr()
            return A

        A_eq = create_matrix(self.conds_eq)
        b_eq = np.array(self.conds_eq_right)
        A_le = create_matrix(self.conds_le)
        b_le = np.array(self.conds_le_right)

        # Variable bounds
        bounds = [(0.0, None) if var in self.nonneg_variables else (None, None)
                  for var in var_list]
        # LP computation and result extraction
        self.opt = scipy.optimize.linprog(
            weights,
            A_ub=A_le,
            b_ub=b_le,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=method,
            options=options)
        if not self.opt.success:
            raise Exception("Computation failed: {}".format(self.opt))
        self.result = {v: self.opt.x[i] for i, v in enumerate(var_list)}
        self.flex_sum = sum(self.result[v] for v in self.flex_variables)

        f = self.feature_0.copy()
        for v in self.features:
            f.__setitem__(v, self.result.get(v, 0.0))
        return f

    def add_condition(self, coeffs, right, le=False, flexvar=None):
        """
        Add a linear constraint where `coeffs` is a dict: `var`: coefficient,
        `right` is the right side, `le` indicates `<=` or `==` constraint.
        If `flexvar` is given, a variable allowing flexibility in both directions
        is added.

        Variables can be any hashable objects (usually tuples). Feature value variables
        (for the first player) are denoted by their feature index, e.g. `(i, j)` for
        matrix features.
        """
        if (flexvar is not None) and (flexvar not in self.var_weights):
            self.var_weights[flexvar] = self.flex_weight
            self.flex_variables.add(flexvar)
            self.nonneg_variables.add(flexvar)
        if le:
            d = dict(coeffs)
            if flexvar is not None:
                d[flexvar] = -1.0
            self.conds_le.append(d)
            self.conds_le_right.append(right)
        else:
            if flexvar is not None:
                d1 = dict(coeffs)
                d1[flexvar] = -1.0
                self.conds_le.append(d1)
                self.conds_le_right.append(right)
                d2 = {v: -c for v, c in coeffs.items()}
                d2[flexvar] = -1.0
                self.conds_le.append(d2)
                self.conds_le_right.append(-right)
            else:
                d = dict(coeffs)
                self.conds_eq.append(d)
                self.conds_eq_right.append(right)

    def _construct_lp(self):
        """
        Construct the LP conditions for features based on all information sets.
        Adds one variable `("val", player, info)` for every information set expected payoff.
        Adds one variable `("flex", player, info, a)` for tightness of the expected payoff
        condition.
        """
        # Add conditions weakly pushing the feature values to 0.0
        for v in self.features:
            pv = ("fpos", v)
            nv = ("fneg", v)
            self.add_condition({pv: 1.0, nv: -1.0, v: -1.0}, 0.0)
            self.nonneg_variables.add(pv)
            self.nonneg_variables.add(nv)
            self.var_weights[pv] = self.POS_NEG_WEIGHT
            self.var_weights[nv] = self.POS_NEG_WEIGHT
        # Add conditions on action values
        for player in range(self.game.players()):
            info_dist = self.infosetsampler.info_distribution(player)
            for info in info_dist.values():
                value_var = ("val", player, info)
                self.value_variables.add(value_var)
                state_dist = self.infosetsampler.state_distribution(player, info)
                state0 = state_dist.values()[0]
                assert state0.player() == player
                strategy = self.strategies[player].distribution(state0)
                for a in state0.actions():
                    flex_var = ("flex", player, info, a)
                    #print("Conditions for player {} in infoset {}, action {}, var {}, terminals:".format(
                    #    player, info, a, value_var))
                    f_w = self.feature_0.copy()
                    for state, state_p in state_dist.items():
                        st_a = state.play(a)
                        for ts, tp in self._terminals_under(st_a, state_p):
                            f_w += tp * self.feature_extractor(ts)
                    if player == 1:
                        f_w = -f_w
                    coeffs = {
                        i: f_w.__getitem__(i)
                        for i in self.features if f_w.__getitem__(i) != 0.0
                    }
                    coeffs[value_var] = -1.0
                    if strategy.probability(a) <= self.EPS:
                        self.add_condition(coeffs, 0.0, le=True, flexvar=flex_var)
                    else:
                        self.add_condition(coeffs, 0.0, le=False, flexvar=flex_var)

    def _terminals_under(self, state, p0=1.0):
        """
        Iterate over terminal nodes under `state`. Generates `(term_state, p_reach)`
        where the reach corresponds to self.strategies.
        """
        if state.is_terminal():
            yield (state, p0)
        else:
            if state.is_chance():
                dist = state.chance_distribution()
            else:
                dist = self.strategies[state.player()].distribution(state)
            for a, ap in dist.items():
                yield from self._terminals_under(state.play(a), p0 * ap)
