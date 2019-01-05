import numpy as np

from ..algorithms.mccfr import RegretStrategy

class PlotStrategyPCA:
    def __init__(self, game, depth=4, with_regret=None, name=""):
        self.game = game
        self.depth = depth
        self.name = name

        self.d_t = []
        self.d_strat = []
        self.d_regret = []

    def _traverse(self, strategies, state, d, p_strat, p_reg, strat_vec, reg_vec):
        if d >= self.depth:
            return
        if state.is_terminal():
            return
        if state.is_chance():
            strat_pol = state.chance
            reg_pol = strat_pol
        else:
            s = strategies[state.player]
            strat_pol = s.strategy(state)
            if isinstance(s, RegretStrategy):
                obs = state.observations[state.player]
                entry = s.get_entry(obs, len(state.actions))
                reg_pol = s.regret_matching(entry[0])                
            else:
                reg_pol = strat_pol
            strat_vec.extend(p_strat * np.array(strat_pol))
            reg_vec.extend(p_reg * np.array(reg_pol))

        for ai, _ in enumerate(state.actions):
            state2 = self.game.play(state, index=ai)
            self._traverse(strategies, state2, d + 1, p_strat * strat_pol[ai], p_reg * reg_pol[ai], strat_vec, reg_vec)
        return (strat_vec, reg_vec)

    def append(self, t, strategies):
        assert len(strategies) == self.game.players
        assert (not self.d_t) or self.d_t[-1] < t
        strat_vec, reg_vec = self._traverse(strategies, self.game.start(), 0, 1.0, 1.0, [], [])
        self.d_t.append(t)
        self.d_strat.append(np.array(strat_vec))
        self.d_regret.append(np.array(reg_vec))

    def plot(self, fig, base=None, color=None, smooth_regret=11, with_regret=False):
        d_t = np.array(self.d_t)
        d_strat = np.array(self.d_strat)
        d_regret = np.array(self.d_regret)
        if base is None:
            base = self.common_base([self], with_regret=with_regret)
        assert base.shape == (2, d_strat.shape[1])
        proj_strat = np.matmul(d_strat, base.transpose())
        fig.line(proj_strat[:, 0], proj_strat[:, 1], legend=self.name + " strategy", color=color, line_width=1.8)
        fig.x(proj_strat[-1, 0], proj_strat[-1, 1], color=color, line_width=3.0, size=10)
        if with_regret:
            proj_regret = np.matmul(d_regret, base.transpose())
            if smooth_regret > 1:
                from scipy import signal
                proj_regret = signal.savgol_filter(proj_regret, smooth_regret, min(4, smooth_regret - 1), axis=0)
            fig.line(proj_regret[:, 0], proj_regret[:, 1], legend=self.name + " regret", alpha=0.7, color=color)
            fig.x(proj_regret[-1, 0], proj_regret[-1, 1], fill_color=None, color=color, size=10, line_width=1.8)
        return (base, fig)

    @classmethod
    def common_base(cls, traces, with_regret=False):
        import sklearn
        from sklearn.decomposition import PCA
        rows = [t.d_strat for t in traces]
        if with_regret:
            rows.extend([t.d_regret for t in traces])
        pca = PCA(2)
        pca.fit(np.concatenate(rows, axis=0))
        return pca.components_

    @classmethod
    def common_plot(cls, fig, traces, base=None, palette=None, with_regret=False):
        if base is None:
            base = cls.common_base(traces, with_regret=with_regret)
        if palette is None:
            import bokeh.palettes
            palette = bokeh.palettes.Category20_20
        for ti, t in enumerate(traces):
            t.plot(fig, base=base, color=palette[ti], with_regret=with_regret)
