import numpy as np

from ..algorithms.mccfr import RegretStrategy

class PlotStrategyPCA:
    def __init__(self, game, depth=4, with_regret=None, name=""):
        self.game = game
        self.depth = depth
        self.with_regret = with_regret
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
        if any(isinstance(s, RegretStrategy) for s in strategies) and self.with_regret is None:
            self.with_regret = True
        strat_vec, reg_vec = self._traverse(strategies, self.game.start(), 0, 1.0, 1.0, [], [])
        self.d_t.append(t)
        self.d_strat.append(np.array(strat_vec))
        self.d_regret.append(np.array(reg_vec))

    def plot(self, fig, base=None, color=None, upsample=3):
        d_t = np.array(self.d_t)
        d_strat = np.array(self.d_strat)
        d_regret = np.array(self.d_regret)
        if base is None:
            base = self.common_base([self])
        assert base.shape == (2, d_strat.shape[1])
        proj_strat = np.matmul(d_strat, base.transpose())
        fig.line(proj_strat[:, 0], proj_strat[:, 1], legend=self.name + " strategy", color=color, line_width=2.0)
        fig.square(proj_strat[-1, 0], proj_strat[-1, 1], color=color)
        if self.with_regret:
            proj_regret = np.matmul(d_regret, base.transpose())
            if upsample is not None and upsample > 1:
                from scipy import signal
                #proj_regret = signal.resample_poly(proj_regret, upsample, 1, axis=0)[:-(upsample - 1), :]
                proj_regret = signal.savgol_filter(proj_regret, 13, 4, axis=0)
            fig.line(proj_regret[:, 0], proj_regret[:, 1], legend=self.name + " regret", alpha=0.7, color=color)
            fig.square(proj_regret[-1, 0], proj_regret[-1, 1], fill_color=None, line_color=color)
        return (base, fig)

    @classmethod
    def common_base(cls, trajectories):
        import sklearn
        from sklearn.decomposition import PCA
        pca = PCA(2)
        pca.fit(np.concatenate([t.d_regret for t in trajectories], axis=0))
        return pca.components_

    @classmethod
    def common_plot(cls, fig, trajectories, base=None, palette=None):
        if base is None:
            base = cls.common_base(trajectories)
        if palette is None:
            import bokeh.palettes
            palette = bokeh.palettes.Dark2_8
        for ti, t in enumerate(trajectories):
            t.plot(fig, base=base, color=palette[ti])
#            t.plot(fig, base=base, color=palette[2*ti+1], upsample=None)
