import numpy as np

from ..algorithms.mccfr import RegretStrategy

class PlotStrategyPCA:
    def __init__(self, game, depth=4, with_regret=None, name=""):
        self.game = game
        self.depth = depth
        self.name = name

        infosets = set()
        def _traverse(state, d):
            if state.is_terminal() or d >= self.depth:
                return
            if not state.is_chance():
                infosets.add((state.player, state.observations[state.player], state.actions))
            for ai, a in enumerate(state.actions):
                _traverse(self.game.play(state, index=ai), d + 1)
        _traverse(self.game.start(), 0)
        self.infosets = sorted(list(infosets))

        self.d_t = []
        self.d_strat = []
        self.d_regret = []

    def append(self, t, strategies):
        assert len(strategies) == self.game.players
        assert (not self.d_t) or self.d_t[-1] < t

        strat_vec = []
        reg_vec = []
        for si, iset, actions in self.infosets:
            s = strategies[si]
            strat = s.strategy(iset, len(actions))
            reg = strat
            if isinstance(s, RegretStrategy):
                entry = s.get_entry(iset, len(actions))
                reg = s.regret_matching(entry[0])
            strat_vec.extend(strat)
            reg_vec.extend(reg)

        self.d_t.append(t)
        self.d_strat.append(np.array(strat_vec))
        self.d_regret.append(np.array(reg_vec))

    def infoset_weight_vec(self, strategies):
        assert len(strategies) == self.game.players
        strat_ps = {i: 0.0 for i in self.infosets}
        #reg_ps = {i: 0.0 for i in self.infosets}

        def _traverse(state, strat_p, reg_p, d):
            iset = state.observations[state.player]
            idx = (state.player, iset, state.actions)
            if state.is_terminal() or d >= self.depth:
                return
            if not state.is_chance():
                strat_ps[idx] += strat_p
                #reg_ps[idx] += reg_p
                s = strategies[state.player]
                strat = s.strategy(state)
                reg = strat
                if isinstance(s, RegretStrategy):
                    entry = s.get_entry(iset, len(state.actions))
                    reg = s.regret_matching(entry[0])
            else:
                strat = state.chance
                reg = strat
            for ai, a in enumerate(state.actions):
                _traverse(self.game.play(state, index=ai), strat_p * strat[ai], reg_p * reg[ai], d + 1)

        _traverse(self.game.start(), 1.0, 1.0, 0)
        strat_vec = []
        #reg_vec = []
        for idx in self.infosets:
            strat_vec.extend([strat_ps[idx]] * len(idx[2]))
            #reg_vec.extend([reg_ps[idx]] * len(idx[2]))
        return np.array(strat_vec)  #, np.array(reg_vec))

    def plot(self, base=None, color=None, smooth_regret=0, with_regret=False):
        import plotly.graph_objs as pgo

        d_t = np.array(self.d_t)
        d_text = ["{} its".format(t) for t in d_t]
        d_strat = np.array(self.d_strat)
        d_regret = np.array(self.d_regret)
        if base is None:
            base = self.common_base([self], with_regret=with_regret)
        assert base.shape == (2, d_strat.shape[1])
        
        proj_strat = np.matmul(d_strat, base.transpose())
        objs = [
            pgo.Scatter(x=proj_strat[:, 0], y=proj_strat[:, 1], name=self.name + " strategy", 
                        mode="lines+markers", hovertext=d_text, hoverinfo="text+name", 
                        line=dict(color=color, width=1.8, shape='spline'),
                        marker=dict(symbol="x", color=color, size=[0] * (proj_strat.shape[0] - 1) + [10])),
        ]
        if with_regret:
            proj_regret = np.matmul(d_regret, base.transpose())
            if smooth_regret > 1:
                from scipy import signal
                proj_regret = signal.savgol_filter(proj_regret, smooth_regret, min(4, smooth_regret - 1), axis=0)
            objs.extend([
                pgo.Scatter(x=proj_regret[:, 0], y=proj_regret[:, 1], name=self.name + " regret", 
                            mode="lines+markers", hovertext=d_text, hoverinfo="text+name", 
                            line=dict(color=color, width=0.7, shape='spline', smoothing=0.3),
                            marker=dict(symbol="x", color=color, opacity=0.7, size=[0] * (proj_regret.shape[0] - 1) + [10])),
            ])
        return objs

    @classmethod
    def common_base(cls, traces, with_regret=False, weight_vec=None):
        import sklearn
        from sklearn.decomposition import PCA

        rows = [t.d_strat for t in traces]
        if with_regret:
            rows.extend([t.d_regret for t in traces])
        pca = PCA(2)
        rows = np.concatenate(rows, axis=0)
        if weight_vec is not None:
            rows *= [weight_vec] 
        pca.fit(rows)
        return pca.components_

    @classmethod
    def common_plot(cls, traces, base=None, palette=None, with_regret=False):
        if base is None:
            base = cls.common_base(traces, with_regret=with_regret)
        if palette is None:
            import bokeh.palettes
            palette = bokeh.palettes.Category20_20
        objs = []
        for ti, t in enumerate(traces):
            objs.extend(t.plot(base=base, color=palette[ti], with_regret=with_regret))
        return objs
