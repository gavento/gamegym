import numpy as np
import scipy as sp

from ..algorithms.mccfr import RegretStrategy


class StrategyTrace:
    """
    Record the progress of strategy computation.

    Has several visualisation and projection methods.
    Tracks the behavior in all information sets encountered up to given depth.
    Keeps reference to last recorded strategy in `self.last_strategies`.
    """

    def __init__(self, game, depth, name=""):
        self.game = game
        self.depth = depth
        self.name = name
        # last added strategies
        self.last_strategies = None

        infosets = set()

        def _traverse(state, d):
            if state.is_terminal() or d >= self.depth:
                return
            if not state.is_chance():
                infosets.add((state.player, state.observations[state.player], state.actions))
            for a in state.actions:
                _traverse(self.game.play(state, a), d + 1)

        _traverse(self.game.start(), 0)

        # List of infosets measured: [(player, observation, actions)]
        self.infosets = sorted(list(infosets))
        # Iterations, shape (t, )
        self.d_t = []
        # Regrets, shape (t, players)
        self.d_exploitability = []
        # Strategy policies in measured infosets, shape (t, actions)
        # `actions` are: first infoset actions, second infoset actions, ...
        self.d_strat = []
        # Regret matching policies in measured infosets, shape (t, actions)
        # Where regret is undefined uses same policy as strategy
        self.d_regret = []

    def append(self, t, strategies, exploitabilities=None):
        """
        Record strategy profile observation at time `t` (iterations).

        Optionally also records the given exploitabilities.
        """
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
        if exploitabilities is None:
            exploitabilities = [np.nan, np.nan]
        assert len(exploitabilities) == self.game.players
        self.d_exploitability.append(exploitabilities)
        self.last_strategies = strategies

    def infoset_weight_vec(self, strategies):
        """
        Compute the probability of every tracked infoset being visited (within given depth).

        Returns array of shape `(actions, )`.
        """
        assert len(strategies) == self.game.players
        strat_ps = {i: 0.0 for i in self.infosets}

        def _traverse(state, strat_p, d):
            iset = state.observations[state.player]
            idx = (state.player, iset, state.actions)
            if state.is_terminal() or d >= self.depth:
                return
            if not state.is_chance():
                strat_ps[idx] += strat_p
                s = strategies[state.player]
                strat = s.strategy(state)
            else:
                strat = state.chance
            for ai, a in enumerate(state.actions):
                _traverse(self.game.play(state, a), strat_p * strat[ai], d + 1)

        _traverse(self.game.start(), 1.0, 0)
        strat_vec = []
        for idx in self.infosets:
            strat_vec.extend([strat_ps[idx]] * len(idx[2]))
        return np.array(strat_vec)

    def plot_trace(self, base, color, smooth_regret=5, with_regret=True):
        """
        Create Plotly objects to plot the trace.

        Base must have shape `(2, tot_actions)`.
        Returns Plotly graph objects. Optionally also returns regret traces.
        """
        import plotly.graph_objs as pgo

        d_t = np.array(self.d_t)
        d_text = ["{} its".format(t) for t in d_t]
        d_text_expl = []
        for dti, dt in enumerate(d_text):
            exp = self.d_exploitability[dti]
            if np.isfinite(exp).any():
                exptext = ", ".join("{:.3f}".format(v) for v in exp)
                d_text_expl.append("{} expls {}".format(dt, exptext))
        d_strat = np.array(self.d_strat)
        d_regret = np.array(self.d_regret)
        assert base.shape == (2, d_strat.shape[1])

        proj_strat = np.matmul(d_strat, base.transpose())
        objs = [
            pgo.Scatter(
                x=proj_strat[:, 0],
                y=proj_strat[:, 1],
                name=self.name + " strategy",
                mode="lines+markers",
                legendgroup=str(id(self)),
                hovertext=d_text_expl,
                hoverinfo="text+name",
                hoverlabel=dict(namelength=-1),
                line=dict(color=color, width=1.8, shape='spline'),
                marker=dict(
                    symbol="x",
                    line=dict(width=0.2, color='black'),
                    color=color,
                    size=[0] * (proj_strat.shape[0] - 1) + [10])),
        ]
        if with_regret:
            proj_regret = np.matmul(d_regret, base.transpose())
            if smooth_regret > 1:
                from scipy import signal
                proj_regret = signal.savgol_filter(
                    proj_regret, smooth_regret, min(4, smooth_regret - 1), axis=0)
            objs.extend([
                pgo.Scatter(
                    x=proj_regret[:, 0],
                    y=proj_regret[:, 1],
                    name=self.name + " regret",
                    mode="lines+markers",
                    hovertext=d_text,
                    hoverinfo="text+name",
                    hoverlabel=dict(namelength=-1),
                    visible='legendonly',
                    line=dict(color=color, width=0.6, shape='spline', smoothing=0.3),
                    marker=dict(
                        line=dict(width=0.2, color='black'),
                        symbol="x",
                        color=color,
                        opacity=0.7,
                        size=[0] * (proj_regret.shape[0] - 1) + [10])),
            ])
        return objs

    def plot_exploitabilities(self, color, showlegend=True):
        """
        Create Plotly objects to plot the exploitability.

        Base must have shape `(2, tot_actions)`.
        Returns Plotly graph objects. Optionally also returns regret traces.
        """
        import plotly.graph_objs as pgo

        d_t = np.array(self.d_t)
        d_exp = np.array(self.d_exploitability)

        objs = []
        for p in range(self.game.players):
            objs.append(
                pgo.Scatter(
                    x=d_t,
                    y=d_exp[:, p] * (-1.0 if p == 1 else 1.0),
                    name="{} p{} best response value".format(self.name, p),
                    mode="lines",
                    legendgroup=str(id(self)),
                    hoverinfo="x+y+name",
                    hoverlabel=dict(namelength=-1),
                    showlegend=showlegend,
                    connectgaps=True,
                    line=dict(color=color, width=1.5)))
        return objs

    @classmethod
    def common_PCA_base(cls, traces, with_regret=True, weight_vec=None, dims=2, seed=None):
        """
        Compute a commpon PCA base of the given trace set.
        
        Returns unit L2-norm base of shape `(dims, tot_actions)`.
        """
        import sklearn
        from sklearn.decomposition import PCA

        rows = [t.d_strat for t in traces]
        if with_regret:
            rows.extend([t.d_regret for t in traces])
        pca = PCA(dims, random_state=seed)
        rows = np.concatenate(rows, axis=0)
        if weight_vec is not None:
            rows *= [weight_vec]
        pca.fit(rows)
        base = pca.components_
        assert base.shape[0] == dims
        base = base / sp.linalg.norm(base, axis=1).reshape((-1, 1))
        return base

    @classmethod
    def common_random_base(cls, traces, weight_vec=None, dims=2, seed=None):
        """
        Compute a random base of the given trace set.

        Returns unit L2-norm base of shape `(dims, tot_actions)`.
        Optionally weights the actions by weight_vec
        """
        base = np.random.normal(size=(dims, len(traces[0].d_strat[0])))
        if weight_vec is not None:
            base *= [weight_vec]
        base = base / sp.linalg.norm(base, axis=1).reshape((-1, 1))
        assert base.shape[0] == dims
        return base

    @classmethod
    def common_plot(cls,
                    traces,
                    base,
                    plot_traces=True,
                    plot_exploitabilities=True,
                    palette=None,
                    with_regret=True,
                    title=None):
        """
        Create a Plotly plot with trace and exploitability subplots.
        """
        import plotly

        assert plot_exploitabilities or plot_traces
        if plot_traces and plot_exploitabilities:
            #fig = plotly.tools.make_subplots(insets=[dict(cell=(1,1), l=0.6, w=0.4, h=0.4, b=0.05)], print_grid=False)
            fig = plotly.tools.make_subplots(cols=2, print_grid=False, column_width=[0.6, 0.4])
        else:
            fig = plotly.graph_objs.Figure()
        if plot_traces:
            fig['layout']['yaxis']['scaleanchor'] = 'x'
        #if plot_exploitabilities:
        #    fig['layout']['yaxis2' if plot_traces else 'yaxis']['type'] = 'log'
        if palette is None:
            palette = plotly.colors.DEFAULT_PLOTLY_COLORS
        for ti, t in enumerate(traces):
            if plot_traces:
                fig.add_traces(t.plot_trace(base=base, color=palette[ti], with_regret=with_regret))
            if plot_exploitabilities:
                objs = t.plot_exploitabilities(color=palette[ti], showlegend=not plot_traces)
                if plot_traces:
                    for o in objs:
                        o.xaxis = 'x2'
                        o.yaxis = 'y2'
                fig.add_traces(objs)
        fig['layout'].update(hovermode='x', title=dict(text=title))
        return fig
