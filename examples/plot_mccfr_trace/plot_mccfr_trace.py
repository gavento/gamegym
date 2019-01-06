import logging

import bokeh.palettes
import bokeh.plotting
import numpy as np
import plotly
import plotly.graph_objs as pgo
import tqdm

from gamegym.algorithms import OutcomeMCCFR
from gamegym.contrib.plot_strategy import PlotStrategyPCA
from gamegym.games import DicePoker, Goofspiel, MatchingPennies

#logging.basicConfig(level=logging.INFO)


def plot_to_files(g, prefix, n_traces, iters, steps, depth=6, base=None, burn=None, burn_from=0):
    pal = bokeh.palettes.Dark2_8
    
    traces = []
    for ti in tqdm.trange(n_traces, desc=prefix):
        name="MCCFR run #{}".format(ti)
        if burn and ti >= burn_from:
            name += " (burn-in)"
        mc = OutcomeMCCFR(g, seed=hash(str(g)) % 2**30 + ti)
        ps = PlotStrategyPCA(g, depth=depth, name=name)
        for i in tqdm.trange(steps, desc="MCCFR steps"):
            w = 1.0
            if burn and ti >= burn_from and i < steps * burn:
                w = 0.03**(1.0 - float(i) / steps / burn)
            mc.compute(int(iters * (i + 1) / steps) - mc.iterations, progress=False, weight=w)
            ps.append(mc.iterations, mc.strategies)
        traces.append(ps)


    title = "Strategy trajectories of {} in {} iters, {} steps".format(g, iters, steps)
    if n_traces > 1:
        if base is None:
            w_vecs = [t.infoset_weight_vec(mc.strategies) for t in traces]
            base = PlotStrategyPCA.common_base(traces, with_regret=True, weight_vec=np.mean(w_vecs, axis=0))
        objs = PlotStrategyPCA.common_plot(traces, base=base, with_regret=True, palette=pal)
        fig = pgo.Figure(objs, layout=pgo.Layout(hovermode='closest', title=dict(text=title)))
        plotly.offline.plot(fig, filename=prefix + "_all.html", auto_open=False, include_plotlyjs='cdn')

    for ti in range(n_traces):
        objs = traces[ti].plot(color=pal[ti], with_regret=True)
        fig = pgo.Figure(objs, layout=pgo.Layout(hovermode='closest', title=dict(text=title + "(independent PCA)")))
        plotly.offline.plot(fig, filename=prefix + "_{}.html".format(ti), auto_open=False, include_plotlyjs='cdn')

    return base


def main():
    g = MatchingPennies()
    base = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    plot_to_files(g, "plot_mccfr_trace_pennies", 3, 500, 100, base=base)

    g = Goofspiel(4, scoring=Goofspiel.Scoring.ZEROSUM)
    #plot_to_files(g, "plot_mccfr_trace_goof4", 6, 50000, 200, depth=6, burn=0.3, burn_from=3)

    g = DicePoker()
    plot_to_files(g, "plot_mccfr_trace_dicepoker", 6, 200000, 200, depth=8, burn=0.3, burn_from=3)

    g = Goofspiel(5, scoring=Goofspiel.Scoring.ZEROSUM)
    #plot_to_files(g, "plot_mccfr_trace_goof5", 3, 100000, 200, depth=6)


if __name__ == '__main__':
    main()
