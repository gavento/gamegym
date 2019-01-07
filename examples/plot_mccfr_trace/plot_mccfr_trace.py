import logging

import numpy as np
import plotly
import plotly.graph_objs as pgo
import tqdm

from gamegym.algorithms import OutcomeMCCFR, exploitability, BestResponse
from gamegym.contrib.plot_strategy import StrategyTrace
from gamegym.games import DicePoker, Goofspiel, MatchingPennies
from gamegym.utils import cached
from gamegym.strategy import UniformStrategy

logging.basicConfig(level=logging.INFO)

# How to include PlotlyJS in output: 'cdn', 'directory', True
INCL_JS = 'cdn'


@cached
def compute_mccfr_traces(g,
                         prefix,
                         n_traces,
                         iters,
                         steps,
                         depth=6,
                         burn=None,
                         burn_from=0,
                         add_uniform=True,
                         exploit_every=None,
                         eploit_max_nodes=1e6):
    """
    Computes independent strategy traces of MCCFR in game `g`.
    """
    traces = []
    for ti in tqdm.trange(n_traces, desc=prefix):
        name = "MCCFR run #{}".format(ti)
        if burn and ti >= burn_from:
            name += " (burn-in)"
        mc = OutcomeMCCFR(g, seed=hash(str(g)) % 2**30 + ti)
        ps = StrategyTrace(g, depth=depth, name=name)
        for i in tqdm.trange(steps, desc="MCCFR steps"):
            w = 1.0
            if burn and ti >= burn_from and i < steps * burn:
                w = 0.03**(1.0 - float(i) / steps / burn)
            mc.compute(int(iters * (i + 1) / steps) - mc.iterations, progress=False, weight=w)
            exps = None
            if exploit_every is not None and (steps - i - 1) % exploit_every == 0:
                exps = [
                    exploitability(g, p, mc.strategies[p], max_nodes=eploit_max_nodes)
                    for p in range(g.players)
                ]
            ps.append(mc.iterations, mc.strategies, exps)
        traces.append(ps)

    if add_uniform:
        rps = StrategyTrace(g, depth=depth, name="Uniform")
        rstrat = [UniformStrategy()] * g.players
        rexps = None
        if exploit_every is not None:
            rexps = [
                exploitability(g, p, rstrat[p], max_nodes=eploit_max_nodes)
                for p in range(g.players)
            ]
        for t in traces[0].d_t:
            rps.append(t, rstrat, rexps)
    traces.append(rps)

    return traces


def plot_to_files(g,
                  prefix,
                  n_traces,
                  iters,
                  steps,
                  depth=6,
                  random_projs=3,
                  base=None,
                  burn=None,
                  exploit_every=None,
                  burn_from=0):
    logging.info("Plotting " + prefix)
    traces = compute_mccfr_traces(
        g,
        prefix,
        n_traces,
        iters,
        steps,
        depth=depth,
        burn=burn,
        burn_from=burn_from,
        exploit_every=exploit_every,
        eploit_max_nodes=1e8)
    title = "Strategy trajectories of {} in {} iters, {} steps".format(g, iters, steps)
    pal = plotly.colors.DEFAULT_PLOTLY_COLORS
    w_vec = np.mean([t.infoset_weight_vec(t.last_strategies) for t in traces], axis=0)

    # Common PCA plot
    if n_traces > 1:
        if base is None:
            base = StrategyTrace.common_PCA_base(traces, weight_vec=w_vec)
        fig = StrategyTrace.common_plot(traces, base=base, palette=pal, title=title)
        plotly.offline.plot(
            fig, filename=prefix + "_PCA_all.html", auto_open=False, include_plotlyjs=INCL_JS)

    # Individual PCA plots
    for ti, t in enumerate(traces):
        if t.name.startswith('Uniform'):
            continue
        t_w_vec = t.infoset_weight_vec(t.last_strategies)
        t_base = StrategyTrace.common_PCA_base([t], weight_vec=t_w_vec, seed=42)
        fig = StrategyTrace.common_plot([t], base=t_base, palette=[pal[ti]])
        plotly.offline.plot(
            fig,
            filename=prefix + "_PCA_{}.html".format(ti),
            auto_open=False,
            include_plotlyjs=INCL_JS)

    # Common plot with random projections
    for rpi in range(random_projs):
        r_base = StrategyTrace.common_random_base(traces, weight_vec=w_vec, seed=43)
        fig = StrategyTrace.common_plot(traces, base=r_base, palette=pal)
        plotly.offline.plot(
            fig,
            filename=prefix + "_Rand{}_all.html".format(rpi),
            auto_open=False,
            include_plotlyjs=INCL_JS)

    return base


def main():
    g = MatchingPennies()
    base = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    plot_to_files(g, "plot_mccfr_trace_pennies", 3, 500, 100, base=base, exploit_every=1)

    g = DicePoker(6)
    plot_to_files(
        g,
        "plot_mccfr_trace_dicepoker",
        6,
        100000,
        200,
        depth=6,
        burn=0.3,
        burn_from=3,
        exploit_every=1)

    g = DicePoker(6)
    plot_to_files(
        g,
        "plot_mccfr_trace_dicepoker_long",
        6,
        1000000,
        500,
        depth=6,
        burn=0.3,
        burn_from=3,
        exploit_every=1)

    g = Goofspiel(4, scoring=Goofspiel.Scoring.ZEROSUM)
    plot_to_files(
        g,
        "plot_mccfr_trace_goof4",
        6,
        100000,
        200,
        depth=6,
        burn=0.3,
        burn_from=3,
        exploit_every=1)

    g = Goofspiel(5, scoring=Goofspiel.Scoring.ZEROSUM)
    plot_to_files(
        g,
        "plot_mccfr_trace_goof5",
        6,
        200000,
        200,
        depth=6,
        burn=0.3,
        burn_from=3,
        exploit_every=10)


if __name__ == '__main__':
    main()
