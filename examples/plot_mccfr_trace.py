from gamegym.algorithms import OutcomeMCCFR
from gamegym.games import Goofspiel, MatchingPennies, DicePoker
from gamegym.contrib.plot_strategy import PlotStrategyPCA

import bokeh.plotting, bokeh.palettes
import logging
import tqdm
#logging.basicConfig(level=logging.INFO)


def plot_to_files(g, prefix, n_traces, iters, steps, depth=6, base=None, burn=None):
    pal = bokeh.palettes.Dark2_8
    bokeh.plotting.output_file("/tmp/bokeh_dummy", mode="cdn")

    traces = []
    for ti in tqdm.trange(n_traces, desc=prefix): 
        mc = OutcomeMCCFR(g, seed=hash(str(g)) % 2**30 + ti)
        ps = PlotStrategyPCA(g, depth=depth, name="MCCFR {}".format(ti))
        for i in tqdm.trange(steps, desc="MCCFR steps"):
            w = 1.0
            if burn and i < steps * burn:
                w = 0.03**(1.0 - float(i) / steps / burn)
            mc.compute(iters // steps, progress=False, weight=w)
            ps.append(mc.iterations, mc.strategies)
        traces.append(ps)

    if base is None:
        base = PlotStrategyPCA.common_base(traces, with_regret=True)

    title = "Strategy trajectories of {} in {} iters".format(g, iters)
    if n_traces > 1:
        fig = bokeh.plotting.figure(title=title)
        PlotStrategyPCA.common_plot(fig, traces, base=base, with_regret=False, palette=pal)
        bokeh.plotting.save(fig, prefix + "_all.html", title=title)

        fig = bokeh.plotting.figure(title=title)
        PlotStrategyPCA.common_plot(fig, traces, base=base, with_regret=True, palette=pal)
        bokeh.plotting.save(fig, prefix + "_all_with_regrets.html", title=title)

    for ti in range(n_traces):
        fig = bokeh.plotting.figure(title=title)
        traces[ti].plot(fig, base=base, color=pal[ti], with_regret=True)
        bokeh.plotting.save(fig, prefix + "_{}.html".format(ti), title=title)

        fig = bokeh.plotting.figure(title=title)
        traces[ti].plot(fig, color=pal[ti], with_regret=True)
        bokeh.plotting.save(fig, prefix + "_{}_ownPCA.html".format(ti), title=title)

    return base


def main():
    g = MatchingPennies()
    plot_to_files(g, "plot_mccfr_trace_pennies", 3, 1000, 100)

    g = Goofspiel(4, scoring=Goofspiel.Scoring.ZEROSUM)
    base = plot_to_files(g, "plot_mccfr_trace_goof4", 3, 50000, 100, depth=6)
    plot_to_files(g, "plot_mccfr_trace_goof4burn", 3, 50000, 100, depth=6, base=base, burn=0.3)

    return
    g = DicePoker()
    plot_to_files(g, "plot_mccfr_trace_dicepoker", 3, 50000, 100, depth=10)

    g = Goofspiel(5, scoring=Goofspiel.Scoring.ZEROSUM)
    plot_to_files(g, "plot_mccfr_trace_goof5", 3, 100000, 100, depth=6)


if __name__ == '__main__':
    main()
