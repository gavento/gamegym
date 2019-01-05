from gamegym.algorithms import OutcomeMCCFR
from gamegym.games import Goofspiel
from gamegym.contrib.plot_strategy import PlotStrategyPCA

import bokeh.plotting, bokeh.palettes
import logging
import tqdm
#logging.basicConfig(level=logging.INFO)


def main():
    N, ITERS = (4, 50000)
    STEPS = 50
    TRIES = 3

    g = Goofspiel(N, scoring=Goofspiel.Scoring.ZEROSUM)
    bokeh.plotting.output_file("bokeh_plot.html")
    fig = bokeh.plotting.figure(title="Strategy trajectories of {} in {} iters".format(g, ITERS))

    trajs = []
    for t in tqdm.trange(TRIES):
        mc = OutcomeMCCFR(g, seed=56 + t)
        ps = PlotStrategyPCA(g, depth=6, with_regret=True)
        for i in tqdm.trange(STEPS):
            mc.compute(ITERS // STEPS, progress=False)
            ps.append(mc.iterations, mc.strategies)
        trajs.append(ps)
    PlotStrategyPCA.common_plot(fig, trajs)
    bokeh.plotting.save(fig)


if __name__ == '__main__':
    main()
