from flask import Flask, Response, redirect, url_for
from .ui import BuildContext
from ..game import Situation

import xml.etree.ElementTree as et


class Server:
    def __init__(self, *, host="127.0.0.1", port=8080):
        self.host = host
        self.port = port

    def play(self, game):
        self._start_server()

    def play_game(self, game, strategies):
        def play_upto(s):
            p = s.player()
            return p == player or p == Situation.P_TERMINAL

        assert strategies.count(None) == 1
        player = strategies.index(None)

        app = self._flask_app()
        history = game.play_strategies(strategies, upto_fn=play_upto)

        @app.route("/")
        def root():
            state = history[-1]
            title = state.game.name()
            body = "<h2>Game: {}</h2>".format(state.game.name())
            screen = state.make_screen(player, not state.is_terminal())
            build_ctx = BuildContext("/play/")
            element = screen.to_xml(build_ctx)
            body += et.tostring(element).decode()
            return PAGE_CODE.format(title=title, body=body)

        @app.route("/play/<callback_id>")
        def play(callback_id):
            state = history[-1]
            callback_id = int(callback_id)
            screen = state.make_screen(player, True)
            build_ctx = BuildContext("/play/")
            screen.to_xml(build_ctx)
            state = build_ctx.callbacks[callback_id](state)
            history.extend(game.play_strategies(strategies, state0=state, upto_fn=play_upto))
            return redirect("/")

        self._start(app)

    def show_history(self, history, player):
        app = self._flask_app()
        history = [s for s in history if s.player() == player or s.is_terminal()]

        @app.route("/")
        def root():
            return redirect("/step/0")

        @app.route("/step/<step>")
        def step(step):
            step = int(step)
            state = history[step]

            title = state.game.name()
            body = "<h2>Game: {}</h2>".format(state.game.name())
            body += "<p>Step {}/{}</p>".format(step + 1, len(history))
            if step + 1 < len(history):
                body += "<a href='/step/{}'>Next step</a><br/>".format(step + 1)
            if step > 0:
                body += "<a href='/step/{}'>Prev step</a><br/>".format(step - 1)

            screen = state.make_screen(player, False)
            build_ctx = BuildContext()
            element = screen.to_xml(build_ctx)
            body += et.tostring(element).decode()

            return PAGE_CODE.format(title=title, body=body)

        self._start(app)

    def _flask_app(self):
        return Flask("gamegym")

    def _start(self, app):
        app.run(host=self.host, port=self.port)


PAGE_CODE = """
<html>
<head>
<title>{title}</title>
</head>
<body>
<h1>Gamegym</h1>
{body}
</body>
</html>
"""


def xtest_server():

    from gamegym.games import Goofspiel
    from gamegym.strategy import UniformStrategy

    g = Goofspiel(5)

    s = Server()
    s.play_game(g, [None, UniformStrategy()])
