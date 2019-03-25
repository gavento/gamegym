import numpy as np
import pytest
import keras

from gamegym.algorithms.mcts import search, buffer, alphazero
from gamegym.utils import Distribution
from gamegym.games import Gomoku, gomoku


def build_model(board_shape):
    inputs = keras.layers.Input((2,) + board_shape)
    x = keras.layers.Flatten()(inputs)
    #x = keras.layers.Dense(32, activation=keras.layers.LeakyReLU)(x)
    x = keras.layers.Dense(32, activation="relu")(x)

    out_values = keras.layers.Dense(2, activation="tanh")(x)

    y = keras.layers.Dense(board_shape[0] * board_shape[1], activation="softmax")(x)
    out_policy = y #  keras.layers.Reshape(board_shape)(y)

    model = keras.models.Model(
        inputs=inputs,
        outputs=[out_values, out_policy])

    model.compile(
        loss=['mean_squared_error', 'categorical_crossentropy'],
        optimizer='adam')

    return model



def test_alphazero():
    g = Gomoku(3, 3, 3)
    adapter = gomoku.GomokuAdapter(g)
    model = build_model(adapter.state_futures_shape())

    az = alphazero.AlphaZero(
        g, adapter, model,
        max_moves=20, num_simulations=10, batch_size=32, replay_buffer_size=128)
    az.prefill_replay_buffer()

    assert 32 <= az.replay_buffer.records_count <= 128

    az.train_network()
    estimator = az.last_estimator()
    print(estimator(g.start()))
    return

    for i in range(10):
        az.play_game()
        az.train_network()

    estimator = az.last_estimator()
    print(estimator(g.start()))