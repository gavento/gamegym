import numpy as np
import pytest
import keras
import tensorflow as tf

from gamegym.algorithms.mcts import search, buffer, alphazero
from gamegym.utils import Distribution
from gamegym.games import Gomoku, gomoku


def build_model(adapter):
    assert len(adapter.data_shapes) == 1
    action_shapes = adapter.action_data_shapes
    assert len(action_shapes) == 1
    action_shape = action_shapes[0]
    inputs = keras.layers.Input(adapter.data_shapes[0])
    x = keras.layers.Flatten()(inputs)
    #x = keras.layers.Dense(32, activation=keras.layers.LeakyReLU)(x)
    x = keras.layers.Dense(32, activation="relu")(x)

    out_values = keras.layers.Dense(2, activation="tanh")(x)

    y = keras.layers.Dense(np.prod(action_shape), activation="softmax")(x)
    out_policy = keras.layers.Reshape(action_shape)(y)

    model = keras.models.Model(
        inputs=inputs,
        outputs=[out_values, out_policy])

    def crossentropy_logits(target, output):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,
                                                          logits=output)

    model.compile(
        loss=['mean_squared_error', crossentropy_logits],
        optimizer='adam')

    return model


def test_alphazero():
    g = Gomoku(3, 3, 3)
    adapter = Gomoku.TensorAdapter(g, symmetrize=True)
    model = build_model(adapter)

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