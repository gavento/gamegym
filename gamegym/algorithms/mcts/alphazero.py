
import tensorflow as tf
import numpy as np
from .search import MctSearch
from .buffer import ReplayBuffer, ReplayRecord

from gamegym.utils import Distribution, flatten_array_list

def dummy_estimator(situation):
    return np.array((0, 0)), Distribution(situation.state[1], None)


class AlphaZero:

    def __init__(self,
                 game,
                 adapter,
                 model,
                 batch_size=128,
                 replay_buffer_size=2024,
                 max_moves=1000,
                 num_simulations=800,
                 num_sampling_moves=30):
        assert batch_size <= replay_buffer_size
        self.game = game
        self.adapter = adapter
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.num_sampling_moves = num_sampling_moves
        self.last_model = model
        self.model_generation = 0
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def prefill_replay_buffer(self):
        while self.replay_buffer.records_count < self.batch_size:
            self.play_game()

    def last_estimator(self):
        def model_estimator(situation):
            data = self.adapter.observe_data(situation, situation.player)
            value, logits = self.last_model.predict(data)[0]
            return value, self.adapter.distribution_from_policy_logits(situation, logits)
        if self.model_generation == 0:
            return dummy_estimator
        else:
            return model_estimator

    def play_game(self):
        situation = self.game.start()
        num_simulations = self.num_simulations
        max_moves = self.max_moves
        estimator = self.last_estimator()
        while not situation.is_terminal():
            s = MctSearch(situation, estimator)
            s.search(num_simulations)
            move = len(situation.history)
            if move > max_moves:
                break
            if move <= self.num_sampling_moves:
                action = s.best_action_softmax()
            else:
                action = s.best_action_max_visits()
            self._record_search(s)
            situation = s.root.children[action].situation
        return situation

    def train_network(self):
        batch = self.replay_buffer.get_batch(self.batch_size)
        model = self.last_model  # TODO: clone model
        #model = self.clone_model(self.last_model)
        model.fit(batch.inputs[0], [batch.target_values, batch.target_policy_logits])
        self.model_generation += 1
        self.last_model = model

    def _record_search(self, search):
        children = search.root.children
        values = []
        p = []
        for action in children:
            values.append(action)
            p.append(children[action].visit_count)
        policy_target = self.adapter.encode_actions(Distribution(values, p, norm=True))
        data = self.adapter.get_observation(search.root.situation).data
        assert len(data) == len(self.adapter.data_shapes)
        record = ReplayRecord(data,
                              search.root.value,
                              policy_target)
        self.replay_buffer.add_record(record)