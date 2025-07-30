import pytest
import numpy as np
from types import SimpleNamespace
import replay_buffer
from self_play import GameHistory

ReplayBufferClass = replay_buffer.ReplayBuffer.__ray_metadata__.modified_class

class DummyConfig(SimpleNamespace):
    pass

def create_dummy_config():
    return DummyConfig(
        seed=0,
        PER=False,
        PER_alpha=0.5,
        replay_buffer_size=10,
        batch_size=1,
        stacked_observations=0,
        action_space=[0, 1],
        td_steps=2,
        discount=0.9,
        num_unroll_steps=2,
    )

def create_game_history():
    gh = GameHistory()
    gh.root_values = [1.0, 2.0, 3.0]
    gh.reward_history = [0.0, 1.0, 2.0, 3.0]
    gh.to_play_history = [0, 0, 0, 0]
    gh.child_visits = [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]]
    gh.action_history = [0, 1, 0, 1]
    return gh

def test_compute_target_value():
    config = create_dummy_config()
    rb = ReplayBufferClass({'num_played_games':0, 'num_played_steps':0}, {}, config)
    gh = create_game_history()
    value = rb.compute_target_value(gh, 1)
    assert value == pytest.approx(4.7, rel=1e-5)

def test_make_target():
    config = create_dummy_config()
    rb = ReplayBufferClass({'num_played_games':0, 'num_played_steps':0}, {}, config)
    gh = create_game_history()
    values, rewards, policies, actions = rb.make_target(gh, 1)
    assert len(values) == config.num_unroll_steps + 1
    assert rewards == [1.0, 2.0, 3.0]
    assert actions == [1, 0, 1]

