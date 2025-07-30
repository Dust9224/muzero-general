import numpy as np
from self_play import GameHistory, MinMaxStats

def test_get_stacked_observations_basic():
    gh = GameHistory()
    gh.observation_history = [np.array([[0.]]), np.array([[1.]]), np.array([[2.]])]
    gh.action_history = [0, 1, 0]
    stacked = gh.get_stacked_observations(2, 1, 2)
    assert stacked.shape == (3, 1)
    assert stacked[0, 0] == 2.0
    assert stacked[1, 0] == 1.0
    assert stacked[2, 0] == 0.0

def test_min_max_stats_normalization():
    stats = MinMaxStats()
    stats.update(1.0)
    stats.update(3.0)
    assert stats.minimum == 1.0
    assert stats.maximum == 3.0
    assert stats.normalize(3.0) == 1.0
