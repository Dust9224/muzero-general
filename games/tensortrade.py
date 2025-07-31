import datetime
import pathlib
from typing import List

import numpy as np
from tensortrade.env.default import create
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.services.execution.simulated import execute_order

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0
        self.max_num_gpus = None

        ### Game
        self.observation_shape = (1, 1, 1)
        self.action_space: List[int] = list(range(181))
        self.players = list(range(1))
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0
        self.opponent = None

        ### Self-Play
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 1000
        self.num_simulations = 50
        self.discount = 0.999
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "fullyconnected"
        self.support_size = 10

        self.encoding_size = 32
        self.fc_representation_layers = [64]
        self.fc_dynamics_layers = [64]
        self.fc_reward_layers = [64]
        self.fc_value_layers = [64]
        self.fc_policy_layers = [64]

        ### Training
        self.results_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "results" / pathlib.Path(__file__).stem
            / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )
        self.save_model = True
        self.training_steps = 10000
        self.batch_size = 64
        self.checkpoint_interval = 10
        self.value_loss_weight = 1
        self.train_on_gpu = False

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.01
        self.lr_decay_rate = 0.8
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 1000
        self.num_unroll_steps = 5
        self.td_steps = 10
        self.PER = True
        self.PER_alpha = 0.5

        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = 1.5
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """TensorTrade trading environment wrapper."""

    def __init__(self, seed: int | None = None):
        np.random.seed(seed if seed is not None else 0)
        prices = np.sin(np.linspace(0, 10, 1000)) + 1.0
        with NameSpace("bitstamp"):
            price_stream = Stream.source(prices, dtype="float").rename("USD-BTC")
        self.feed = DataFeed([price_stream])
        self.feed.compile()
        exchange = Exchange("bitstamp", service=execute_order)(price_stream)
        portfolio = Portfolio(
            USD,
            [Wallet(exchange, 1000 * USD), Wallet(exchange, 1 * BTC)],
        )
        self.env = create(
            portfolio=portfolio,
            action_scheme="managed-risk",
            reward_scheme="simple",
            feed=self.feed,
            window_size=1,
        )

    def step(self, action: int):
        observation, reward, done, _ = self.env.step(action)
        obs = np.array(observation, dtype=np.float32).reshape((1,) + observation.shape)
        return obs, float(reward), bool(done)

    def legal_actions(self):
        return list(range(self.env.action_space.n))

    def reset(self):
        observation = self.env.reset()
        return np.array(observation, dtype=np.float32).reshape((1,) + observation.shape)

    def close(self):
        pass

    def render(self):
        self.env.render()

    def action_to_string(self, action_number: int) -> str:
        return f"Action {action_number}"
