import datetime
import pathlib
import random

import numpy as np
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0
        self.max_num_gpus = None

        # Game parameters
        self.board_size = 10
        self.observation_shape = (4, self.board_size, self.board_size)
        self.action_space = list(range(4))  # up, down, left, right
        self.players = list(range(1))
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0
        self.opponent = "random"

        # Self-Play
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 200
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "fullyconnected"
        self.support_size = 10
        self.downsample = False
        self.blocks = 1
        self.channels = 16
        self.reduced_channels_reward = 16
        self.reduced_channels_value = 16
        self.reduced_channels_policy = 16
        self.resnet_fc_reward_layers = []
        self.resnet_fc_value_layers = []
        self.resnet_fc_policy_layers = []

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [64]
        self.fc_dynamics_layers = [64]
        self.fc_reward_layers = [64]
        self.fc_value_layers = [64]
        self.fc_policy_layers = [64]

        # Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 50000
        self.batch_size = 64
        self.checkpoint_interval = 10
        self.value_loss_weight = 1
        self.train_on_gpu = torch.cuda.is_available()

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.01
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 1000

        # Replay Buffer
        self.replay_buffer_size = 1000
        self.num_unroll_steps = 10
        self.td_steps = 50
        self.PER = True
        self.PER_alpha = 0.5

        # Reanalyze
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        # Ratio
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        return 1.0


class Game(AbstractGame):
    """Game wrapper for the KSnakeDuel environment."""

    def __init__(self, seed: int | None = None):
        config = MuZeroConfig()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.env = KSnakeDuelEnv(board_size=config.board_size)

    def step(self, action: int):
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def legal_actions(self):
        return self.env.legal_actions()

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number: int) -> str:
        actions = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
        return f"{action_number}. {actions[action_number]}"

    def expert_agent(self) -> int:
        return random.choice(self.legal_actions())


class KSnakeDuelEnv:
    """Simplified two player Tron-like snake duel."""

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, board_size: int = 10, max_steps: int = 200):
        self.size = board_size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        center = self.size // 2
        self.p1_dir = (0, 1)
        self.p2_dir = (0, -1)
        self.p1 = [(center, 1)]
        self.p2 = [(center, self.size - 2)]
        self.done = False
        self.steps = 0
        return self.get_observation()

    def legal_actions(self):
        return list(range(4))

    def random_opponent(self) -> int:
        return random.choice(self.legal_actions())

    def check_collision(self, pos) -> bool:
        x, y = pos
        return (
            x < 0
            or x >= self.size
            or y < 0
            or y >= self.size
            or pos in self.p1
            or pos in self.p2
        )

    def step(self, action: int):
        if self.done:
            return self.get_observation(), 0.0, True

        p1_move = KSnakeDuelEnv.ACTIONS.get(action, self.p1_dir)
        if (-p1_move[0], -p1_move[1]) == self.p1_dir:
            p1_move = self.p1_dir
        self.p1_dir = p1_move

        opp_action = self.random_opponent()
        p2_move = KSnakeDuelEnv.ACTIONS.get(opp_action, self.p2_dir)
        if (-p2_move[0], -p2_move[1]) == self.p2_dir:
            p2_move = self.p2_dir
        self.p2_dir = p2_move

        new_head1 = (self.p1[-1][0] + self.p1_dir[0], self.p1[-1][1] + self.p1_dir[1])
        new_head2 = (self.p2[-1][0] + self.p2_dir[0], self.p2[-1][1] + self.p2_dir[1])

        c1 = self.check_collision(new_head1)
        c2 = self.check_collision(new_head2)
        if new_head1 == new_head2:
            c1 = c2 = True

        if not c1:
            self.p1.append(new_head1)
        if not c2:
            self.p2.append(new_head2)

        self.steps += 1

        if c1 and c2:
            reward = 0.0
            self.done = True
        elif c1:
            reward = -1.0
            self.done = True
        elif c2:
            reward = 1.0
            self.done = True
        else:
            reward = -0.01
            if self.steps >= self.max_steps:
                self.done = True

        return self.get_observation(), reward, self.done

    def get_observation(self):
        obs = np.zeros((4, self.size, self.size), dtype=np.float32)
        for x, y in self.p1[:-1]:
            obs[1, x, y] = 1.0
        hx1, hy1 = self.p1[-1]
        obs[0, hx1, hy1] = 1.0
        for x, y in self.p2[:-1]:
            obs[3, x, y] = 1.0
        hx2, hy2 = self.p2[-1]
        obs[2, hx2, hy2] = 1.0
        return obs

    def render(self):
        grid = [["."] * self.size for _ in range(self.size)]
        for x, y in self.p1:
            grid[x][y] = "1"
        for x, y in self.p2:
            if grid[x][y] == "1":
                grid[x][y] = "X"
            else:
                grid[x][y] = "2"
        hx1, hy1 = self.p1[-1]
        hx2, hy2 = self.p2[-1]
        grid[hx1][hy1] = "A"
        grid[hx2][hy2] = "B"
        print("\n".join(" ".join(row) for row in grid))
        print()

