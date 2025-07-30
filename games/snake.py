import datetime
import pathlib
import random

import numpy as np
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.board_size = 10
        self.num_obstacles = 0
        self.num_food = 1
        self.observation_shape = (4, self.board_size, self.board_size)
        self.action_space = list(range(4))  # Up, Down, Left, Right
        self.players = list(range(1))
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0
        self.opponent = None

        ### Self-Play
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

        ### Network
        self.network = "fullyconnected"
        self.support_size = 10

        # Residual Network
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

        ### Training
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

        # Exponential learning rate schedule
        self.lr_init = 0.01
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 1000
        self.num_unroll_steps = 10
        self.td_steps = 50
        self.PER = True
        self.PER_alpha = 0.5

        # Reanalyze
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        """Return the temperature for the softmax of visit counts."""
        return 1.0


class Game(AbstractGame):
    """Game wrapper for the Snake environment."""

    def __init__(self, seed: int | None = None):
        config = MuZeroConfig()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.env = SnakeEnv(
            board_size=config.board_size,
            num_obstacles=config.num_obstacles,
            num_food=config.num_food,
        )

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


class SnakeEnv:
    """Snake game environment with optional obstacles and multiple food items."""

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, board_size: int = 10, num_obstacles: int = 0, num_food: int = 1):
        self.size = board_size
        self.num_obstacles = max(0, num_obstacles)
        self.num_food = max(1, num_food)
        self.score = 0
        self.obstacles = set()
        self.food = []
        self.reset()

    def reset(self):
        center = self.size // 2
        self.direction = (0, 1)
        self.snake = [(center, center - 1), (center, center), (center, center + 1)]
        self.score = 0
        self.done = False
        self.spawn_obstacles()
        self.spawn_food(initial=True)
        return self.get_observation()

    def legal_actions(self):
        return list(range(4))

    def step(self, action: int):
        if self.done:
            return self.get_observation(), 0.0, True
        new_dir = SnakeEnv.ACTIONS.get(action, self.direction)
        if (-new_dir[0], -new_dir[1]) == self.direction and len(self.snake) > 1:
            new_dir = self.direction
        self.direction = new_dir

        head_x, head_y = self.snake[-1]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        if (
            not 0 <= new_head[0] < self.size
            or not 0 <= new_head[1] < self.size
            or new_head in self.snake
            or new_head in self.obstacles
        ):
            self.done = True
            return self.get_observation(), -1.0, True

        reward = -0.01
        self.snake.append(new_head)
        if new_head in self.food:
            reward = 1.0
            self.score += 1
            self.food.remove(new_head)
            self.spawn_food()
        else:
            self.snake.pop(0)

        if len(self.snake) == self.size * self.size - len(self.obstacles):
            self.done = True
            reward = 1.0

        return self.get_observation(), reward, self.done

    def spawn_obstacles(self):
        self.obstacles = set()
        if self.num_obstacles <= 0:
            return
        cells = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in self.snake
        ]
        if self.num_obstacles >= len(cells):
            self.num_obstacles = max(0, len(cells) - 1)
        self.obstacles.update(random.sample(cells, self.num_obstacles))

    def spawn_food(self, initial: bool = False):
        available = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in self.snake
            and (x, y) not in self.obstacles
            and (x, y) not in self.food
        ]
        if not available:
            self.done = True
            return

        needed = self.num_food if initial else self.num_food - len(self.food)
        needed = max(0, min(needed, len(available)))
        self.food.extend(random.sample(available, needed))

    def render(self):
        grid = [["."] * self.size for _ in range(self.size)]
        for x, y in self.obstacles:
            grid[x][y] = "#"
        for x, y in self.snake:
            grid[x][y] = "O"
        head_x, head_y = self.snake[-1]
        grid[head_x][head_y] = "H"
        for fx, fy in self.food:
            grid[fx][fy] = "F"
        print("\n".join(" ".join(row) for row in grid))
        print(f"Score: {self.score}")

    def get_observation(self):
        obs = np.zeros((4, self.size, self.size), dtype=np.float32)
        for x, y in self.snake[:-1]:
            obs[1, x, y] = 1.0
        head_x, head_y = self.snake[-1]
        obs[0, head_x, head_y] = 1.0
        for food_x, food_y in self.food:
            obs[2, food_x, food_y] = 1.0
        for ox, oy in self.obstacles:
            obs[3, ox, oy] = 1.0
        return obs
