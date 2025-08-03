import datetime
import pathlib
import numpy as np
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0
        self.max_num_gpus = None

        # Game parameters
        self.initial_stones = 15
        self.max_take = 3
        self.observation_shape = (1, 1, 2)
        self.action_space = list(range(self.max_take))
        self.players = list(range(2))
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0
        self.opponent = "expert"

        # Self-Play
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = self.initial_stones
        self.num_simulations = 50
        self.discount = 1
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "fullyconnected"
        self.support_size = 10
        self.encoding_size = 32
        self.fc_representation_layers = [64]
        self.fc_dynamics_layers = [64]
        self.fc_reward_layers = [64]
        self.fc_value_layers = [64]
        self.fc_policy_layers = [64]

        # Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 10000
        self.batch_size = 64
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = torch.cuda.is_available()

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = 0.01
        self.lr_decay_rate = 1
        self.lr_decay_steps = 10000

        # Replay Buffer
        self.replay_buffer_size = 1000
        self.num_unroll_steps = 5
        self.td_steps = 15
        self.PER = True
        self.PER_alpha = 0.5

        # Reanalyze
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        # Adjust the self play / training ratio
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        if trained_steps < 5000:
            return 1.0
        elif trained_steps < 7000:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """Wrapper to interface the Nim environment with MuZero."""

    def __init__(self, seed: int = None):
        self.env = NimEnv(
            initial_stones=MuZeroConfig().initial_stones,
            max_take=MuZeroConfig().max_take,
        )

    def step(self, action: int):
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        return self.env.to_play()

    def legal_actions(self):
        return self.env.legal_actions()

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()
        input("Press enter to take a step ")

    def expert_agent(self) -> int:
        return self.env.expert_action()

    def action_to_string(self, action_number: int) -> str:
        return f"Remove {action_number + 1} stone{'s' if action_number else ''}"


class NimEnv:
    """Simple single-pile Nim game."""

    def __init__(self, initial_stones: int = 15, max_take: int = 3):
        self.initial_stones = initial_stones
        self.max_take = max_take
        self.reset()

    def to_play(self) -> int:
        return 0 if self.player == 1 else 1

    def legal_actions(self):
        max_action = min(self.max_take, self.stones)
        return list(range(max_action))

    def reset(self):
        self.stones = self.initial_stones
        self.player = 1
        return self.get_observation()

    def step(self, action: int):
        take = action + 1
        self.stones -= take
        done = self.stones <= 0
        reward = 1.0 if done else 0.0
        self.player *= -1
        return self.get_observation(), reward, done

    def get_observation(self):
        remaining = self.stones / self.initial_stones
        to_play = float(self.player)
        return np.array([[[remaining, to_play]]], dtype=np.float32)

    def expert_action(self) -> int:
        target = self.max_take + 1
        remainder = self.stones % target
        if remainder == 0:
            return np.random.choice(self.legal_actions())
        return remainder - 1

    def render(self):
        print(f"Stones remaining: {self.stones} | Player {self.to_play() + 1} to play")
