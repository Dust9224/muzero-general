import datetime
import pathlib
import subprocess
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import pyautogui
import mss

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0
        self.max_num_gpus = None

        ### Game
        self.observation_shape = (3, 84, 84)
        self.action_space = list(range(8))
        self.players = list(range(1))
        self.stacked_observations = 0

        self.muzero_player = 0
        self.opponent = None

        ### Self-Play
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 1000
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = None

        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"
        self.support_size = 10
        self.downsample = "CNN"
        self.blocks = 2
        self.channels = 16
        self.reduced_channels_reward = 4
        self.reduced_channels_value = 4
        self.reduced_channels_policy = 4
        self.resnet_fc_reward_layers = [16]
        self.resnet_fc_value_layers = [16]
        self.resnet_fc_policy_layers = [16]

        self.encoding_size = 16
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [32]
        self.fc_reward_layers = [32]
        self.fc_value_layers = []
        self.fc_policy_layers = []

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 100000
        self.batch_size = 64
        self.checkpoint_interval = 100
        self.value_loss_weight = 1
        self.train_on_gpu = False

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.01
        self.lr_decay_rate = 1
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 1000
        self.num_unroll_steps = 10
        self.td_steps = 50
        self.PER = True
        self.PER_alpha = 0.5

        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = 1
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class OvergrowthInterface:
    """Helper class to interact with the Overgrowth executable."""

    def __init__(
        self,
        exe_path: str,
        monitor: Dict[str, int],
        health_boxes: Dict[str, Tuple[int, int, int, int]],
    ):
        self.exe_path = exe_path
        self.monitor = monitor
        self.health_boxes = health_boxes
        self.process = None
        self.sct = mss.mss()

    def start(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen([self.exe_path])
            time.sleep(10)  # Give the game some time to start

    def restart_level(self) -> None:
        pyautogui.press("r")
        time.sleep(1)

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def capture(self) -> np.ndarray:
        img = np.array(self.sct.grab(self.monitor))[:, :, :3]
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.moveaxis(img, -1, 0)
        return img

    def _extract_bar(self, box: Tuple[int, int, int, int]) -> float:
        x, y, w, h = box
        img = np.array(self.sct.grab({"top": y, "left": x, "width": w, "height": h}))[
            :, :, :3
        ]
        return img.mean() / 255.0

    def health(self) -> Dict[str, float]:
        return {name: self._extract_bar(box) for name, box in self.health_boxes.items()}


class Game(AbstractGame):
    ACTION_MAP = {
        0: None,
        1: ["a"],
        2: ["d"],
        3: ["w"],
        4: ["s"],
        5: ["space"],
        6: "left",
        7: "right",
    }

    def __init__(
        self,
        seed=None,
        exe_path="/path/to/Overgrowth.exe",
        monitor=None,
        health_boxes=None,
        action_duration=0.1,
    ):
        self.interface = OvergrowthInterface(
            exe_path,
            monitor or {"top": 0, "left": 0, "width": 1920, "height": 1080},
            health_boxes or {"player": (50, 50, 200, 20), "enemy": (1670, 50, 200, 20)},
        )
        self.action_duration = action_duration
        self.interface.start()
        self.previous_health = self.interface.health()

    def perform_action(self, action: int) -> None:
        mapping = self.ACTION_MAP.get(action)
        if mapping is None:
            return
        if isinstance(mapping, list):
            for key in mapping:
                pyautogui.keyDown(key)
            time.sleep(self.action_duration)
            for key in mapping:
                pyautogui.keyUp(key)
        else:
            pyautogui.mouseDown(button=mapping)
            time.sleep(self.action_duration)
            pyautogui.mouseUp(button=mapping)

    def step(self, action: int):
        self.perform_action(action)
        time.sleep(self.action_duration)
        observation = self.interface.capture()
        health = self.interface.health()
        reward = (self.previous_health["enemy"] - health["enemy"]) - (
            self.previous_health["player"] - health["player"]
        )
        done = health["player"] <= 0 or health["enemy"] <= 0
        self.previous_health = health
        return observation, reward, done

    def legal_actions(self):
        return list(self.ACTION_MAP.keys())

    def reset(self):
        self.interface.restart_level()
        self.previous_health = self.interface.health()
        return self.interface.capture()

    def close(self):
        self.interface.close()

    def render(self):
        pass

    def action_to_string(self, action_number):
        actions = {
            0: "No-op",
            1: "Move left",
            2: "Move right",
            3: "Move forward",
            4: "Move backward",
            5: "Jump",
            6: "Attack",
            7: "Block",
        }
        return f"{action_number}. {actions.get(action_number, 'Unknown')}"
