from __future__ import annotations
from typing import Dict, Tuple, Any, Union, Literal

import numpy as np
from numpy.typing import NDArray

import pygame


class Environment:

    observation_dim = 6
    action_dim = 2
    
    horizon: int
    render_mode: str

    _canvas: pygame.Surface
    _bg_color = (255, 255, 255)
    _chip_color = (255, 0, 0)
    _chip_middle_color = (0, 0, 0)
    _goal_color = (0, 255, 0)
    _actuator_color = (0, 0, 255)
    _pos_radius = 4

    _step_delta = 10
    _chip_radius = 20
    _min_initial_dist = 200
    _min_pos = np.array([0, 0])
    _chip_bound = 2 * np.array([_chip_radius, _chip_radius])
    _max_pos = np.array([1000, 700])

    _np_random: np.random.Generator = None
    _step: int = None
    _goal: NDArray = None
    _chip: NDArray = None
    _actuator: NDArray = None

    _last_dist_to_goal: float = None
    _last_dist_to_chip: float = None
    _last_dist_to_push: float = None

    def __init__(self, horizon: int, render_mode: Union[Literal["human"], None] = "human") -> None:
        self._np_random = np.random.default_rng()

        self.horizon = horizon
        self.render_mode = render_mode
        
        if render_mode == "human":
            self._init_gui()

    def _init_gui(self) -> None:
        pygame.init()
        pygame.display.set_caption("Environment")

        self._canvas = pygame.display.set_mode(tuple(self._max_pos))

    def _perform_action(self, action: NDArray) -> None:
        self._step += 1
        action_size = np.linalg.norm(action)
        
        if action_size < 0.01:
            return
        
        action_norm = action / action_size
        action = self._step_delta * action
        new_actuator = self._actuator + action

        prev_polarity = np.sign((self._actuator - self._chip) @ action_norm)
        new_polarity = np.sign((new_actuator - self._chip) @ action_norm)
        new_actuator_chip_sdist = np.sum((new_actuator - self._chip) ** 2)
        chip_trajectory_dist = np.cross(action_norm, self._chip - self._actuator)

        if (abs(chip_trajectory_dist) < self._chip_radius
            and (prev_polarity != new_polarity or new_actuator_chip_sdist < self._chip_radius ** 2)):
            in_chip_trajectory = np.sqrt(self._chip_radius ** 2 - chip_trajectory_dist ** 2)
            sin_alpha = chip_trajectory_dist / self._chip_radius
            cos_alpha = in_chip_trajectory / self._chip_radius

            rotation = np.array([[cos_alpha, - sin_alpha],
                                 [sin_alpha, cos_alpha]])

            chip_move_norm = rotation @ action_norm
            chip_move = min(self._step_delta, in_chip_trajectory) * cos_alpha * chip_move_norm 
        else:
            chip_move = np.zeros(2)

        new_chip = self._chip + chip_move

        new_actuator_new_chip_sdist = np.sum((new_actuator - new_chip) ** 2)
        if new_actuator_new_chip_sdist < self._chip_radius ** 2:
            new_chip_new_actuator_norm = (new_chip - new_actuator)
            new_chip_new_actuator_norm /= np.linalg.norm(new_chip_new_actuator_norm)
            cos_alpha = chip_move_norm @ new_chip_new_actuator_norm
            
            chip_correction = chip_move_norm * np.sqrt(self._chip_radius ** 2 - new_actuator_new_chip_sdist) / cos_alpha
            new_chip = new_chip + chip_correction

        self._actuator = new_actuator
        self._chip = new_chip
    
    def _get_observation(self) -> NDArray:
        return np.hstack([self._actuator, self._chip - self._actuator, self._goal - self._actuator])
    
    def _get_dist_to_goal(self) -> float:
        return np.linalg.norm(self._goal - self._chip)
    
    def _get_dist_to_chip(self) -> float:
        return np.linalg.norm(self._chip - self._actuator)
    
    def _get_dist_to_push(self) -> float:
        chip_goal_norm = self._chip - self._goal
        chip_goal_norm /= np.linalg.norm(chip_goal_norm)
        actuator_chip_norm = self._actuator - self._chip
        actuator_chip_norm /= np.linalg.norm(actuator_chip_norm)

        return 1 - chip_goal_norm @ actuator_chip_norm
    
    def _evaluate_action(self) -> Tuple[float, bool]:
        dist_to_goal = self._get_dist_to_goal()
        dist_to_chip = self._get_dist_to_chip()
        dist_to_push = self._get_dist_to_push()

        reward = (self._last_dist_to_goal - dist_to_goal) / self._step_delta
        
        if dist_to_goal < self._chip_radius / 2:
            reward += 0.1
        else:
            reward += (self._last_dist_to_chip - dist_to_chip) / self._step_delta

            if dist_to_push > 0.02:
                reward += 2 * (self._last_dist_to_push - dist_to_push)

        if (any(self._actuator < self._min_pos)
            or any(self._actuator > self._max_pos)
            or any(self._chip < self._min_pos + self._chip_bound)
            or any(self._chip > self._max_pos - self._chip_bound)):
            reward -= 0.1
        
        done = self._step >= self.horizon

        return reward, done
    
    def _pos_to_render(self, pos: NDArray) -> Tuple[float]:
        return (pos[0], self._max_pos[1] - pos[1])
    
    def _render(self) -> None:
        if self.render_mode is None:
            return
        
        self._canvas.fill(self._bg_color)
        
        pygame.draw.circle(
            self._canvas, self._chip_color, self._pos_to_render(self._chip), self._chip_radius
        )
        pygame.draw.circle(
            self._canvas, self._chip_middle_color, self._pos_to_render(self._chip), self._pos_radius
        )
        pygame.draw.circle(
            self._canvas, self._goal_color, self._pos_to_render(self._goal), self._pos_radius
        )
        pygame.draw.circle(
            self._canvas, self._actuator_color, self._pos_to_render(self._actuator), self._pos_radius
        )

        pygame.display.flip()
        pygame.time.delay(25)
    
    def step(self, action: NDArray) -> Tuple[NDArray, float, bool, Dict[str, Any]]:
        self._perform_action(action)
        reward, done = self._evaluate_action()
        observation = self._get_observation()

        self._last_dist_to_goal = self._get_dist_to_goal()
        self._last_dist_to_chip = self._get_dist_to_chip()
        self._last_dist_to_push = self._get_dist_to_push()
        info = {"dist": self._last_dist_to_goal}

        self._render()

        return observation, reward, done, info
    
    def reset(self, *, seed: int = None) -> Tuple[NDArray, Dict[str, Any]]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        scale = (self._max_pos - self._min_pos - 2 * self._chip_bound)
        offset = self._min_pos + self._chip_bound
        
        while True:
            self._goal = self._np_random.random(2) * scale / 2 + offset + scale / 4
            self._chip = self._np_random.random(2) * scale + offset
            self._actuator = self._np_random.random(2) * scale + offset

            goal_chip_dist = np.linalg.norm(self._goal - self._chip)
            actuator_chip_dist = np.linalg.norm(self._actuator - self._chip)

            if min(goal_chip_dist, actuator_chip_dist) > self._min_initial_dist:
                break

        self._step = 0
        self._last_dist_to_goal = self._get_dist_to_goal()
        self._last_dist_to_chip = self._get_dist_to_chip()
        self._last_dist_to_push = self._get_dist_to_push()

        self._render()
        
        return self._get_observation(), {"dist": goal_chip_dist}

    def create_snapshot(self) -> Dict[str, Any]:
        return {
            "_np_random_state": dict(self._np_random.bit_generator.state),
            "_step": self._step,
            "_goal": self._goal,
            "_chip": self._chip,
            "_actuator": self._actuator,
            "_last_dist_to_goal": self._last_dist_to_goal,
            "_last_dist_to_chip": self._last_dist_to_chip,
            "_last_dist_to_push": self._last_dist_to_push
        }
    
    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._np_random.bit_generator.state = snapshot["_np_random_state"]
        self._step = snapshot["_step"]
        self._goal = snapshot["_goal"]
        self._chip = snapshot["_chip"]
        self._actuator = snapshot["_actuator"]
        self._last_dist_to_goal = snapshot["_last_dist_to_goal"]
        self._last_dist_to_chip = snapshot["_last_dist_to_chip"]
        self._last_dist_to_push = snapshot["_last_dist_to_push"]


class NormalizedEnvironment(Environment):

    update: bool

    max_norm = 10
    mean = 0
    var = 1
    count = 0
    eps = np.finfo(np.float64).eps.item()
    
    def __init__(
        self, update: bool, horizon: int, render_mode: Union[Literal["human"], None] = "human"
    ) -> None:
        super().__init__(horizon, render_mode)
        
        self.update = update

    def _get_observation(self):
        obs = super()._get_observation()

        if self.update:
            self._update(obs)

        return self._normalize(obs)
    
    def _update(self, observation: NDArray) -> NDArray:
        delta = observation - self.mean
        total_count = self.count + 1

        new_mean = self.mean + delta / total_count

        m_a = self.var * self.count
        m_2 = m_a + (delta ** 2) * self.count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count
    
    def _normalize(self, observation: NDArray) -> NDArray:
        norm = (observation - self.mean) / np.sqrt(self.var + self.eps)
        norm = np.clip(norm, -self.max_norm, self.max_norm)
        
        return norm

    def copy_normalization_params(self, other: NormalizedEnvironment) -> None:
        self.max_norm = other.max_norm
        self.mean = other.mean
        self.var = other.var
        self.count = other.count
        self.eps = other.eps

    def create_snapshot(self) -> Dict[str, Any]:
        return super().create_snapshot() | {
            "max_norm": self.max_norm,
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
            "eps": self.eps
        }
    
    def restore_snapshot(self, snapshot):
        super().restore_snapshot(snapshot)
        
        self.max_norm = snapshot["max_norm"]
        self.mean = snapshot["mean"]
        self.var = snapshot["var"]
        self.count = snapshot["count"]
        self.eps = snapshot["eps"]
