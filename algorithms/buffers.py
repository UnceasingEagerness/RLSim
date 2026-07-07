from __future__ import annotations

# Copyright notice
#
# This file contains code adapted from stable-baselines3
# (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
# licensed under the MIT License.
#
# Copyright (c) 2019-2023 Antonin Raffin, Ashley Hill, Anssi Kanervisto,
# Maximilian Ernestus, Rinu Boney, Pavan Goli, and other contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


__all__ = [
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "RolloutBufferSamples",
    "ReplayBufferSamples",
]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: dict[str, th.Tensor] | th.Tensor
    actions: th.Tensor
    next_observations: dict[str, th.Tensor] | th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    active_masks: th.Tensor


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_device(device: th.device | str = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples | RolloutBufferSamples:
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if self.device.type == "cuda":
            return th.from_numpy(array).pin_memory().to(device=self.device, non_blocking=True)
        else:
            if copy:
                return th.tensor(array, device=self.device)
            return th.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        self.buffer_size = max(buffer_size // n_envs, 1)

        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage
        
        self.is_dict_obs = isinstance(self.observation_space, spaces.Dict)

        # Initialize storage (Dictionary vs Flat Array)
        if self.is_dict_obs:
            self.observations = {
                key: np.zeros((self.buffer_size, self.n_envs, *shape), dtype=self.observation_space.spaces[key].dtype)
                for key, shape in self.obs_shape.items()
            }
            if not optimize_memory_usage:
                self.next_observations = {
                    key: np.zeros((self.buffer_size, self.n_envs, *shape), dtype=self.observation_space.spaces[key].dtype)
                    for key, shape in self.obs_shape.items()
                }
        else:
            self.obs_kin = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape[:-1], self.obs_shape[-1] - 64), dtype=np.float32)
            self.obs_lidar = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape[:-1], 64), dtype=np.uint8)
            if not optimize_memory_usage:
                self.next_obs_kin = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape[:-1], self.obs_shape[-1] - 64), dtype=np.float32)
                self.next_obs_lidar = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape[:-1], 64), dtype=np.uint8)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype))
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.active_masks = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: list[dict[str, Any]], active: np.ndarray = None) -> None:
        action = action.reshape((self.n_envs, self.action_dim))

        if self.is_dict_obs:
            for key in self.observation_space.spaces.keys():
                self.observations[key][self.pos] = np.array(obs[key])
                if self.optimize_memory_usage:
                    self.observations[key][(self.pos + 1) % self.buffer_size] = np.array(next_obs[key])
                else:
                    self.next_observations[key][self.pos] = np.array(next_obs[key])
        else:
            _obs_arr = np.array(obs)
            _next_obs_arr = np.array(next_obs)
            self.obs_kin[self.pos] = _obs_arr[..., :-64].astype(np.float32)
            self.obs_lidar[self.pos] = (_obs_arr[..., -64:] * (255.0 / 70.0)).clip(0, 255).astype(np.uint8)
            if self.optimize_memory_usage:
                self.obs_kin[(self.pos + 1) % self.buffer_size] = _next_obs_arr[..., :-64].astype(np.float32)
                self.obs_lidar[(self.pos + 1) % self.buffer_size] = (_next_obs_arr[..., -64:] * (255.0 / 70.0)).clip(0, 255).astype(np.uint8)
            else:
                self.next_obs_kin[self.pos] = _next_obs_arr[..., :-64].astype(np.float32)
                self.next_obs_lidar[self.pos] = (_next_obs_arr[..., -64:] * (255.0 / 70.0)).clip(0, 255).astype(np.uint8)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.active_masks[self.pos] = np.array(active) if active is not None else np.ones(self.n_envs, dtype=np.float32)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Extract and convert dictionary observations
        if self.is_dict_obs:
            obs_batch = {k: self.to_torch(self.observations[k][batch_inds, env_indices, :]) for k in self.observations.keys()}
            
            if self.optimize_memory_usage:
                next_obs_batch = {k: self.to_torch(self.observations[k][(batch_inds + 1) % self.buffer_size, env_indices, :]) for k in self.observations.keys()}
            else:
                next_obs_batch = {k: self.to_torch(self.next_observations[k][batch_inds, env_indices, :]) for k in self.next_observations.keys()}
                
        # Extract and convert standard array observations
        else:
            k_batch = self.obs_kin[batch_inds, env_indices, :]
            l_batch = self.obs_lidar[batch_inds, env_indices, :].astype(np.float32) * (70.0 / 255.0)
            obs_batch = self.to_torch(np.concatenate([k_batch, l_batch], axis=-1))

            if self.optimize_memory_usage:
                nk_batch = self.obs_kin[(batch_inds + 1) % self.buffer_size, env_indices, :]
                nl_batch = self.obs_lidar[(batch_inds + 1) % self.buffer_size, env_indices, :].astype(np.float32) * (70.0 / 255.0)
                next_obs_batch = self.to_torch(np.concatenate([nk_batch, nl_batch], axis=-1))
            else:
                nk_batch = self.next_obs_kin[batch_inds, env_indices, :]
                nl_batch = self.next_obs_lidar[batch_inds, env_indices, :].astype(np.float32) * (70.0 / 255.0)
                next_obs_batch = self.to_torch(np.concatenate([nk_batch, nl_batch], axis=-1))

        return ReplayBufferSamples(
            observations=obs_batch,
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations=next_obs_batch,
            dones=self.to_torch((self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)),
            rewards=self.to_torch(self.rewards[batch_inds, env_indices].reshape(-1, 1)),
            active_masks=self.to_torch(self.active_masks[batch_inds, env_indices].reshape(-1, 1)),
        )

    def save(self, path: str) -> None:
        """Save the replay buffer to a .npz file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "pos": self.pos,
            "full": self.full,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "timeouts": self.timeouts,
            "active_masks": self.active_masks,
        }
        if self.is_dict_obs:
            for k, v in self.observations.items():
                data[f"obs_{k}"] = v
            if not self.optimize_memory_usage:
                for k, v in self.next_observations.items():
                    data[f"next_obs_{k}"] = v
        else:
            data["obs_kin"] = self.obs_kin
            data["obs_lidar"] = self.obs_lidar
            if not self.optimize_memory_usage:
                data["next_obs_kin"] = self.next_obs_kin
                data["next_obs_lidar"] = self.next_obs_lidar
        np.savez_compressed(path, **data)

    def load(self, path: str) -> None:
        """Load the replay buffer from a .npz file."""
        data = np.load(path)
        self.pos = data["pos"].item()
        self.full = data["full"].item()
        self.actions[:] = data["actions"]
        self.rewards[:] = data["rewards"]
        self.dones[:] = data["dones"]
        self.timeouts[:] = data["timeouts"]
        self.active_masks[:] = data["active_masks"]
        
        if self.is_dict_obs:
            for k in self.observations.keys():
                self.observations[k][:] = data[f"obs_{k}"]
            if not self.optimize_memory_usage:
                for k in self.next_observations.keys():
                    self.next_observations[k][:] = data[f"next_obs_{k}"]
        else:
            self.obs_kin[:] = data["obs_kin"]
            self.obs_lidar[:] = data["obs_lidar"]
            if not self.optimize_memory_usage:
                self.next_obs_kin[:] = data["next_obs_kin"]
                self.next_obs_lidar[:] = data["next_obs_lidar"]

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        if dtype == np.float64:
            return np.float32
        return dtype


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


# =========================================================================
# PRIORITIZED EXPERIENCE REPLAY (PER) ADDONS - FOR FUTURE USE
# =========================================================================

class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    Used for O(log N) sampling of prioritized experience replay.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
        self.size = 0

    def add(self, priority: float):
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we reach bottom, end the search
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
                
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
                
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx
        
    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) Buffer.
    Inherits from ReplayBuffer but uses a SumTree for prioritized sampling.
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.6,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device,
            n_envs, optimize_memory_usage, handle_timeout_termination
        )
        self.alpha = alpha
        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # ReplayBuffer.add() handles the transition writing to self.observations, etc.
        # But we must update the SumTree for each environment transition added
        for i in range(self.n_envs):
            self.tree.add(self.max_priority ** self.alpha)
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample a batch of transitions proportionally to their priority.
        Returns the samples, the indices, and the Importance Sampling (IS) weights.
        """
        indices = np.zeros(batch_size, dtype=np.int32)
        is_weights = np.zeros(batch_size, dtype=np.float32)
        
        segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            tree_idx, priority, data_idx = self.tree.get_leaf(v)
            
            indices[i] = data_idx
            # Calculate IS weight: (1/N * 1/P(i)) ** beta
            prob = priority / self.tree.total_priority
            is_weights[i] = (self.tree.size * prob) ** -beta
            
        # Normalize weights
        is_weights /= is_weights.max()
        
        # Get samples from ReplayBuffer's underlying arrays
        samples = self._get_samples(indices)
        
        return samples, indices, th.tensor(is_weights, dtype=th.float32, device=self.device)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update the tree priorities based on the new absolute TD-errors.
        """
        for i, idx in enumerate(indices):
            priority = (abs(td_errors[i]) + 1e-5) ** self.alpha
            tree_idx = idx + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)
            self.max_priority = max(self.max_priority, priority)