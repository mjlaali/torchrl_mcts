from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import TensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.utils import exploration_type, ExplorationType, set_exploration_type
from torchrl.objectives.value import ValueEstimatorBase

from mcts.tensordict_map import TensorDictMap


class RolloutPolicy(ABC):
    @abstractmethod
    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        pass


@dataclass
class PureMctsRolloutPolicy(RolloutPolicy):
    action_spec: TensorSpec

    def __call__(self, tensordict: TensorDictBase) -> TensorDict:
        node = TensorDict(
            {
                "q_sa": torch.zeros((self.action_spec.shape[-1],), dtype=torch.float32),
                "p_sa": torch.zeros((self.action_spec.shape[-1],), dtype=torch.float32),
                "n_sa": torch.zeros((self.action_spec.shape[-1],), dtype=torch.long),
            },
            batch_size=tensordict.batch_size,
        )
        return node


class AlphaZeroRolloutPolicy(RolloutPolicy):
    value_module: TensorDictModule
    action_value_key: str = "action_value"

    def __call__(self, tensordict: TensorDictBase) -> TensorDict:
        module_output = self.value_module(tensordict)
        q_sa = module_output[self.action_value_key]

        node = TensorDict(
            {q_sa: "q_sa", "p_sa": torch.clone(q_sa), "n_sa": torch.zeros_like(q_sa)},
            batch_size=tensordict.batch_size,
        )
        return node


def ucb_1(node: TensorDictBase) -> torch.Tensor:
    x_hat = node["q_sa"]
    n_sa = node["n_sa"]
    mask = n_sa != 0
    n = torch.sum(n_sa)
    optimism_estimation = x_hat
    optimism_estimation[mask] = x_hat[mask] + 2 * torch.sqrt(torch.log(n) / n_sa[mask])
    return optimism_estimation


def puct_agz(node: TensorDictBase, cpuct: float) -> torch.Tensor:
    """
    puct formula copied from AlphaGoZero paper
    https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf
    """
    n_sa = node["n_sa"]
    p_sa = node["p_sa"]
    x_hat = node["q_sa"]

    n = torch.sum(n_sa, dim=-1)
    u_sa = cpuct * p_sa * torch.sqrt(n) / (1 + n_sa)

    optimism_estimation = x_hat + u_sa
    return optimism_estimation


@dataclass
class TreePolicy:
    exploration_strategy: Callable

    def __call__(self, node: TensorDictBase) -> torch.Tensor:
        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            return self.explore_action(node)

        if exploration_type() == ExplorationType.MODE:
            return self.get_greedy_action(node)
        raise ValueError(exploration_type())

    def get_greedy_action(self, node: TensorDictBase) -> torch.Tensor:
        action = torch.argmax(node["n_sa"], dim=-1)
        return torch.nn.functional.one_hot(action, node["n_sa"].shape[-1])

    def explore_action(self, node: TensorDictBase) -> torch.Tensor:
        action_value = self.exploration_strategy(node)
        max_value, _ = torch.max(action_value, dim=-1)
        action = torch.argmax(
            torch.rand_like(action_value) * (action_value == max_value)
        )
        return torch.nn.functional.one_hot(action, action_value.shape[-1])


def visualize(tensordict: TensorDictBase):
    return
    print("\n\nboard:\n" + str(tensordict["observation"].reshape(-1, 12).numpy()))
    if "action" in tensordict.keys():
        print("\naction:\n" + str(tensordict["action"].numpy()))


@dataclass
class MctsPolicy:
    tree: TensorDictMap
    tree_policy: TreePolicy
    rollout_policy: RolloutPolicy
    value_estimator: ValueEstimatorBase
    action_key: str

    def __call__(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        node = self.tree.get(tensordict)
        if node is None:
            node = self.rollout_policy(tensordict)
            self.tree[tensordict] = node

        action = self.tree_policy(node)

        tensordict[self.action_key] = action
        visualize(tensordict)
        return tensordict

    def start_simulation(self):
        self.tree.clear()

    def update(self, rollout: TensorDictBase) -> None:
        next_state_value = torch.stack(
            [
                torch.sum(
                    rollout[i][self.action_key] * self.tree[rollout[i]]["q_sa"],
                    dim=0,
                    keepdim=True,
                )
                for i in range(1, rollout.shape[-1])
            ]
            + [torch.zeros(1)],
            dim=0,
        )
        rollout[("next", self.value_estimator.value_key)] = next_state_value
        value_estimator_input = rollout.unsqueeze(dim=0)
        target_value = self.value_estimator.value_estimate(value_estimator_input)
        target_value = target_value.squeeze(dim=0)

        for idx in range(rollout.batch_size[0]):
            state = rollout[idx, ...]
            node = self.tree[state]
            action = state["action"]
            mask = (node["n_sa"] + action) > 0
            node["q_sa"][mask] = (
                node["q_sa"] * node["n_sa"] + target_value[idx, ...] * action
            )[mask] / (node["n_sa"] + action)[mask]
            node["n_sa"] += action


@dataclass
class SimulatedSearchPolicy:
    policy: MctsPolicy
    env: EnvBase
    num_simulation: int
    max_steps: int

    def __call__(self, tensordict: TensorDictBase):
        with torch.no_grad():
            self.policy.start_simulation()
            for i in range(self.num_simulation):
                self.simulate(tensordict)

            with set_exploration_type(ExplorationType.MODE):
                tensordict = self.policy(tensordict)
            return tensordict

    def simulate(self, tensordict: TensorDictBase):
        tensordict = tensordict.clone(False)
        visualize(tensordict)
        rollout = self.env.rollout(
            max_steps=self.max_steps,
            policy=self.policy,
            tensordict=tensordict,
            auto_reset=False,
        )
        self.policy.update(rollout)
