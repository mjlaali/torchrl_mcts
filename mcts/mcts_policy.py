from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.data import TensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.utils import exploration_type, ExplorationType, set_exploration_type
from torchrl.objectives.value import ValueEstimatorBase, TDLambdaEstimator

from mcts.tensordict_map import TensorDictMap


class UpdateTreeStrategy:
    """
    The strategy to update tree after each rollout. This class uses the given value estimator
    to compute a target value after each roll out and compute the mean of target values in the tree.

    It also updates the number of time nodes get visited in tree.

    Args:
        tree: A TensorDictMap that store stats of the tree.
        value_estimator: A ValueEstimatorBase that compute target value.
        action_key: A key in the rollout TensorDict to store the selected action.
        q_sa_key: A key in the tree nodes that stores the mean of Q(s, a).
        n_sa_key: A key in the tree nodes that stores the number of times nodes get visited.
    """

    def __init__(
        self,
        tree: TensorDictMap,
        value_estimator: ValueEstimatorBase = TDLambdaEstimator(
            gamma=1.0, lmbda=1.0, value_network=None
        ),
        action_key: str = "action",
        q_sa_key: str = "q_sa",
        n_sa_key: str = "n_sa",
    ):
        self.tree = tree
        self.action_key = action_key
        self.q_sa_key = q_sa_key
        self.n_sa_key = n_sa_key
        self.value_estimator = value_estimator

    def update(self, rollout: TensorDictBase) -> None:
        tree = self.tree
        n_sa_key = self.n_sa_key
        q_sa_key = self.q_sa_key

        next_state_value = torch.stack(
            [
                torch.sum(
                    rollout[i][self.action_key] * tree[rollout[i]][q_sa_key],
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
            node = tree[state]
            action = state[self.action_key]
            mask = (node[n_sa_key] + action) > 0
            node[q_sa_key][mask] = (
                node[q_sa_key] * node[n_sa_key] + target_value[idx, ...] * action
            )[mask] / (node[n_sa_key] + action)[mask]
            node[n_sa_key] += action

    def start_simulation(self):
        self.tree.clear()


class ExpansionStrategy(TensorDictModuleBase):
    """
    The rollout policy in expanding tree.
    This policy will use to initialize a node when it gets expanded at the first time.
    """

    def __init__(
        self,
        tree: TensorDictMap,
        out_keys: List[str],
        in_keys: Optional[List[str]] = None,
    ):
        self.in_keys = list(set([tree.key] + [] if in_keys is None else in_keys))
        self.out_keys = out_keys
        super().__init__()
        self.tree = tree

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        The node to be expanded. The output Tensordict will be used in future
        to select action.
        Args:
            tensordict: The state that need to be explored

        Returns:
            A initialized statistics to select actions in the future.
        """
        node = self.tree.get(tensordict)
        if node is None:
            node = self.expand(tensordict)
            self.tree[tensordict] = node
            return node

        return self.expand(node)

    @abstractmethod
    def expand(self, tensordict: TensorDictBase) -> TensorDictBase:
        pass


class ZeroExpansion(ExpansionStrategy):
    """
    A rollout policy to initialize a state with zero Q(s, a).
    """

    def __init__(
        self,
        tree: TensorDictMap,
        action_spec: TensorSpec,
        q_sa_key: str = "q_sa",
        n_sa_key: str = "n_sa",
    ):
        super().__init__(tree=tree, out_keys=[q_sa_key, n_sa_key])
        self.action_spec = action_spec
        self.q_sa_key = q_sa_key
        self.n_sa_key = n_sa_key

    def expand(self, tensordict: TensorDictBase) -> TensorDict:
        tensordict = tensordict.clone(False)
        tensordict[self.q_sa_key] = torch.zeros(
            (self.action_spec.shape[-1],), dtype=torch.float32
        )
        tensordict[self.n_sa_key] = torch.zeros(
            (self.action_spec.shape[-1],), dtype=torch.long
        )
        return tensordict


class AlphaZeroExpansionStrategy(ExpansionStrategy):
    """
    An implementation of Alpha Zero to initialize a node at its first time.

    Args:
            value_module: a TensorDictModule to initialize a prior for Q(s, a)
            action_value_key: a key in the output of value_module that contains Q(s, a) values
    """

    def __init__(
        self,
        tree: TensorDictMap,
        value_module: TensorDictModule,
        q_sa_key: str = "q_sa",
        p_sa_key: str = "p_sa",
        n_sa_key: str = "n_sa",
        action_value_key: str = "action_value",
    ):
        super().__init__(
            tree=tree,
            out_keys=value_module.out_keys + [q_sa_key, p_sa_key, n_sa_key],
            in_keys=value_module.in_keys,
        )
        assert action_value_key in value_module.out_keys
        self.action_value_key = action_value_key
        self.q_sa_key = q_sa_key
        self.p_sa_key = p_sa_key
        self.n_sa_key = n_sa_key

    def expand(self, tensordict: TensorDictBase) -> TensorDict:
        module_output = self.value_module(tensordict)
        p_sa = module_output[self.action_value_key]
        module_output[self.q_sa_key] = torch.clone(p_sa)
        module_output[self.p_sa_key] = p_sa
        module_output[self.n_sa_key] = torch.zeros_like(p_sa)
        return module_output


class PucbSelectionPolicy(TensorDictModuleBase):
    """
    puct formula copied from AlphaZero paper.
    https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf

    Args:
        node: A tensordict with keys of
            q_sa representing the mean of Q(s, a) for every action `a` at state `s`.
            p_sa representing the prior of Q(s, a) for every action `a` at state `s`.
            n_sa representing the number of times action `a` is selected at state `s`.

    Returns:
        The optimism under uncertainty estimation computed by the PUCB formula in AlphaZero paper
    """

    def __init__(
        self,
        cpuct: float = 2.0,
        action_value_key: str = "action_value",
        q_sa_key: str = "q_sa",
        p_sa_key: str = "p_sa",
        n_sa_key: str = "n_sa",
    ):
        self.in_keys = [q_sa_key, n_sa_key, p_sa_key]
        self.out_keys = [action_value_key]
        super().__init__()
        self.cpuct = cpuct
        self.q_sa_key = q_sa_key
        self.p_sa_key = p_sa_key
        self.n_sa_key = n_sa_key
        self.action_value_key = action_value_key

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        n_sa = tensordict[self.n_sa_key]
        p_sa = tensordict[self.p_sa_key]
        x_hat = tensordict[self.q_sa_key]

        n = torch.sum(n_sa, dim=-1)
        u_sa = self.cpuct * p_sa * torch.sqrt(n) / (1 + n_sa)

        optimism_estimation = x_hat + u_sa
        tensordict[self.action_value_key] = optimism_estimation
        return tensordict


class UcbSelectionPolicy(TensorDictModuleBase):
    """
    A policy to select an action in every node in the tree using UCB estimation.
    See Section 2.6 Upper-Confidence-Bound Action Selection on
    Sutton, Richard S., and Andrew G. Barto. 2018. “Reinforcement Learning: An Introduction (Second Edition).”
    http://incompleteideas.net/book/RLbook2020.pdf

    Args:
        q_sa_key: The input key representing the mean of Q(s, a) for every action `a` at state `s`. Defaults to ``q_sa``
        n_sa_key: The input key representing the number of times action `a` is selected at state `s`.Defaults to ``n_sa``
        action_value_key: The output key representing estimated action value.
    """

    def __init__(
        self,
        cucb: float = 2.0,
        action_value_key: str = "action_value",
        q_sa_key: str = "q_sa",
        n_sa_key: str = "n_sa",
    ):
        self.in_keys = [q_sa_key, n_sa_key]
        self.out_keys = [action_value_key]
        super().__init__()
        self.cucb = cucb
        self.q_sa_key = q_sa_key
        self.n_sa_key = n_sa_key
        self.action_value_key = action_value_key

    def forward(self, node: TensorDictBase) -> torch.Tensor:
        node = node.clone(False)
        x_hat = node[self.q_sa_key]
        n_sa = node[self.n_sa_key]
        mask = n_sa != 0
        n = torch.sum(n_sa)
        optimism_estimation = x_hat
        optimism_estimation[mask] = x_hat[mask] + self.cucb * torch.sqrt(
            torch.log(n) / n_sa[mask]
        )
        node[self.action_value_key] = optimism_estimation
        return node


class ActionExplorationModule(TensorDictModuleBase):
    def __init__(
        self,
        action_cnt_key: str = "n_sa",
        action_value_key: str = "action_value",
        action_key: str = "action",
    ):
        self.in_keys = [action_cnt_key, action_value_key]
        self.out_keys = [action_key]
        super().__init__()
        self.action_value_key = action_value_key
        self.action_cnt_key = action_cnt_key
        self.action_key = action_key

    def forward(self, tensordict: TensorDictBase):
        tensordict = tensordict.clone(False)

        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            tensordict[self.action_key] = self.explore_action(tensordict)
        elif exploration_type() == ExplorationType.MODE:
            tensordict[self.action_key] = self.get_greedy_action(tensordict)
        return tensordict

    def get_greedy_action(self, node: TensorDictBase) -> torch.Tensor:
        action_cnt_key = self.action_cnt_key
        action = torch.argmax(node[action_cnt_key], dim=-1)
        return torch.nn.functional.one_hot(action, node[action_cnt_key].shape[-1])

    def explore_action(self, node: TensorDictBase) -> torch.Tensor:
        action_value = node[self.action_value_key]

        max_value, _ = torch.max(action_value, dim=-1)
        action = torch.argmax(
            torch.rand_like(action_value) * (action_value == max_value)
        )
        return torch.nn.functional.one_hot(action, action_value.shape[-1])


@dataclass
class MctsPolicy(TensorDictSequential):
    """
    An implementation of MCTS algorithm.

    Args:
        expansion_strategy: a policy to initialize stats of a node at its first visit.
        selection_strategy: a policy to select action in each state
        value_estimator: a value estimator to update stats of node per each complete rollout
        action_key: the action key of environment.

    """

    def __init__(
        self,
        expansion_strategy: ExpansionStrategy,
        selection_strategy: TensorDictModuleBase = UcbSelectionPolicy(),
        action_exploration: ActionExplorationModule = ActionExplorationModule(),
    ):
        super().__init__(expansion_strategy, selection_strategy, action_exploration)


@dataclass
class SimulatedSearchPolicy(TensorDictModuleBase):
    """
    A simulated search policy. In each step, it simulates `n` rollout of maximum steps of `max_steps`
    using the given policy and then choose the best action given the simulation results.

    Args:
        policy: a policy to select action in each simulation rollout.
        env: an environment to simulate a rollout
        num_simulation: the number of simulation
        max_steps: the max steps of each simulated rollout

    """

    def __init__(
        self,
        policy: MctsPolicy,
        tree_updater: UpdateTreeStrategy,
        env: EnvBase,
        num_simulation: int,
        max_steps: int,
    ):
        self.in_keys = policy.in_keys
        self.out_keys = policy.out_keys

        super().__init__()
        self.policy = policy
        self.tree_updater = tree_updater
        self.env = env
        self.num_simulation = num_simulation
        self.max_steps = max_steps

    def forward(self, tensordict: TensorDictBase):
        with torch.no_grad():
            self.tree_updater.start_simulation()
            for i in range(self.num_simulation):
                self.simulate(tensordict)

            with set_exploration_type(ExplorationType.MODE):
                tensordict = self.policy(tensordict)
            return tensordict

    def simulate(self, tensordict: TensorDictBase):
        tensordict = tensordict.clone(False)
        rollout = self.env.rollout(
            max_steps=self.max_steps,
            policy=self.policy,
            tensordict=tensordict,
            auto_reset=False,
        )
        self.tree_updater.update(rollout)
