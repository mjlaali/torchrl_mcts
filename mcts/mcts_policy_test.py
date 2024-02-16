import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.envs import (
    GymEnv,
)
from torchrl.modules import QValueActor, MLP

from mcts.mcts_policy import (
    MctsPolicy,
    ZeroExpansion,
    SimulatedSearchPolicy,
    UcbSelectionPolicy,
    ActionExplorationModule,
    UpdateTreeStrategy,
    AlphaZeroExpansionStrategy,
    PucbSelectionPolicy,
)
from mcts.tensordict_map import TensorDictMap


def test_explore_action_breaks_ties():
    torch.manual_seed(1)

    n = 10
    policy = ActionExplorationModule()
    zeros = torch.zeros((n,))
    node = TensorDict(
        {
            "action_value": zeros.to(torch.float32),
        },
        batch_size=(),
    )

    action_value = policy.explore_action(node)
    first_action = torch.argmax(action_value, dim=-1).item()

    action_value = policy.explore_action(node)
    second_action = torch.argmax(action_value, dim=-1).item()

    assert first_action != second_action


def test_one_step():
    env = GymEnv("CliffWalking-v0")

    state = env.reset()

    rollout_policy = ZeroExpansion(
        tree=TensorDictMap("observation"), action_spec=env.action_spec
    )

    mcts_policy = MctsPolicy(
        expansion_strategy=rollout_policy,
    )

    state_action = mcts_policy(state)

    assert "action" in state_action.keys()


def test_rollout() -> None:
    env = GymEnv("CliffWalking-v0")

    rollout_policy = ZeroExpansion(
        tree=TensorDictMap("observation"), action_spec=env.action_spec
    )

    mcts_policy = MctsPolicy(
        expansion_strategy=rollout_policy,
    )

    env.rollout(policy=mcts_policy, max_steps=2)
    assert len(rollout_policy.tree) > 0


def test_simulated_search_policy():
    torch.manual_seed(1)
    env = GymEnv("CliffWalking-v0")

    tree = TensorDictMap("observation")
    policy = SimulatedSearchPolicy(
        policy=MctsPolicy(
            expansion_strategy=ZeroExpansion(tree=tree, action_spec=env.action_spec),
        ),
        tree_updater=UpdateTreeStrategy(tree),
        env=env,
        num_simulation=10,
        max_steps=3,
    )

    rollout = env.rollout(
        max_steps=30,
        policy=policy,
    )
    assert torch.min(rollout[("next", "reward")]).item() == -1


def test_ucb_selection():
    ucb = UcbSelectionPolicy()

    res = ucb(
        TensorDict(
            {"q_sa": torch.Tensor([0.5, 0.5]), "n_sa": torch.Tensor([1, 2])},
            batch_size=(),
        )
    )

    idx = torch.argmax(res["action_value"], dim=-1)
    assert idx.item() == 0

    res = ucb(
        TensorDict(
            {"q_sa": torch.Tensor([0.6, 0.5]), "n_sa": torch.Tensor([1, 1])},
            batch_size=(),
        )
    )

    idx = torch.argmax(res["action_value"], dim=-1)
    assert idx.item() == 0


def test_alpha_zero_expansion():
    value_module = QValueActor(
        MLP(in_features=5, out_features=7),
        in_keys="observation",
        action_space="one-hot",
    )
    alpha_zero_expansion = AlphaZeroExpansionStrategy(
        tree=TensorDictMap("observation"),
        value_module=value_module,
    )

    tensordict = TensorDict(
        {"observation": torch.Tensor([[0.1 * i for i in range(5)]])}, batch_size=(1,)
    )
    tensordict = alpha_zero_expansion(tensordict)

    assert "q_sa" in tensordict.keys()
    assert "p_sa" in tensordict.keys()
    assert "n_sa" in tensordict.keys()

    np.testing.assert_almost_equal(
        tensordict["p_sa"].detach().numpy(), tensordict["action_value"].detach().numpy()
    )


@pytest.mark.parametrize(
    "tensordict,expected_action",
    [
        (
            TensorDict(
                {
                    "q_sa": torch.Tensor([0.1, 0.1]),
                    "p_sa": torch.Tensor([0.1, 0.1]),
                    "n_sa": torch.Tensor([1, 2]),
                },
                batch_size=(),
            ),
            0,
        ),
        (
            TensorDict(
                {
                    "q_sa": torch.Tensor([0.1, 0.1]),
                    "p_sa": torch.Tensor([0.2, 0.1]),
                    "n_sa": torch.Tensor([1, 1]),
                },
                batch_size=(),
            ),
            0,
        ),
        (
            TensorDict(
                {
                    "q_sa": torch.Tensor([0.1, 0.2]),
                    "p_sa": torch.Tensor([0.1, 0.1]),
                    "n_sa": torch.Tensor([1, 1]),
                },
                batch_size=(),
            ),
            1,
        ),
    ],
)
def test_pucb_selection_policy(tensordict: TensorDict, expected_action: int):
    pucb = PucbSelectionPolicy()

    res = pucb(tensordict)

    assert "action_value" in res.keys()

    assert torch.argmax(res["action_value"]).item() == expected_action
