from typing import cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.envs import (
    GymEnv,
)
from torchrl.modules import QValueActor, MLP

from mcts.stateless_cliffwalking import StatelessCliffWalking
from mcts.mcts_policy import (
    MctsPolicy,
    ZeroExpansion,
    SimulatedSearchPolicy,
    UcbSelectionPolicy,
    ActionExplorationModule,
    UpdateTreeStrategy,
    AlphaZeroExpansionStrategy,
    PuctSelectionPolicy,
    safe_weighted_avg,
)
from mcts.tensordict_map import TensorDictMap


def test_explore_action_breaks_ties():
    torch.manual_seed(1)

    n = 10
    policy = ActionExplorationModule()
    zeros = torch.zeros((n,))
    node = TensorDict(
        {
            "action_value_under_uncertainty": zeros.to(torch.float32),
        },
        batch_size=(),
    )

    action_value = policy.explore_action(node)
    first_action = torch.argmax(action_value, dim=-1).item()

    action_value = policy.explore_action(node)
    second_action = torch.argmax(action_value, dim=-1).item()

    assert first_action != second_action


def test_one_step():
    env = StatelessCliffWalking()

    state = env.reset()

    expansion_strategy = ZeroExpansion(
        tree=TensorDictMap("observation"), num_action=env.action_spec.shape[-1]
    )

    mcts_policy = MctsPolicy(
        expansion_strategy=expansion_strategy,
    )

    state_action = mcts_policy(state)

    assert "action" in state_action.keys()


def test_rollout() -> None:
    env = StatelessCliffWalking()

    rollout_policy = ZeroExpansion(
        tree=TensorDictMap("observation"), num_action=env.action_spec.shape[-1]
    )

    mcts_policy = MctsPolicy(
        expansion_strategy=rollout_policy,
    )

    env.rollout(policy=mcts_policy, max_steps=2)
    assert len(rollout_policy.tree) > 0


def test_simulated_search_policy():
    # TODO: This test fails because the action value of reset state is getting changed between simulation 1 and
    #   simulation 2, the general hypothesis is that the tensor changed in the dict when we explore a new action in this
    #   state in UcbSelectionPolicy
    torch.manual_seed(1)
    env = StatelessCliffWalking()

    tree = TensorDictMap("observation")
    policy = SimulatedSearchPolicy(
        policy=MctsPolicy(
            expansion_strategy=ZeroExpansion(
                tree=tree, num_action=env.action_spec.shape[-1]
            ),
        ),
        tree_updater=UpdateTreeStrategy(tree, num_action=env.action_spec.shape[-1]),
        env=env,
        num_simulation=10,
        max_steps=3,
    )

    rollout = env.rollout(
        max_steps=16,
        policy=policy,
    )
    for idx, v in enumerate(rollout[("next", "reward")].detach().numpy()):
        print(f"{idx}: {v}")
    assert torch.min(rollout[("next", "reward")]).item() == -1


def test_ucb_selection():
    ucb = UcbSelectionPolicy()

    res = ucb(
        TensorDict(
            {
                "action_value": torch.Tensor([0.5, 0.5]),
                "action_count": torch.Tensor([1, 2]),
            },
            batch_size=(),
        )
    )

    idx = torch.argmax(res["action_value"], dim=-1)
    assert idx.item() == 0

    res = ucb(
        TensorDict(
            {
                "action_value": torch.Tensor([0.6, 0.5]),
                "action_count": torch.Tensor([1, 1]),
            },
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

    assert "action_value" in tensordict.keys()
    assert "prior_action_value" in tensordict.keys()
    assert "action_count" in tensordict.keys()

    np.testing.assert_almost_equal(
        tensordict["prior_action_value"].detach().numpy(),
        tensordict["action_value"].detach().numpy(),
    )


@pytest.mark.parametrize(
    "tensordict,expected_action",
    [
        (
            TensorDict(
                {
                    "action_value": torch.Tensor([0.1, 0.1]),
                    "prior_action_value": torch.Tensor([0.1, 0.1]),
                    "action_count": torch.Tensor([1, 2]),
                },
                batch_size=(),
            ),
            0,
        ),
        (
            TensorDict(
                {
                    "action_value": torch.Tensor([0.1, 0.1]),
                    "prior_action_value": torch.Tensor([0.2, 0.1]),
                    "action_count": torch.Tensor([1, 1]),
                },
                batch_size=(),
            ),
            0,
        ),
        (
            TensorDict(
                {
                    "action_value": torch.Tensor([0.1, 0.2]),
                    "prior_action_value": torch.Tensor([0.1, 0.1]),
                    "action_count": torch.Tensor([1, 1]),
                },
                batch_size=(),
            ),
            1,
        ),
    ],
)
def test_puct_selection_policy(tensordict: TensorDict, expected_action: int):
    puct = PuctSelectionPolicy()

    res = puct(tensordict)

    assert "action_value" in res.keys()

    assert torch.argmax(res["action_value"]).item() == expected_action


def test_update_tree():
    tree = TensorDictMap("observation")
    tree_updater = UpdateTreeStrategy(tree=tree, num_action=2)

    done_state = TensorDict(
        {
            "observation": torch.Tensor([1]),
            "action_value": torch.Tensor([0.0, 0.0]),
            "action": torch.Tensor([0, 1]),
            "chosen_action_value": torch.Tensor([2.0]),
            "action_count": torch.Tensor([0, 0]),
            "next": {
                "observation": torch.Tensor([2]),
                "reward": torch.Tensor([1.0]),
                "done": torch.Tensor([1]).to(torch.bool),
            },
        },
        batch_size=(),
    )

    tree_updater.start_simulation()
    tree[done_state] = done_state
    tree_updater.update(done_state.unsqueeze(dim=0))

    np.testing.assert_equal(tree[done_state]["action_count"].detach().numpy(), [0, 1])
    np.testing.assert_equal(
        tree[done_state]["action_value"].detach().numpy(), [0.0, 1.0]
    )

    init_state = TensorDict(
        {
            "observation": torch.Tensor([0]),
            "action_value": torch.Tensor([0.0, 1.0]),
            "action_count": torch.Tensor([1, 1]),
            "action": torch.Tensor([1, 0]),
            "next": {
                "reward": torch.Tensor([2.0]),
                "observation": done_state["observation"],
                "done": torch.Tensor([0]).to(torch.bool),
            },
        },
        batch_size=(),
    )

    # noinspection PyTypeChecker
    rollout = cast(TensorDict, torch.stack((init_state, done_state), dim=0))
    tree_updater.start_simulation()
    tree[init_state] = init_state
    tree[done_state] = done_state
    tree_updater.update(rollout)

    np.testing.assert_equal(tree[init_state]["action_count"].detach().numpy(), [2, 1])
    total_reward = init_state[("next", "reward")] + done_state[("next", "reward")]
    np.testing.assert_equal(
        tree[init_state]["action_value"].detach().numpy(), [total_reward.item() / 2, 1]
    )


@pytest.mark.parametrize(
    "w1,v1,w2,v2,expected_output",
    [
        (
            torch.Tensor([0, 1]),
            torch.Tensor([0, 2]),
            torch.Tensor([0, 0]),
            torch.Tensor([1, 2]),
            torch.Tensor([0, 2]),
        ),
        (
            torch.Tensor([0, 1]),
            torch.Tensor([0, 2]),
            torch.Tensor([1, 1]),
            torch.Tensor([1, 1]),
            torch.Tensor([1, 1.5]),
        ),
        (
            torch.Tensor([0, 1]),
            torch.Tensor([0, 2]),
            torch.Tensor([0, 1]),
            torch.Tensor([1]),
            torch.Tensor([0, 1.5]),
        ),
    ],
)
def test_weighted_sum(
    w1: torch.Tensor,
    v1: torch.Tensor,
    w2: torch.Tensor,
    v2: torch.Tensor,
    expected_output: torch.Tensor,
):
    avg = safe_weighted_avg(w1, v1, w2, v2)
    np.testing.assert_almost_equal(
        avg.detach().numpy(), expected_output.detach().numpy()
    )
