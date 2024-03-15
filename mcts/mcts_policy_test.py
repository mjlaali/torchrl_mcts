from typing import cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import QValueActor, MLP

from mcts.mcts_policy import (
    MctsPolicy,
    SimulatedSearchPolicy,
    UcbSelectionPolicy,
    ActionExplorationModule,
    UpdateTreeStrategy,
    AlphaZeroExpansionStrategy,
    PuctSelectionPolicy,
    safe_weighted_avg,
)
from examples.stateless_cliffwalking import StatelessCliffWalking
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

    expansion_strategy = AlphaZeroExpansionStrategy(
        tree=TensorDictMap("observation"),
        value_module=TensorDictModule(
            module=lambda x: torch.zeros(env.action_spec.shape),
            in_keys=["observation"],
            out_keys=["action_value"],
        ),
    )

    mcts_policy = MctsPolicy(
        expansion_strategy=expansion_strategy,
    )

    state_action = mcts_policy(state)

    assert "action" in state_action.keys()


def test_rollout() -> None:
    env = StatelessCliffWalking()

    rollout_policy = AlphaZeroExpansionStrategy(
        tree=TensorDictMap("observation"),
        value_module=TensorDictModule(
            module=lambda x: torch.zeros(env.action_spec.shape),
            in_keys=["observation"],
            out_keys=["action_value"],
        ),
    )

    mcts_policy = MctsPolicy(
        expansion_strategy=rollout_policy,
    )

    env.rollout(policy=mcts_policy, max_steps=2)
    assert len(rollout_policy.tree) > 0


def test_simulated_search_policy():
    torch.manual_seed(1)
    env = StatelessCliffWalking()

    tree = TensorDictMap("observation")
    policy = SimulatedSearchPolicy(
        policy=MctsPolicy(
            expansion_strategy=AlphaZeroExpansionStrategy(
                tree=tree,
                value_module=TensorDictModule(
                    module=lambda x: torch.zeros(env.action_spec.shape),
                    in_keys=["observation"],
                    out_keys=["action_value"],
                ),
            ),
        ),
        tree_updater=UpdateTreeStrategy(tree),
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
    tree_updater = UpdateTreeStrategy(tree=tree)

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
    "action_value,action_count,current_action,target_value,expected_output",
    [
        (
            torch.Tensor([1, 1]),  # initialized values
            torch.Tensor([0, 0]),  # the count is zero as it just initialized
            torch.Tensor([1, 0]),  # one-hot encoded value of new action
            torch.Tensor([2]),  # target value (total reward) from selecting this action
            torch.Tensor([2.0, 1]),
        ),
        (
            torch.Tensor([1.5, 1]),  # next round after initialization
            torch.Tensor([1, 0]),  # first action only explored
            torch.Tensor([1, 0]),  # first action get selected as it has higher value
            torch.Tensor([-2.5]),  # negative reward
            torch.Tensor([-0.5, 1]),
        ),
        (
            torch.Tensor([-0.5, 1]),  # last round
            torch.Tensor([2, 0]),  # second action has not been explored
            torch.Tensor([0, 1]),  # second action selected
            torch.Tensor([0.5]),  # reward of second action
            torch.Tensor([-0.5, 0.5]),
        ),
    ],
)
def test_weighted_sum(
    action_value: torch.Tensor,
    action_count: torch.Tensor,
    current_action: torch.Tensor,
    target_value: torch.Tensor,
    expected_output: torch.Tensor,
):
    avg = safe_weighted_avg(action_count, action_value, current_action, target_value)
    np.testing.assert_almost_equal(
        avg.detach().numpy(), expected_output.detach().numpy()
    )
