import torch
from tensordict import TensorDict
from torchrl.envs import GymEnv

from mcts.mcts_policy import (
    MctsPolicy,
    ZeroExpansion,
    SimulatedSearchPolicy,
    UcbSelectionPolicy,
    ActionExplorationModule,
    UpdateTreeStrategy,
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
