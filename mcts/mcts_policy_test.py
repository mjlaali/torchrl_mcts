import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import GymEnv, TransformedEnv, Compose, DTypeCastTransform, Transform
from torchrl.objectives.value import TDLambdaEstimator

from mcts.mcts_policy import (
    MctsPolicy,
    ActionSelectionPolicy,
    ZeroExpansion,
    SimulatedSearchPolicy,
    ucb_1,
)
from mcts.tensordict_map import TensorDictMap


def test_explore_action_breaks_ties():
    torch.manual_seed(1)

    policy = ActionSelectionPolicy(exploration_strategy=ucb_1)
    zeros = torch.zeros((10,))
    node = TensorDict(
        {
            "q_sa": zeros.to(torch.float32),
            "p_sa": zeros.to(torch.float32),
            "n_sa": zeros,
        },
        batch_size=(),
    )
    first_action = torch.argmax(policy.explore_action(node), dim=-1).item()
    second_action = torch.argmax(policy.explore_action(node), dim=-1).item()

    assert first_action != second_action


def test_one_step():
    env = GymEnv("CliffWalking-v0")

    state = env.reset()

    rollout_policy = ZeroExpansion(action_spec=env.action_spec)
    value_estimator = TDLambdaEstimator(gamma=1.0, lmbda=1.0, value_network=None)

    mcts_policy = MctsPolicy(
        tree=TensorDictMap("observation"),
        tree_policy=ActionSelectionPolicy(exploration_strategy=ucb_1),
        rollout_policy=rollout_policy,
        value_estimator=value_estimator,
        action_key=env.action_key,
    )

    state_action = mcts_policy(state)

    assert "action" in state_action.keys()


def test_rollout() -> None:
    env = GymEnv("CliffWalking-v0")

    rollout_policy = ZeroExpansion(action_spec=env.action_spec)
    value_estimator = TDLambdaEstimator(gamma=1.0, lmbda=1.0, value_network=None)

    mcts_policy = MctsPolicy(
        tree=TensorDictMap("observation"),
        tree_policy=ActionSelectionPolicy(exploration_strategy=ucb_1),
        rollout_policy=rollout_policy,
        value_estimator=value_estimator,
        action_key=env.action_key,
    )

    env.rollout(policy=mcts_policy, max_steps=2)
    assert len(mcts_policy.tree) > 0


def test_simulated_search_policy():
    torch.manual_seed(1)
    env = GymEnv("CliffWalking-v0")

    policy = SimulatedSearchPolicy(
        policy=MctsPolicy(
            tree=TensorDictMap("observation"),
            tree_policy=ActionSelectionPolicy(exploration_strategy=ucb_1),
            rollout_policy=ZeroExpansion(env.action_spec),
            value_estimator=TDLambdaEstimator(gamma=1.0, lmbda=1.0, value_network=None),
            action_key=env.action_key,
        ),
        env=env,
        num_simulation=10,
        max_steps=3,
    )

    rollout = env.rollout(
        max_steps=30,
        policy=policy,
    )
    assert torch.min(rollout[("next", "reward")]).item() == -1


def test_ucb_1():
    res = ucb_1(
        TensorDict(
            {"q_sa": torch.Tensor([0.5, 0.5]), "n_sa": torch.Tensor([1, 2])},
            batch_size=(),
        )
    )

    idx = torch.argmax(res, dim=-1)
    assert idx.item() == 0

    res = ucb_1(
        TensorDict(
            {"q_sa": torch.Tensor([0.6, 0.5]), "n_sa": torch.Tensor([1, 1])},
            batch_size=(),
        )
    )

    idx = torch.argmax(res, dim=-1)
    assert idx.item() == 0
