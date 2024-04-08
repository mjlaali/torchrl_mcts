import torch
from torchrl.envs import (
    TransformedEnv,
    Compose,
    DTypeCastTransform,
    check_env_specs,
    step_mdp,
)

from mcts.transforms import TruncateTrajectory
from torchrl_env.stateless_frozenlake import StatelessFrozenLake


def test_early_trajectory_specs():
    env = TransformedEnv(
        StatelessFrozenLake(render_mode="ansi", is_slippery=False),
        Compose(
            DTypeCastTransform(
                dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]
            ),
            TruncateTrajectory(),
        ),
    )

    check_env_specs(env)


def test_keys_in_output():
    env = TransformedEnv(
        StatelessFrozenLake(render_mode="ansi", is_slippery=False),
        Compose(
            DTypeCastTransform(
                dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]
            ),
            TruncateTrajectory(),
        ),
    )
    state = env.reset()
    assert "terminated" in state.keys()


def test_terminated_key_changes_done():
    env = TransformedEnv(
        StatelessFrozenLake(render_mode="ansi", is_slippery=False),
        Compose(
            DTypeCastTransform(
                dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]
            ),
            TruncateTrajectory(),
        ),
    )

    state = env.reset()
    state["action"] = torch.zeros((4,))
    state["action"][0] = 1

    state = step_mdp(env.step(state))

    assert "done" in state.keys()
    assert not state["done"].item()

    state = env.reset()
    state["action"] = torch.zeros((4,))
    state["action"][0] = 1
    state["truncated"] = torch.ones((1,)).to(torch.bool)

    state = step_mdp(env.step(state))

    assert "done" in state.keys()
    assert state["done"].item()
