import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    TransformedEnv,
    Compose,
    DTypeCastTransform,
    StepCounter,
    EnvBase,
    step_mdp,
)

from mcts.mcts_policy import (
    SimulatedSearchPolicy,
    MctsPolicy,
    UpdateTreeStrategy,
    AlphaZeroExpansionStrategy,
    PuctSelectionPolicy,
)
from mcts.stateless_frozenlake import StatelessFrozenLake
from mcts.tensordict_map import TensorDictMap


def make_env() -> EnvBase:
    return TransformedEnv(
        StatelessFrozenLake(render_mode="ansi", is_slippery=False),
        Compose(
            DTypeCastTransform(
                dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]
            ),
            StepCounter(),
        ),
    )


def main():
    torch.manual_seed(1)
    env = make_env()
    tree = TensorDictMap(["observation", "step_count"])

    mcts_policy = MctsPolicy(
        expansion_strategy=AlphaZeroExpansionStrategy(
            tree=tree,
            value_module=TensorDictModule(
                module=lambda x: torch.ones((4,)) * 1.0 / 4,
                in_keys=["observation"],
                out_keys=["action_value"],
            ),
        ),
        selection_strategy=PuctSelectionPolicy(),
    )
    policy = SimulatedSearchPolicy(
        policy=mcts_policy,
        tree_updater=UpdateTreeStrategy(
            tree,
        ),
        env=env,
        num_simulation=2000,
        max_steps=100,
    )

    def render(an_env: EnvBase, next_state: TensorDictBase) -> None:
        print("\n------------")
        for k in (
            "action_value",
            "action_count",
            "action_value",
            "action",
            "done",
            "step_count",
        ):
            if k in next_state.keys():
                print(f"{k}:\n{next_state[k].detach().numpy()}")

        print(an_env.render())

    env.reset()
    rollout = env.rollout(policy=policy, max_steps=20, callback=render)
    last_state = step_mdp(rollout[-1])
    render(env, last_state)


if __name__ == "__main__":
    main()
