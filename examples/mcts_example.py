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
from torchrl.modules import QValueActor

from mcts.mcts_policy import (
    SimulatedSearchPolicy,
    MctsPolicy,
    UpdateTreeStrategy,
    AlphaZeroExpansionStrategy,
    PuctSelectionPolicy,
    ConstantValueExpansion,
    UcbSelectionPolicy,
)
from mcts.stateless_frozenlake import StatelessFrozenLake
from mcts.tensordict_map import TensorDictMap


def make_q_value(num_observation, num_action, action_space):
    net = torch.nn.Linear(num_observation, num_action)
    qvalue_module = QValueActor(net, in_keys=["observation"], action_space=action_space)
    return qvalue_module


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
    # qvalue_module = make_q_value(
    #     env.observation_spec["observation"].shape[-1],
    #     env.action_spec.shape[-1],
    #     env.action_spec,
    # )
    # print(qvalue_module(env.reset()))

    # loss_module = DQNLoss(qvalue_module, action_space=env.action_spec)

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
        env=make_env(),
        num_simulation=10000,
        max_steps=10,
    )

    # res = policy(env.reset())

    # for k in ("action_value", "action_count", "action_value", "action"):
    #     print(f"{k}:\n{res[k].detach().numpy()}")
    #
    # print("start exploring")

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

    rollout = env.rollout(policy=policy, max_steps=4, callback=render)

    last_state = step_mdp(rollout[-1])
    render(env, last_state)

    # print("rollout actions")
    # print(rollout["action"])
    #
    # print("reward")
    # print(rollout[("next", "reward")])


if __name__ == "__main__":
    main()
