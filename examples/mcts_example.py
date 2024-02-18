import torch
from torchrl.envs import (
    TransformedEnv,
    Compose,
    DTypeCastTransform,
    StepCounter,
)
from torchrl.modules import QValueActor

from mcts.mcts_policy import (
    SimulatedSearchPolicy,
    MctsPolicy,
    UpdateTreeStrategy,
    AlphaZeroExpansionStrategy,
    PuctSelectionPolicy,
)
from mcts.stateless_cliffwalking import StatelessCliffWalking
from mcts.tensordict_map import TensorDictMap


def make_q_value(num_observation, num_action, action_space):
    net = torch.nn.Linear(num_observation, num_action)
    qvalue_module = QValueActor(net, in_keys=["observation"], action_space=action_space)
    return qvalue_module


def main():
    torch.manual_seed(1)
    env = TransformedEnv(
        StatelessCliffWalking(),
        Compose(
            DTypeCastTransform(
                dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]
            ),
            StepCounter(),
        ),
    )
    qvalue_module = make_q_value(
        env.observation_spec["observation"].shape[-1],
        env.action_spec.shape[-1],
        env.action_spec,
    )
    print(qvalue_module(env.reset()))

    # loss_module = DQNLoss(qvalue_module, action_space=env.action_spec)

    tree = TensorDictMap(["observation", "step_count"])

    policy = SimulatedSearchPolicy(
        policy=MctsPolicy(
            expansion_strategy=AlphaZeroExpansionStrategy(
                value_module=qvalue_module, tree=tree
            ),
            selection_strategy=PuctSelectionPolicy(),
        ),
        tree_updater=UpdateTreeStrategy(
            tree,
        ),
        env=env,
        num_simulation=10,
        max_steps=1000,
    )

    res = policy(env.reset())

    for k in ("action_value", "action_count", "action_value", "action"):
        print(f"{k}:\n{res[k].detach().numpy()}")

    print("start exploring")
    rollout = env.rollout(policy=policy, max_steps=50)

    print("rollout actions")
    print(rollout["action"])

    print("reward")
    print(rollout[("next", "reward")])


if __name__ == "__main__":
    main()
