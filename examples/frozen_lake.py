import torch
from torchrl.envs import GymEnv, step_mdp

from torchrl_env.stateless_frozenlake import StatelessFrozenLake


def main():
    env = StatelessFrozenLake(render_mode="ansi", is_slippery=False)

    state = env.reset()

    while True:
        print(env.render())
        action = input("Please provide your input (empty input to exit, -1 to reset):")
        if not action:
            break

        action = int(action)
        if action == -1:
            state = env.reset()
            continue
        state["action"] = torch.zeros((4,))
        state["action"][action] = 1

        state = env.step(state)
        for key in (("next", "reward"), ("next", "done")):
            print(f"{key}: {state[key].detach().numpy()}")

        state = step_mdp(state)


if __name__ == "__main__":
    main()
