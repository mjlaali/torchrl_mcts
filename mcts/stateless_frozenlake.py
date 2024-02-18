import torch
from tensordict import TensorDictBase
from torchrl.envs import GymEnv


class StatelessFrozenLake(GymEnv):
    def __init__(self, *args, **kwargs):
        super().__init__("FrozenLake-v1", *args, **kwargs)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        s = torch.argmax(tensordict["observation"])
        self.env.s = s.item()
        return super()._step(tensordict)
