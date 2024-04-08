import torch
from tensordict import TensorDictBase
from torchrl.envs import GymEnv


class StatelessFrozenLake(GymEnv):
    """
    Actions:
       0 -> Left
       1 -> Down
       2 -> Right
       3 -> Top
    """

    def __init__(self, *args, **kwargs):
        super().__init__("FrozenLake-v1", *args, **kwargs)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.reset()
        s = torch.argmax(tensordict["observation"])
        self.env.env.s = s.item()
        return super()._step(tensordict)
