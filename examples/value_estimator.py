from typing import cast

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import SelectTransform
from torchrl.modules import QValueActor
from torchrl.objectives.value import TDLambdaEstimator


def main():
    value_network = TensorDictModule(
        module=lambda x: x,
        in_keys=["objective"],
        out_keys=["state_value"],
    )
    estimator = TDLambdaEstimator(
        gamma=1, lmbda=0, value_network=value_network, vectorized=False
    )

    td_template = TensorDict(
        {
            "objective": torch.Tensor([1]),
            "next": {
                "objective": torch.Tensor([1]),
                "reward": torch.Tensor([1]),
                "done": torch.Tensor([0]).to(torch.bool),
            },
        },
        batch_size=(),
    )

    step_a = td_template.clone(True)
    step_b = td_template.clone(True)
    step_b[("next", "done")] = torch.Tensor([1]).to(torch.bool)

    steps = cast(TensorDict, torch.stack([step_a, step_b], dim=0))

    res = estimator(steps)
    print(res)


if __name__ == "__main__":
    main()
