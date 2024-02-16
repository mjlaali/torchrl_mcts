import torch
from tensordict import TensorDict

from mcts.tensordict_map import TensorDictMap


def test_td_dict():
    td = TensorDict(
        {"key": torch.Tensor([1, 2]), "dummy": torch.Tensor([2, 3])}, batch_size=()
    )

    td_dict = TensorDictMap(["key"])
    td_dict[td] = TensorDict({"value": torch.Tensor([True])}, batch_size=())

    td = TensorDict(
        {"key": torch.Tensor([1, 2]), "dummy": torch.Tensor([3, 4])}, batch_size=()
    )

    td_none = TensorDict(
        {"key": torch.Tensor([2, 1]), "dummy": torch.Tensor([3, 4])}, batch_size=()
    )

    assert td in td_dict
    assert len(td_dict) == 1
    assert td_none not in td_dict
    assert td_dict.get(td_none) is None

    td_dict.clear()
    assert len(td_dict) == 0
