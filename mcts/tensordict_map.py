from collections.abc import MutableMapping
from typing import TypeVar, Tuple, Union

from tensordict import TensorDictBase

T = TypeVar("T")


class TensorDictMap(MutableMapping[TensorDictBase, TensorDictBase]):
    def __init__(
        self,
        key: Union[str, Tuple[str, ...]],
    ):
        self.key = key
        self._dict = {}

    def __delitem__(self, item: Union[TensorDictBase, int]):
        key = self._get_key(item)
        return self._dict.__delitem__(key)

    def __iter__(self):
        return self._dict.__iter__()

    def _get_key(self, item: Union[TensorDictBase, int]) -> int:
        if isinstance(item, TensorDictBase):
            observation = item[self.key].detach().numpy()
            key = hash(observation.tobytes())
        elif isinstance(item, int):
            key = item
        else:
            raise ValueError(f"{type(item)} is not supported.")

        return key

    def __getitem__(self, item: Union[TensorDictBase, int]) -> TensorDictBase:
        key = self._get_key(item)
        return self._dict[key]

    def __setitem__(self, item: TensorDictBase, value: TensorDictBase):
        key = self._get_key(item)
        self._dict[key] = value

    def __len__(self):
        return len(self._dict)
