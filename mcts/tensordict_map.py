from collections.abc import MutableMapping
from typing import TypeVar, Union, List, Sequence

from tensordict import TensorDictBase, NestedKey

T = TypeVar("T")


class TensorDictMap(MutableMapping[TensorDictBase, TensorDictBase]):
    def __init__(
        self,
        keys: Union[NestedKey, Sequence[NestedKey]],
    ):
        if isinstance(keys, List):
            self.keys = keys
        else:
            self.keys = [keys]

        self._dict = {}

    def __delitem__(self, item: Union[TensorDictBase, int]):
        key = self._get_key(item)
        return self._dict.__delitem__(key)

    def __iter__(self):
        return self._dict.__iter__()

    def _get_key(self, item: Union[TensorDictBase, int]) -> int:
        if isinstance(item, TensorDictBase):
            hash_values = []
            for key in self.keys:
                val = item[key].detach().numpy()
                hash_values.append(hash(val.tobytes()))
            hash_val = hash(tuple(hash_values))
        elif isinstance(item, int):
            hash_val = item
        else:
            raise ValueError(f"{type(item)} is not supported.")

        return hash_val

    def __getitem__(self, item: Union[TensorDictBase, int]) -> TensorDictBase:
        key = self._get_key(item)
        return self._dict[key]

    def __setitem__(self, item: TensorDictBase, value: TensorDictBase):
        key = self._get_key(item)
        self._dict[key] = value

    def __len__(self):
        return len(self._dict)
