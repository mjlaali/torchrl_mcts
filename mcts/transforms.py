import torch
from tensordict import TensorDictBase, unravel_key
from torchrl.data import CompositeSpec, BoundedTensorSpec, DiscreteTensorSpec
from torchrl.envs.transforms import Transform


class TruncateTrajectory(Transform):
    """Terminate a trajectory based on a key in the input tensordict

    The input tensordict is the output tensordict of env.step(...)

    """

    invertible = False

    def __init__(
        self,
        truncated_key: str | None = "truncated",
    ):
        if not isinstance(truncated_key, str):
            raise ValueError("truncated_key must be a string.")
        self.truncated_key = truncated_key
        super().__init__()

    @property
    def truncated_keys(self):
        truncated_keys = self.__dict__.get("_truncated_keys", None)
        if truncated_keys is None:
            # make the default truncated keys
            truncated_keys = []
            for reset_key in self.parent._filtered_reset_keys:
                if isinstance(reset_key, str):
                    key = self.truncated_key
                else:
                    key = (*reset_key[:-1], self.truncated_key)
                truncated_keys.append(key)
        self._truncated_keys = truncated_keys
        return truncated_keys

    @property
    def done_keys(self):
        done_keys = self.__dict__.get("_done_keys", None)
        if done_keys is None:
            # make the default done keys
            done_keys = []
            for reset_key in self.parent._filtered_reset_keys:
                if isinstance(reset_key, str):
                    key = "done"
                else:
                    key = (*reset_key[:-1], "done")
                done_keys.append(key)
        self.__dict__["_done_keys"] = done_keys
        return done_keys

    @property
    def terminated_keys(self):
        terminated_keys = self.__dict__.get("_terminated_keys", None)
        if terminated_keys is None:
            # make the default terminated keys
            terminated_keys = []
            for reset_key in self.parent._filtered_reset_keys:
                if isinstance(reset_key, str):
                    key = "terminated"
                else:
                    key = (*reset_key[:-1], "terminated")
                terminated_keys.append(key)
        self.__dict__["_terminated_keys"] = terminated_keys
        return terminated_keys

    @property
    def reset_keys(self):
        if self.parent is not None:
            return self.parent._filtered_reset_keys
        # fallback on default "_reset"
        return ["_reset"]

    @property
    def full_done_spec(self):
        return self.parent.output_spec["full_done_spec"] if self.parent else None

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for truncated_key, done_key, terminated_key in zip(
            self.truncated_keys,
            self.done_keys,
            self.terminated_keys,
        ):
            truncated = tensordict.get(truncated_key, None)
            if truncated is None:
                continue

            done = next_tensordict.get(done_key, None)
            terminated = next_tensordict.get(terminated_key, None)
            if terminated is not None:
                truncated = truncated.to(torch.bool) & ~terminated
            done = truncated | done  # we assume no done after reset
            next_tensordict.set(done_key, done)
            next_tensordict.set(truncated_key, truncated)
        return next_tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            "StepCounter cannot be called independently, only its step and reset methods "
            "are functional. The reason for this is that it is hard to consider using "
            "StepCounter with non-sequential data, such as those collected by a replay buffer "
            "or a dataset. If you need StepCounter to work on a batch of sequential data "
            "(ie as LSTM would work over a whole sequence of data), file an issue on "
            "TorchRL requesting that feature."
        )
