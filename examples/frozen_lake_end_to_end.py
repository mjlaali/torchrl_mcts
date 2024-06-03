from dataclasses import dataclass
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, cast, Callable, Dict, Any, ClassVar

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential, TensorDictModule, TensorDictModuleBase
from torchrl.collectors import (
    DataCollectorBase,
    SyncDataCollector,
)
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import (
    EnvBase,
    TransformedEnv,
    Compose,
    DTypeCastTransform,
)
from torchrl.envs.utils import ExplorationType, step_mdp
from torchrl.modules import (
    QValueActor,
    EGreedyModule,
    MLP,
)
from torchrl.objectives import DQNLoss, LossModule, ValueEstimators
from torchrl.record import TensorboardLogger
from torchrl.record.loggers import Logger
from torchrl.trainers import (
    Trainer,
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    TrainerHookBase,
)

from mcts.mcts_policy import (
    MctsPolicy,
    AlphaZeroExpansionStrategy,
    PuctSelectionPolicy,
    UpdateTreeStrategy,
    SimulatedSearchPolicy,
    SimulationListener,
    MCEstimator,
)
from mcts.tensordict_map import TensorDictMap
from mcts.transforms import TruncateTrajectory
from torchrl_env.stateless_frozenlake import StatelessFrozenLake


class MctsLossModule(LossModule):
    def __init__(self, value_network: QValueActor):
        super().__init__()
        self.value_network = value_network

        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )

        self.value_network_in_keys = value_network.in_keys

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_copy = tensordict.clone(False)
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_copy)

        action = tensordict.get(self.tensor_keys.action)
        pred_val = td_copy.get(self.tensor_keys.action_value)

        target_value = self.value_estimator(tensordict)["target_value"]
        predicted_value = self.policy(tensordict)["chosen_action_value"]

        loss = None  # L2(target_value and predicted_value) + CE (mcts_policy and predicted policy)


@dataclass
class DebuggerLogger(TrainerHookBase):
    policy: MctsPolicy
    env: EnvBase
    log_pbar: bool = True

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError

    def __call__(self, batch: TensorDictBase):
        # rewards = batch[("next", "reward")]  # batch x steps x 1
        # max_reward = torch.max(rewards)
        action_values = self.policy(self.env.reset())["action_value"]
        log_values = {
            f"a_{i}": action_values[i].item() for i in range(len(action_values))
        }
        # print(json.dumps(log_values))
        log_values.update(
            {
                # "max_reward": max_reward.item(),
                "log_pbar": self.log_pbar,
                # "len_batch": batch.batch_size[0],
            }
        )

        return log_values

    def register(self, trainer: Trainer, name: str = "log_max_reward"):
        trainer.register_module(name, self)
        trainer.register_op(
            "post_steps_log",
            self,
        )


class LogTreeStats(TrainerHookBase, SimulationListener):
    _instance: ClassVar[Optional["LogTreeStats"]] = None

    def __init__(self, tree: TensorDictMap, log_pbar: bool = True):
        self.tree: TensorDictMap = tree
        self.cnt: int = 0
        self.num_simulation: int = 0
        self.log_pbar: bool = log_pbar

    def __new__(cls, *args, **kwargs):
        # Data collector use deep copy to copy its policy. As a result of this
        # an instance of this class that registered to the trainer will be disconnected
        # from trainer, because the new instance is not registered in the trainer.
        # this is a hacky trick to address this issue for now by implementing singleton pattern.

        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def after_simulation(self):
        self.cnt += len(self.tree)
        self.num_simulation += 1

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError()

    def __call__(self, batch: TensorDictBase):
        out = {"log_pbar": self.log_pbar}
        mean_tree_size = 0
        if self.cnt != 0:
            mean_tree_size = self.cnt / self.num_simulation
            self.cnt = 0
            self.num_simulation = 0
        out["tree"] = mean_tree_size

        return out

    def register(self, trainer: Trainer, name: str = "log_tree_stat"):
        trainer.register_module(name, self)
        trainer.register_op(
            "pre_steps_log",
            self,
        )


@dataclass
class Experiment:
    exp_prefix: str = ""
    num_collectors: int = 1
    device: str = "cpu"

    # gamma in value estimator
    gamma: float = 0.99

    # MCTS setting
    num_simulation: int = 100
    simulation_max_step: int = 10

    # The number of total interaction with env
    total_frames: int = 10000
    # A size of data set generated by the current policy interactions
    # processing (training policy on) this data set is one step
    frames_per_batch: int = 1
    # batch size/sample size of the generated data set by policy interactions
    frames_per_mini_batch: int = 1
    # how many times collect random step until use policy to get data.
    init_random_frames: int = frames_per_batch * 0
    # interval (defined for optim step, or batch size) to record test reward
    record_test_interval: int = 1
    # number of sampling batch from the generated data set by policy interactions
    optim_steps_per_batch: int = frames_per_batch // frames_per_mini_batch
    # the number of steps before we update the weights of the policy
    num_steps_to_update: int = 1

    log_interval: int = 10
    eps_init: float = 0.5
    eps_end: float = 0.1

    frame_skip: int = 1

    # the learning rate of the optimizer
    lr = 1e-2
    # weight decay
    wd = 1e-5
    # the beta parameters of Adam
    betas = (0.9, 0.999)

    loss_module: Optional[DQNLoss] = None
    truncate_key: str = "explored"

    actor: Optional[TensorDictModuleBase] = None
    actor_explore: Optional = None

    make_actor_explore: Callable[
        [EnvBase, TensorDictModuleBase, TensorDictModuleBase], TensorDictModuleBase
    ] = None

    logger: Optional[Logger] = None
    log_tree_stats: Optional[LogTreeStats] = None

    def __post_init__(self):
        # self.make_actor_explore = self.make_default_explore_actor
        # self.exp_prefix = "sarsa-"
        self.make_actor_explore = self.make_mcts_actor
        self.exp_prefix = "mcts-"
        self.logger = self.make_logger()

    def make_env(
        self,
    ) -> EnvBase:
        transforms = (
            DTypeCastTransform(
                dtype_in=torch.long,
                dtype_out=torch.float32,
                in_keys=["observation"],
            ),
        )

        if self.make_actor_explore == self.make_mcts_actor:
            transforms = transforms + (TruncateTrajectory(self.truncate_key),)
        return TransformedEnv(
            StatelessFrozenLake(render_mode="ansi", is_slippery=False),
            Compose(*transforms),
        )

    def make_mcts_actor(
        self, env: EnvBase, qvalue_net: TensorDictModuleBase, actor: TensorDictModule
    ) -> TensorDictModuleBase:
        tree = TensorDictMap(["observation"])

        mcts_policy = MctsPolicy(
            expansion_strategy=AlphaZeroExpansionStrategy(
                tree=tree,
                explored_flag_key=self.truncate_key,
                value_module=TensorDictModule(
                    module=qvalue_net,
                    in_keys=["observation"],
                    out_keys=["action_value"],
                ),
            ),
            selection_strategy=PuctSelectionPolicy(),
        )

        log_tree_stats = LogTreeStats(tree)

        actor_explore = SimulatedSearchPolicy(
            policy=mcts_policy,
            tree_updater=UpdateTreeStrategy(
                tree,
            ),
            env=env,
            num_simulation=self.num_simulation,
            max_steps=self.simulation_max_step,
            listeners=[log_tree_stats],
        )
        self.log_tree_stats = log_tree_stats
        return actor_explore

    def make_default_explore_actor(
        self,
        env: EnvBase,
        qvalue_net: TensorDictModuleBase,
        actor: TensorDictModuleBase,
    ) -> TensorDictModuleBase:
        # noinspection PyTypeChecker
        return TensorDictSequential(
            actor,
            EGreedyModule(
                spec=env.action_spec,
                annealing_num_steps=self.total_frames,
                eps_init=self.eps_init,
                eps_end=self.eps_end,
                action_key=env.action_key,
            ),
        )

    def make_model(
        self, env: EnvBase
    ) -> Tuple[TensorDictModuleBase, TensorDictModuleBase]:
        num_inputs = env.observation_spec["observation"].shape[-1]
        num_actions = env.action_spec.shape[-1]

        qvalue_net = MLP(
            out_features=num_actions,
            activation_class=torch.nn.Sigmoid,
            depth=0,
            num_cells=num_inputs,
            device=self.device,
            activate_last_layer=True,
        )

        actor = QValueActor(
            module=qvalue_net, action_space="one-hot", spec=env.action_spec
        )

        actor_explore = self.make_actor_explore(env, qvalue_net, actor)

        # Test actors
        actor(env.reset())
        actor_explore(env.reset())

        return cast(TensorDictModule, actor), cast(TensorDictModule, actor_explore)

    def make_collector(self, actor_explore: TensorDictModuleBase) -> DataCollectorBase:
        cls = SyncDataCollector

        data_collector = cls(
            self.make_env,
            policy=actor_explore,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            init_random_frames=self.init_random_frames,
            # set the number of the steps to max when policy explores env
            max_frames_per_traj=None,
            # this is the default behaviour: the collector runs in ``"random"`` (or explorative) mode
            exploration_type=ExplorationType.RANDOM,
            # We set the all the devices to be identical. Below is an example of heterogeneous devices
            device=self.device,
            storing_device=self.device,
            split_trajs=False,
        )
        return data_collector

    def make_loss_module(
        self,
        actor: TensorDictModuleBase,
    ) -> LossModule:
        loss_module = DQNLoss(actor)
        loss_module.make_value_estimator(value_type=ValueEstimators.TD1)
        return loss_module

    def make_optimizer(self, loss_module: DQNLoss) -> torch.optim.Adam:
        return torch.optim.AdamW(
            loss_module.parameters(), lr=self.lr, weight_decay=self.wd, betas=self.betas
        )

    def make_logger(self) -> Logger:
        exp_name = f"{self.exp_prefix}{datetime.now().isoformat()}"
        print(f"exp name is {exp_name}")
        logger = TensorboardLogger(exp_name)

        return logger

    def add_loggers(self, trainer: Trainer):
        # Recorde logs test reward after every optim step in the trainer
        # It uses the given policy and execute it the given exploration type,
        # then logs its reward as test_reward
        recorder = Recorder(
            record_interval=self.record_test_interval,  # log every n optimization steps
            record_frames=100,  # maximum number of frames in the record
            frame_skip=1,
            policy_exploration=self.actor,
            environment=self.make_env(),
            exploration_type=ExplorationType.MODE,
            log_keys=[("next", "reward")],
            out_keys={("next", "reward"): "test_reward"},
            log_pbar=True,
        )
        recorder.register(trainer)

        # Compute the mean batch reward that has been collected by the policy
        # at pre optim step
        log_reward = LogReward(
            log_pbar=True, reward_key=("next", "reward"), logname="train_reward"
        )
        log_reward.register(trainer)

        log_max_reward = DebuggerLogger(self.actor, self.make_env())
        log_max_reward.register(trainer)

        if self.log_tree_stats:
            self.log_tree_stats.register(trainer)

    def make_trainer(self) -> Trainer:
        self.actor, self.actor_explore = self.make_model(self.make_env())
        loss_module = self.make_loss_module(self.actor)

        collector = self.make_collector(self.actor_explore)

        # see https://pytorch.org/rl/reference/trainers.html
        # step can be considered as value improvement.
        # In other words, optimizing loss will improve the value estimation on the batch
        trainer = Trainer(
            collector=collector,
            total_frames=self.total_frames,
            frame_skip=self.frame_skip,
            loss_module=loss_module,
            optimizer=self.make_optimizer(loss_module),
            logger=self.logger,
            optim_steps_per_batch=self.optim_steps_per_batch,
            log_interval=self.log_interval,
        )
        # FIXME: It seems trainer._log_dict is not used in the code and may cause memory drainage.

        buffer_hook = ReplayBufferTrainer(
            TensorDictReplayBuffer(
                batch_size=self.frames_per_mini_batch,
                storage=LazyMemmapStorage(self.frames_per_batch),
                prefetch=self.frames_per_mini_batch,
            ),
            flatten_tensordicts=True,
        )
        buffer_hook.register(trainer)

        self.add_loggers(trainer)

        return trainer


def render(env: EnvBase, next_state: TensorDictBase) -> None:
    print("\n------------")
    for k in (
        "action_value",
        "action_count",
        "action_value",
        "action",
        "done",
    ):
        if k in next_state.keys():
            print(f"{k}:\n{next_state[k].detach().numpy()}")
    print(env.render())


def main():
    torch.manual_seed(1)
    exp = Experiment()
    trainer = exp.make_trainer()
    env = exp.make_env()

    print(f"actor before train:\n {exp.actor(env.reset())['action_value']}")
    # print(
    #     f"actor before train:\n {exp.actor_explore(env.reset())['prior_action_value']}"
    # )
    # res = exp.actor_explore(env.reset())
    # for key in res.keys():
    #     print(f"actor_explore before train {key}:\n {res[key]}")

    trainer.train()
    print(f"actor after train:\n {exp.actor(env.reset())['action_value']}")

    env.reset()
    rollout = env.rollout(policy=exp.actor, max_steps=20, callback=render)
    last_state = step_mdp(rollout[-1])
    render(env, last_state)


if __name__ == "__main__":
    main()
