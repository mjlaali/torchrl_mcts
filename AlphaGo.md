This is a summary of my understanding of AlphaGo after reading the `pseudocode.py` in the supplement materials 
([pdf](https://www.science.org/doi/suppl/10.1126/science.aar6404/suppl_file/aar6404-silver-sm.pdf), [zip](https://www.science.org/doi/suppl/10.1126/science.aar6404/suppl_file/aar6404_datas1.zip)).

# Overall 
> AlphaZero training is split into two independent parts: Network training and
self-play data generation.
These two parts only communicate by transferring the latest network checkpoint
from the training to the self-play, and the finished games from the self-play
to the training.


```python
def alphazero(config: AlphaZeroConfig):
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)

  for i in range(config.num_actors):
    launch_job(run_selfplay, config, storage, replay_buffer)

  train_network(config, storage, replay_buffer)

  return storage.latest_network()
```

train_network is quite standard, just we need to have special attention on
the loss function and what is collected as the data for the algorithm (i.e. update_weights):

$$
loss = MSE(value(net), value(sim)) + CE(policy(net), policy(mcts)) + L2(weight(net))
$$


```python
def train_network(config: AlphaZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  network = Network()
  optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
                                         config.momentum)
  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch()
    update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
  loss = 0
  # added note: target_value is the game value, it is not depend on the step and the palyer. 
  # Most probably it is the outcome of the game in case it gets finished, or heuristic based otherwise.
  # target_policy is normalized count of selecting an action in MCTS search.
  for image, (target_value, target_policy) in batch:
    value, policy_logits = network.inference(image)
    loss += (
        tf.losses.mean_squared_error(value, target_value) +
        tf.nn.softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=target_policy))

  for weights in network.get_weights():
    loss += weight_decay * tf.nn.l2_loss(weights)

  optimizer.minimize(loss)

class ReplayBuffer(object):
  ...

  def sample_batch(self):
    """
    Added note:
    Sample randomly from game actions in the buffer. 
    For each action, we get state by calling g.make_image(action_idx) and
    (target_value, target_policy) by calling g.make_target(action_idx):
    
    target_value: the game result from the player perspective who made the action, 
    in case we reach to max number of action (len(game.history) == config.max_moves), it is not clear how define target_value
    it is not clear how to decide game.terminal_value(...) when len(game.history) == config.max_moves

    target_policy: normalized visit_count (visit_count/total_count)  
    """
    # Sample uniformly across positions.
    # added note: g.history -> is a list of actions 
    move_sum = float(sum(len(g.history) for g in self.buffer)) 
    games = numpy.random.choice(
        self.buffer,
        size=self.batch_size,
        p=[len(g.history) / move_sum for g in self.buffer])
    game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]
    return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]
```

* target_value: is the final output of a game play (a monte-carlo estimation of one play with the mcts policy)
* target_policy: is normalized action count (action_count/total_count) from n simulation.

Now lets see how target_value and target_policy is defined.

# Playing a game

> Each self-play job is independent of all others; it takes the latest network
snapshot, produces a game and makes it available to the training job by
writing it to a shared replay buffer.

```python
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)
```

Note that during playing a single game the network does not change, but every game
can use different networks.


> Each game is produced by starting at the initial board position, then
repeatedly executing a Monte Carlo Tree Search to generate moves until the end
of the game is reached.

```python
def play_game(config: AlphaZeroConfig, network: Network):
  game = Game()
  while not game.terminal() and len(game.history) < config.max_moves:
    action, root = run_mcts(config, game, network)
    
    game.apply(action)  # added note: game.history.append(action)
    
    # added note: this normalized action count in the root and save it 
    # in game.child_visit[action_idx]
    game.store_search_statistics(root)
      
  # added note: if we exits the loop with len(game.history) == config.max_moves
  # what would be game.terminal_value(...) ??
  return game
```


> Core Monte Carlo Tree Search algorithm.
To decide on an action, we run N simulations, always starting at the root of
the search tree and traversing the tree according to the UCB formula until we
reach a leaf node.

```python
def run_mcts(config: AlphaZeroConfig, game: Game, network: Network):
  root = Node(0)  # added note: 0 is prior value for root node.
  evaluate(root, game, network)  # initialize children (Q(s,a) or action_values) with softmax(network(state).policy_logits) 
  add_exploration_noise(config, root)  # add noise to action values

  for _ in range(config.num_simulations):
    node = root
    scratch_game = game.clone()
    search_path = [node]  # search_path only contains expanded node

    while node.expanded():   # play until reach to not seen state
      action, node = select_child(config, node) # Greedy select the child with the highest UCB score.
      scratch_game.apply(action)  
      search_path.append(node)

    value = evaluate(node, scratch_game, network)  # expand the node
    # note that search_path does not contain not expanded node
    # expanded nodes: +1 visit_count +value value_sum
    backpropagate(search_path, value, scratch_game.to_play())  
  return select_action(config, game, root), root


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network) -> float:
  """
  Use the given network, compute prior value (predicted_target_value, predicted_target_policy) for the current state of the game
  
  * predicted_target_value: predicted outcome of the game, will be returned by function call
  * predicted_target_policy: softmax(policy_logits) will be saved as the prior value of each node
  """
  value, policy_logits = network.inference(game.make_image(-1))

  # Expand the node.
  node.to_play = game.to_play()
  policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
  policy_sum = sum(policy.itervalues())
  for action, p in policy.iteritems():
    node.children[action] = Node(p / policy_sum)
  return value



# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
  actions = node.children.keys()
  # added note: config.root_dirichlet_alpha = 0.3 for chess, 0.03 for Go and 0.15 for shogi.
  #   the bigger config.root_dirichlet_alpha, the bigger the noise values
  #   config.root_exploration_fraction = 0.25
  noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
  _, action, child = max((ucb_score(config, node, child), action, child)
                         for action, child in node.children.iteritems())
  return action, child


```


> The score for a node is based on its value, plus an exploration bonus based on
the prior.

```python
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
  pb_c = math.log(
      (parent.visit_count + config.pb_c_base + 1) /
       config.pb_c_base
  ) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score
```

$ucb(c^i, p^i, v^i| tc, c_b, c_i)=(\frac{(\frac{log(tc+c_b+1)}{c_b} + c_i) \times sqrt(tc)}{c^i + 1}) \times p^i + v^i$

- $c^i$: visit count for i-th action
- $p^i=net(obs)[i]$: prior value (probability of selecting) for i-th action
- $v^i = \frac{\sum_{j=0}^{c^i}v^i_j}{c^i}$: current monte-carlo estimation of action value for i-th action 
- $tc = \sum_{i=0}^k c^i + 1$: Total visited count + 1
* $c_b$: constant value
* $c_i$: constant value