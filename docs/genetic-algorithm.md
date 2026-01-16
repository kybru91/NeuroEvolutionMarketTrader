# The Genetic Algorithm

## Introduction

### What is a Genetic Algorithm?

A Genetic Algorithm (GA) is a method of solving problems by mimicking how nature evolves species over generations. Instead of explicitly programming a solution, we let solutions *evolve* through survival of the fittest.

Here's the core idea: imagine you have a population of 256 different trading strategies, each making slightly different decisions. You let them all trade on historical data and see which ones make money. The profitable ones "survive" and "reproduce" — their characteristics are combined and slightly modified to create a new generation. Over hundreds of generations, the population evolves toward increasingly profitable strategies.

### Why Use Evolution Instead of Traditional Training?

Traditional neural networks learn through **backpropagation** — they calculate how wrong they were and adjust their weights to be less wrong next time. This works well when you have a clear, immediate error signal.

But trading is different:
- **Delayed feedback**: A trade might take hours or days to resolve
- **Noisy signals**: A good decision can still lose money due to random market movement
- **Sparse rewards**: Most moments in the market, nothing significant happens

Genetic algorithms sidestep these problems. They don't need a gradient or immediate feedback. They simply ask: **"Did this strategy make money overall?"** If yes, it survives. If not, it doesn't. This binary pressure, applied over many generations, naturally evolves effective trading behaviors.

---

## The Evolution Cycle

Every generation follows the same cycle:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │Initialize│───►│ Evaluate │───►│  Select  │             │
│   │Population│    │ Fitness  │    │ Survivors│             │
│   └──────────┘    └──────────┘    └────┬─────┘             │
│        ▲                               │                    │
│        │                               ▼                    │
│   ┌────┴─────┐    ┌──────────┐    ┌──────────┐             │
│   │   New    │◄───│  Mutate  │◄───│Reproduce │             │
│   │Generation│    │ Offspring│    │(Crossover)│            │
│   └──────────┘    └──────────┘    └──────────┘             │
│                                                             │
│                    Repeat for N generations                 │
└─────────────────────────────────────────────────────────────┘
```

**In natural terms:**

1. **Initialize**: Create a diverse population of random individuals
2. **Evaluate**: Test each individual's ability to survive (trade profitably)
3. **Select**: The fittest individuals survive to reproduce
4. **Reproduce**: Survivors combine their "genes" to create offspring
5. **Mutate**: Add small random variations to explore new possibilities
6. **Repeat**: Each generation is slightly better than the last

After 100 generations with a population of 256, you've effectively tested **25,600 different trading strategies** — and the final survivor carries the accumulated wisdom of all its ancestors.

---

## Core Concepts

### 1. The Population

A **population** is a collection of neural networks, each representing a different trading strategy. They all have the same structure (same number of layers and neurons), but different **weights** — the numerical values that determine how they process information.

In this project, each generation has **256 networks** competing against each other:

```python
GeneticNetworks(
    architecture=(15, 16, 32, 64, 32, 16, 3),  # Network shape
    population_size=256,                        # Networks per generation
    survival_ratio=0.15,                        # Top 15% survive
    # ...
)
```

Think of it like this: you have 256 traders, all looking at the same market data, but each making slightly different decisions based on their unique "personality" (weights).

### 2. Evaluation — Testing Fitness

Each network must prove its worth by trading on historical market data. The **fitness** of a network is simply how much profit it makes.

Here's how a single network is evaluated:

```python
def evaluate(self, env, episodes=10):
    """Test this network across multiple trading sessions"""
    rewards = []

    for episode in range(episodes):
        observation = env.reset()  # Start fresh
        total_reward = 0
        done = False

        while not done:
            # Network decides: SIT, LONG, or SHORT
            action = self.act(observation)

            # Execute trade and get reward
            observation, reward, done, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    # Fitness = average profit across all episodes
    return np.mean(rewards)
```

Each network trades for **10 episodes** (different market periods), and its fitness is the **average reward**. This prevents a network from being lucky once — it must perform consistently.

To speed things up, all 256 networks are evaluated **in parallel** using multiple CPU cores:

```python
with Pool(num_cpus) as p:
    rewards = p.map(evaluate_network, all_networks)
```

### 3. Selection — Survival of the Fittest

After evaluation, only the **top performers survive**. In this project, the top 15% (about 38 networks) get to reproduce:

```python
# Sort by fitness, keep the best
n_survivors = int(0.15 * population_size)  # Top 15%
top_indices = np.argsort(rewards)[-n_survivors:]
survivors = [networks[i] for i in top_indices]
```

This is called **elite selection** — brutal but effective. 85% of the population dies each generation, ensuring only the best genetic material passes forward.

### 4. Reproduction — Creating Offspring

The survivors must repopulate. New networks are created through three strategies:

| Strategy | Probability | Description |
|----------|-------------|-------------|
| **Two-parent crossover** | 40% | Combine two survivors |
| **One-parent mutation** | 50% | Clone and mutate one survivor |
| **Direct copy** | 10% | Clone a survivor unchanged |

#### Crossover: Combining Two Parents

When two networks reproduce, their offspring inherits genes (weights) from both:

```python
def crossover(self, parent1, parent2):
    """Each weight has 50% chance from each parent"""

    # Start with parent1's weights
    self.weights = copy(parent1.weights)

    # For each weight, maybe take from parent2 instead
    for i, weight_matrix in enumerate(self.weights):
        for j in range(weight_matrix.shape[0]):
            for k in range(weight_matrix.shape[1]):
                if random() > 0.5:  # 50% chance
                    self.weights[i][j][k] = parent2.weights[i][j][k]
```

This is **uniform crossover** — each weight independently comes from either parent. The offspring might inherit mom's caution about volatile markets and dad's aggressive trend-following, creating a new unique combination.

#### Mutation: Adding Variation

After crossover (or copying), offspring are **mutated** — small random changes are applied:

```python
def mutate(self, variance=0.02):
    """Add small random noise to all weights"""

    for i, weight_matrix in enumerate(self.weights):
        # Add Gaussian noise: mean=0, std=variance
        noise = np.random.normal(0, variance, weight_matrix.shape)
        self.weights[i] = weight_matrix + noise
```

With `variance=0.02`, most changes are tiny (between -0.04 and +0.04). This allows **gradual exploration** — big jumps would destroy good solutions, but small tweaks might find improvements.

### 5. The Neural Network Individual

Each individual in the population is a neural network. Here's what one looks like:

```
Input Layer      Hidden Layers                    Output Layer
(15 neurons)     (16→32→64→32→16 neurons)        (3 neurons)

   [Price]                                         [SIT]
   [RSI  ] ──►  ████  ████  ████  ████  ████  ──►  [LONG]
   [MACD ]      (16)  (32)  (64)  (32)  (16)       [SHORT]
   [ ... ]
```

**The forward pass** — how a network makes a decision:

```python
def act(self, observation):
    """Given market data, decide what to do"""

    # Pass through each layer
    x = observation
    for weights, biases in zip(self.weights, self.biases):
        x = relu(x @ weights + biases)  # Matrix multiply + activation

    # Final layer: convert to probabilities
    probabilities = softmax(x)

    # Return the action with highest probability
    return np.argmax(probabilities)  # 0=SIT, 1=LONG, 2=SHORT
```

**Activation functions** give networks their power:

- **ReLU** (hidden layers): `max(0, x)` — simple non-linearity that allows complex patterns
- **Softmax** (output): Converts raw scores to probabilities that sum to 1

---

## Key Parameters

| Parameter | Default | What It Controls |
|-----------|---------|------------------|
| `population_size` | 256 | Networks competing each generation |
| `generations` | 100 | How many evolution cycles |
| `survival_ratio` | 0.15 | Percentage that survive (15% = 38 of 256) |
| `both_parent_percentage` | 0.4 | Offspring from two parents (40%) |
| `mutation_variance` | 0.005 | Size of random mutations |
| `episodes` | 10 | Trading sessions per evaluation |
| `stagnation_end` | True | Stop if no improvement for 10+ generations |

**Tuning guidance:**

- **Larger population** = More diversity, slower training
- **Higher survival ratio** = Less selection pressure, slower convergence
- **Higher mutation** = More exploration, might destroy good solutions
- **More episodes** = More reliable fitness, slower evaluation

---

## Implementation Highlights

### Parallel Processing

Evaluating 256 networks sequentially would be slow. Instead, all networks run simultaneously:

```python
from multiprocessing import Pool, cpu_count

with Pool(cpu_count()) as pool:
    fitness_scores = pool.map(evaluate_network, population)
```

On an 8-core machine, this is roughly **8x faster** than sequential evaluation.

### Early Stopping (Stagnation Detection)

If the best fitness doesn't improve for 10 generations, evolution stops early:

```python
if best_reward > best_so_far:
    stagnation = 0  # Reset counter
    best_so_far = best_reward
else:
    stagnation += 1

if stagnation > 10:
    print("No improvement for 10 generations, stopping")
    break
```

This prevents wasting computation when evolution has converged.

### Numerical Stability

The softmax function can overflow with large numbers. The implementation subtracts the maximum value first:

```python
def softmax(x):
    # Prevent overflow by subtracting max
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)
```

---

## How It Connects

The Genetic Algorithm is the **learning engine**, but it needs an environment to learn in. That environment is the **Market Environment** — a simulation of forex trading where networks can practice without risking real money.

In the next section, we'll explore how this trading environment works: how it represents market state, how it processes actions, and how it calculates rewards.

---

## Summary

The Genetic Algorithm evolves trading strategies through natural selection:

1. **Population**: 256 neural networks with random initial weights
2. **Evaluation**: Each network trades historical data; fitness = profit
3. **Selection**: Top 15% survive
4. **Reproduction**: Survivors create offspring through crossover and mutation
5. **Iteration**: Repeat for 100 generations

The result: a neural network that has been optimized across millions of simulated trades to find profitable patterns in the EUR/USD market.

*Next: [The Market Environment](./market-environment.md)*
