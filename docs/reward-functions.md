# Reward Functions

## Introduction

### The Language of Learning

How does a neural network learn what "good trading" looks like? Through **rewards**.

Every time the agent takes an action — buy, sell, or wait — the environment gives it a number: positive for good outcomes, negative for bad ones. Over thousands of generations, the genetic algorithm evolves networks that maximize this reward signal.

The **reward function** defines what "good" means. Different reward functions create different trading styles:

- Want aggressive profit-seeking? Use one reward function.
- Want steady, low-risk returns? Use another.
- Want a balance? Use a third.

This project implements three reward schemes, each encoding a different trading philosophy.

---

## The Three Reward Schemes

### Overview

| Reward Function | Philosophy | Best For |
|-----------------|------------|----------|
| **PercentChange** | "What have you done for me lately?" | Quick, reactive trading |
| **SimpleProfit** | "Show me compound growth" | Consistent trend-following |
| **RiskAdjustedReturns** | "Don't lose my money" | Professional risk management |

---

### 1. PercentChange

**Philosophy:** Reward the most recent gain or loss.

This is the simplest approach — it looks only at the **last step** and asks: "Did your portfolio go up or down?"

```python
def get_reward(self, portfolio):
    net_worth = portfolio['performance']['net_worth']

    # Calculate percentage change between steps
    pct_change = (net_worth[-1] - net_worth[-2]) / net_worth[-2]

    return pct_change
```

**Example:**

```
Portfolio history: [100, 102, 98, 103]
                              ↑    ↑
                           Last two values

Change: (103 - 98) / 98 = +5.1%
Reward: +0.051
```

**Characteristics:**

| Aspect | Description |
|--------|-------------|
| **Focus** | Single step (immediate) |
| **Responsiveness** | Very high — reacts to every price move |
| **Stability** | Low — can be noisy |
| **Risk awareness** | None — treats gains and losses symmetrically |

**When to use:** When you want agents that react quickly to market changes, or for high-frequency strategies.

**Drawback:** Can lead to overtrading and whipsaw behavior in choppy markets.

---

### 2. SimpleProfit

**Philosophy:** Reward cumulative, compounded growth.

Instead of looking at just the last step, SimpleProfit considers the **entire history** and calculates compound returns — the same way investment performance is measured in the real world.

```python
def get_reward(self, portfolio):
    net_worth = portfolio['performance']['net_worth']

    # Calculate percentage changes
    pct_change = np.diff(net_worth) / net_worth[1:]

    # Compound them: (1+r1) × (1+r2) × ... × (1+rn) - 1
    cumulative_return = np.cumprod(1.0 + pct_change) - 1

    return cumulative_return[-1]
```

**Example:**

```
Portfolio history: [100, 102, 98, 103]
Step changes:      +2%, -3.9%, +5.1%

Compound calculation:
(1.02) × (0.961) × (1.051) - 1 = 1.031 - 1 = +3.1%

Reward: +0.031
```

**Why compounding matters:**

```
Scenario A: +10%, then -10%
  Simple sum: 0%
  Compound: 1.10 × 0.90 = 0.99 → -1% (reality!)

Scenario B: +5%, +5%, +5%, +5%
  Simple sum: +20%
  Compound: 1.05⁴ = 1.215 → +21.5% (better than sum!)
```

**Characteristics:**

| Aspect | Description |
|--------|-------------|
| **Focus** | Entire window (cumulative) |
| **Responsiveness** | Medium — smoothed by history |
| **Stability** | High — single bad step doesn't ruin everything |
| **Risk awareness** | Implicit — losses reduce compound growth |

**When to use:** For agents that should build sustained wealth over time, following trends rather than chasing noise.

**This is the default reward function used in production (v3.0141).**

---

### 3. RiskAdjustedReturns (Sortino Ratio)

**Philosophy:** Reward returns, but penalize volatility — especially downside volatility.

This is the most sophisticated reward function. It implements the **Sortino Ratio**, a professional risk metric that asks: "How much return are you generating per unit of *downside* risk?"

```python
def get_reward(self, portfolio):
    net_worth = portfolio['performance']['net_worth']
    returns = np.diff(net_worth) / net_worth[1:]

    # Identify losing periods (downside)
    downside_returns = [r**2 for r in returns if r < 0]

    # Calculate metrics
    expected_return = np.mean(returns)
    downside_deviation = np.sqrt(np.std(downside_returns))

    # Sortino Ratio
    sortino = expected_return / downside_deviation

    return sortino / scaling_factor
```

**The key insight:** Not all volatility is bad.

- **Upside volatility** (surprise gains): Good! Don't penalize this.
- **Downside volatility** (surprise losses): Bad! Penalize this heavily.

The Sortino Ratio captures this asymmetry.

**Example:**

```
Strategy A: Returns [+2%, -4%, +5%, +1%, -2%]
  Mean return: +0.4%
  Downside returns: [-4%, -2%] → squared: [0.16%, 0.04%]
  Downside deviation: ~0.3%
  Sortino: 0.4 / 0.3 = 1.33

Strategy B: Returns [+1%, +1%, +1%, +1%, +1%]
  Mean return: +1%
  Downside returns: [] (none!)
  Downside deviation: ~0
  Sortino: Very high (stable!)

Even though Strategy A has higher peaks, Strategy B scores better
because it has no downside volatility.
```

**Characteristics:**

| Aspect | Description |
|--------|-------------|
| **Focus** | Risk-adjusted performance |
| **Responsiveness** | Medium — considers full history |
| **Stability** | High — rewards consistency |
| **Risk awareness** | Explicit — penalizes losses more than rewards gains |

**When to use:** For professional-grade trading where risk management matters as much as returns.

---

## Comparing the Three

### Same Portfolio, Different Rewards

```
Portfolio history: [100, 105, 95, 102, 108, 103]
Changes:                +5%  -9.5% +7.4% +5.9% -4.6%
```

| Function | Calculation | Reward |
|----------|-------------|--------|
| **PercentChange** | Last step: -4.6% | **-0.046** |
| **SimpleProfit** | Compound: 1.05×0.905×1.074×1.059×0.954 - 1 | **+0.030** |
| **RiskAdjustedReturns** | Mean (+0.9%) / Downside std | **~0.15** (scaled) |

Notice:
- PercentChange is **negative** because the last step was a loss
- SimpleProfit is **positive** because overall the portfolio grew
- RiskAdjustedReturns is **moderate** because there was significant downside volatility

---

### Behavioral Impact

The reward function shapes the agent's "personality":

```
PercentChange Agent:
├── Reacts to every price tick
├── Trades frequently
├── Chases short-term momentum
└── Can get whipsawed in ranging markets

SimpleProfit Agent:
├── Focuses on overall growth
├── Holds positions longer
├── Rides trends
└── More stable across market conditions

RiskAdjustedReturns Agent:
├── Avoids volatile strategies
├── Prefers steady, predictable returns
├── Cuts losses quickly
└── Professional, conservative style
```

---

## The Penalty System

Beyond the reward function, the environment applies **penalties** to discourage bad behavior.

### Inaction Penalty

```python
inaction_penalty = -0.00001  # Small but persistent
```

**What it does:** Every time the agent chooses to "SIT" (do nothing), it receives a tiny negative reward.

**Why:** Without this, agents might learn that the safest strategy is to never trade. The inaction penalty gently pushes them to participate in the market.

**Impact over 1000 steps of sitting:**
```
Total penalty: -0.00001 × 1000 = -0.01
```

Small, but enough to favor active strategies over passive ones.

---

### Bankruptcy Penalty

```python
lost_all_cash_penalty = -100  # Catastrophic
```

**What it does:** If the agent loses all its money (or shorts too aggressively), the episode ends immediately with a massive negative reward.

**Trigger conditions:**
1. Cash drops to zero or below
2. Short positions exceed initial capital

```python
def _lost_all_cash(self):
    if self.cash <= 0:
        return True
    if sum(self.inventory) < -self.initial_cash:
        return True  # Shorts too large
    return False
```

**Why:** This teaches agents that survival matters. A -100 penalty is so severe that networks learn to avoid risky strategies that might lead to bankruptcy.

---

### No Trades Penalty

```python
no_trades_penalty = -100  # Applied at episode end
```

**What it does:** If an agent completes an entire episode without making a single trade, it receives a large negative reward.

**Why:** Ensures agents learn to take action. Combined with the inaction penalty, this prevents the evolution of "do nothing" strategies.

---

## The Complete Reward Calculation

Every step, the environment calculates the final reward:

```python
def make_action(self, state, action):
    penalties = 0

    # Execute the action (buy/sell/sit)
    execute_trade(action)

    # Update portfolio tracking
    self._update_portfolio()

    # Apply penalties
    if action == 0:  # Sitting
        penalties += self.inaction_penalty

    if self._lost_all_cash():
        penalties += self.lost_all_cash_penalty

    # Get base reward from selected function
    base_reward = self.reward_function.get_reward(self.portfolio)

    # Final reward
    reward = base_reward + penalties

    return reward
```

**The formula:**
```
Final Reward = Reward Function Output + Penalties
```

---

## Portfolio Tracking

### The Sliding Window

Reward functions need historical data to calculate returns. The environment maintains a **sliding window** of net worth values:

```python
self.portfolio = {
    'performance': {
        'net_worth': deque(maxlen=100)  # Last 100 values
    }
}
```

**How it works:**

```
Step 1: net_worth = [100]
Step 2: net_worth = [100, 101]
Step 3: net_worth = [100, 101, 99]
...
Step 100: net_worth = [100, 101, 99, ..., 105]  (100 values)
Step 101: net_worth = [101, 99, ..., 105, 106]  (oldest dropped)
```

The `deque` automatically removes the oldest value when full, keeping memory bounded.

**Net worth calculation:**
```python
net_worth = cash + (holdings × current_price)
```

---

## Connecting to Genetic Algorithm

The reward function **is** the fitness function.

When the genetic algorithm evaluates a network, it:

1. Runs the network for multiple episodes (default: 10)
2. Accumulates rewards from each step
3. Averages the total rewards across episodes
4. Uses this average as the network's **fitness score**

```python
def evaluate(self, env, episodes=10):
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = self.act(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    fitness = np.mean(total_rewards)
    return fitness
```

**Selection pressure:**
- Top 15% of networks (by fitness) survive
- They reproduce to create the next generation
- Over generations, networks evolve to maximize reward

The reward function directly shapes what kind of trading strategies evolve.

---

## Configuration

### Choosing a Reward Function

In the configuration file:

```json
{
  "params": {
    "reward_function": "SimpleProfit",
    "profit_window_size": 100,
    "inaction_penalty": -0.00001,
    "lost_all_cash_penalty": -100
  }
}
```

**Options:**
- `"PercentChange"` — Reactive, short-term focus
- `"SimpleProfit"` — Compound growth focus (recommended)
- `"RiskAdjustedReturns"` — Risk-adjusted focus

---

## Summary

Reward functions are the **soul** of the learning process:

| Function | Formula | Creates Agents That... |
|----------|---------|------------------------|
| **PercentChange** | Last step's % change | React quickly, trade often |
| **SimpleProfit** | Compound returns | Build steady wealth |
| **RiskAdjustedReturns** | Sortino ratio | Manage risk professionally |

**Penalties** add constraints:
- Inaction penalty: "Don't just sit there"
- Bankruptcy penalty: "Don't blow up"
- No trades penalty: "You must participate"

Together, reward functions and penalties define the **fitness landscape** that guides evolution toward profitable, robust trading strategies.

*Previous: [Technical Analysis](./technical-analysis.md)*
*Next: [Flask API & Deployment](./deployment.md)*
