# The Market Environment

## Introduction

### What is the Market Environment?

The Market Environment is a **trading simulator** — a virtual world where neural networks can practice trading without risking real money. It's like a flight simulator for pilots, but for trading algorithms.

Every time a network wants to learn, it steps into this environment and experiences thousands of trading scenarios. It sees price data, makes decisions (buy, sell, or wait), and receives feedback in the form of profits or losses. Over many iterations, networks learn which patterns lead to profits and which lead to losses.

### Why Build a Custom Environment?

Standard machine learning environments (like video games) don't capture the nuances of financial markets:

- **Partial information**: You see current prices but don't know the future
- **Transaction costs**: Every trade has a cost (commissions, spreads)
- **Position management**: You can hold multiple shares, go long or short
- **Capital constraints**: You can't buy more than your cash allows
- **Risk of ruin**: Lose too much and you're out of the game

This custom environment simulates all of these realities, preparing networks for actual market conditions.

---

## The Trading World

### What the Agent Sees (State Space)

At each moment, the agent receives a snapshot of the market — a set of numbers describing what's happening:

```
State = [Price Data] + [Technical Indicators] + [CNN Predictions]
```

**Example state (15 dimensions):**

| Index | Feature | Example Value |
|-------|---------|---------------|
| 0-3 | Detrended OHLC | 0.002, 0.005, -0.001, 0.003 |
| 4-5 | RSI at intervals | 0.65, 0.58 |
| 6-7 | MACD histogram | 0.0012, 0.0008 |
| 8-11 | More indicators | ... |
| 12-14 | CNN predictions | 0.2, 0.5, 0.3 |

The agent doesn't see raw prices — it sees **processed features** that highlight patterns:
- **Detrended prices**: Removes overall trend, focuses on relative movements
- **Technical indicators**: Mathematical transformations that traders use
- **CNN predictions**: Another model's opinion on market direction

### What the Agent Can Do (Action Space)

The agent has three choices at every moment:

| Action | Code | What It Means |
|--------|------|---------------|
| **SIT** | 0 | Do nothing — stay out of the market |
| **LONG** | 1 | Buy EUR/USD — bet the Euro will strengthen |
| **SHORT** | 2 | Sell EUR/USD — bet the Euro will weaken |

```python
# The agent outputs a number 0, 1, or 2
action = agent.act(state)

if action == 0:    # SIT
    # Wait and observe
elif action == 1:  # LONG
    # Open a buy position (or close shorts first)
elif action == 2:  # SHORT
    # Open a sell position (or close longs first)
```

**Key behavior**: If the agent is long and chooses SHORT, it first closes all long positions, then opens short positions. This prevents holding contradictory positions.

---

## Position Management

### How Trades Work

The environment tracks your trading activity with a few key variables:

```python
holdings = 0        # How many units you hold (+positive=long, -negative=short)
inventory = []      # The prices at which you bought each unit
cash = 20.0         # Available money
```

**Example trading sequence:**

```
Step 1: Start
  cash=20.0, holdings=0, inventory=[]
  Agent sees: price=$5.00, decides LONG

Step 2: After LONG
  cash=15.0, holdings=1, inventory=[5.0]
  (Bought 1 unit at $5, cash decreased by $5)

Step 3: Price rises to $5.50, agent decides LONG again
  cash=9.5, holdings=2, inventory=[5.5, 5.0]
  (Bought another unit at $5.50)

Step 4: Price rises to $6.00, agent decides SHORT
  First: Close all longs
    Sell 2 units at $6.00 = $12.00
    Cost basis was $5.50 + $5.00 = $10.50
    Profit = $12.00 - $10.50 = $1.50
    cash = 9.5 + 10.5 + 1.5 = 21.5
  Then: Open short position
    holdings=-1, inventory=[-6.0]
```

### Going Long vs Going Short

**Long position** (betting price will rise):
- You buy at current price, hoping to sell higher later
- Profit = (Sell Price - Buy Price) × Quantity

**Short position** (betting price will fall):
- You "borrow and sell" at current price, hoping to buy back cheaper
- Profit = (Short Price - Cover Price) × Quantity

```python
def _close_all_long(self, current_price):
    """Close long positions and calculate profit"""
    # Sell all holdings at current price
    proceeds = self.holdings * current_price
    # Subtract what we paid originally
    cost = sum(self.inventory)
    # Calculate profit
    profit = proceeds - cost - self.commission
    return profit

def _close_all_short(self, current_price):
    """Close short positions and calculate profit"""
    # We received money when we shorted (stored as negative in inventory)
    proceeds = -sum(self.inventory)
    # Now we must buy back at current price
    cost = -self.holdings * current_price  # holdings is negative for shorts
    # Calculate profit
    profit = proceeds - cost - self.commission
    return profit
```

### Position Limits

The environment enforces realistic constraints:

```python
max_possible_holdings = 20  # Can't hold more than 20 units

# Before buying, check constraints:
if current_price < self.cash and abs(self.holdings) < self.max_possible_holdings:
    # Allowed to buy
else:
    # Can't afford it or position too large
```

This prevents unrealistic scenarios like buying infinite shares on margin.

---

## Rewards and Penalties

### How the Agent Learns What's Good

The environment communicates through **rewards** — positive numbers for good outcomes, negative for bad. The agent's goal is to maximize total reward.

```
Total Reward = Base Reward + Penalties
```

### Base Reward: Measuring Performance

Three different schemes are available:

#### 1. SimpleProfit (Used in Production)

Measures cumulative returns over a sliding window:

```python
def get_reward(self, portfolio):
    # Get recent net worth history
    net_worth = portfolio['performance']['net_worth']

    # Calculate percentage changes
    pct_change = np.diff(net_worth) / net_worth[1:]

    # Compound them: (1+r1) × (1+r2) × ... - 1
    cumulative_return = np.cumprod(1.0 + pct_change) - 1

    return cumulative_return[-1]
```

**Example:**
```
Net worth over 5 steps: [100, 102, 101, 104, 106]
Changes: [+2%, -1%, +3%, +2%]
Cumulative: (1.02)(0.99)(1.03)(1.02) - 1 = 6.1%
Reward = 0.061
```

#### 2. RiskAdjustedReturns (Sortino Ratio)

Rewards consistent profits, penalizes volatility:

```python
def get_reward(self, portfolio):
    returns = calculate_returns(portfolio)

    # Separate downside returns (losses)
    downside = [r for r in returns if r < 0]
    downside_volatility = np.std(downside)

    # Sortino ratio: return / downside_risk
    mean_return = np.mean(returns)
    sortino = mean_return / (downside_volatility + 1e-9)

    return sortino
```

**Why this matters:**
- Strategy A: Returns [+5%, -4%, +6%, -3%] → High mean, but volatile
- Strategy B: Returns [+2%, +1%, +2%, +1%] → Lower mean, but stable

RiskAdjustedReturns might prefer Strategy B because it has lower downside risk.

#### 3. PercentChange

Simple approach — just the last period's change:

```python
def get_reward(self, portfolio):
    net_worth = portfolio['performance']['net_worth']
    return (net_worth[-1] - net_worth[-2]) / net_worth[-2]
```

### Penalties: Discouraging Bad Behavior

#### Inaction Penalty

```python
inaction_penalty = -0.00001  # Small negative reward for doing nothing
```

If the agent just sits and never trades, it slowly accumulates negative reward. This pushes the network to at least try trading.

#### Lost All Cash Penalty

```python
lost_all_cash_penalty = -100  # Severe punishment for bankruptcy
```

If the agent loses too much money:
- Cash drops to zero, or
- Short positions become too large

...the episode ends immediately with a massive penalty. This teaches networks to manage risk.

```python
def _lost_all_cash(self):
    """Check if agent has gone bankrupt"""
    # Out of cash
    if self.cash <= 0:
        return True

    # Shorts have grown too large (unlimited loss potential)
    if sum(self.inventory) < -self.initial_cash:
        return True

    return False
```

---

## The Episode Lifecycle

### Starting an Episode: reset()

Each learning episode starts fresh:

```python
def reset(self):
    """Start a new trading session"""

    # Reset money
    self.cash = self.initial_cash  # Back to $20
    self.holdings = 0               # No positions
    self.inventory = []             # Empty portfolio

    # Reset statistics
    self.total_profit = 0.0
    self.total_trades = 0
    self.win_trades = 0

    # Reset time
    self.step_value = 0

    # Return initial observation
    return self.get_state(0)
```

### Taking a Step: step()

The core loop — agent observes, decides, and learns:

```python
def step(self, action):
    """
    Execute one trading decision

    Args:
        action: 0 (SIT), 1 (LONG), or 2 (SHORT)

    Returns:
        next_state: What the market looks like now
        reward: How well did this action do?
        done: Is the episode over?
        info: Additional information
    """

    # Get current market state
    state = self.get_state(self.step_value)

    # Execute the action and get reward
    reward = self.make_action(state, action)

    # Check if episode should end
    done = self.check_done()

    # Move to next time step
    self.step_value += 1

    # Get new state
    next_state = self.get_state(self.step_value)

    return next_state, reward, done, {}
```

### Ending an Episode: check_done()

Episodes end in two ways:

```python
def check_done(self):
    # Bad ending: lost all money
    if self._lost_all_cash():
        return True

    # Normal ending: reached end of data
    if self.step_value >= self.num_steps - 2:
        return True

    return False
```

### A Complete Episode

```python
# Initialize
env = MarketEnvironmentV1(market_data, initial_cash=20.0)
agent = trained_neural_network

# Start episode
state = env.reset()
done = False
total_reward = 0

# Trading loop
while not done:
    # Agent decides
    action = agent.act(state)  # Returns 0, 1, or 2

    # Environment responds
    next_state, reward, done, _ = env.step(action)

    # Track performance
    total_reward += reward
    state = next_state

# Episode finished
print(f"Total reward: {total_reward}")
print(f"Final profit: ${env.total_profit}")
print(f"Win rate: {env.win_trades / env.total_trades * 100}%")
```

---

## Environment Versions

The project evolved through multiple versions:

### V0: Basic (Long Only)

```python
# Actions: SIT, BUY, SELL
# Can only go long (buy then sell)
# Simplest version for initial testing
```

### V1: Production (Long + Short)

```python
# Actions: SIT, LONG, SHORT
# Can profit from both rising and falling markets
# Automatically closes opposite positions
# Used for final agent training
```

### V2: Enhanced State

```python
# Same as V1, but state includes:
# - Current holdings count
# - Unrealized P&L ratio
# Gives agent awareness of its own position
```

---

## Realistic Trading Simulation

### What Makes It Realistic

| Feature | How It Works |
|---------|--------------|
| **Cash constraints** | Can't buy more than you can afford |
| **Position limits** | Maximum 20 units at once |
| **Per-share tracking** | Tracks exact entry price of each unit |
| **Commission support** | Can add transaction costs |
| **Bankruptcy** | Episode ends if you lose everything |

### What's Simplified

| Simplification | Real World |
|----------------|------------|
| **No slippage** | Real orders might fill at worse prices |
| **Instant execution** | Real orders take time to fill |
| **No spread** | Real markets have bid/ask spread |
| **Single asset** | Real portfolios are diversified |

These simplifications keep training fast while capturing the essential dynamics of profitable trading.

---

## Performance Tracking

After each episode, you can inspect detailed statistics:

```python
info = env.get_info()
```

| Metric | Meaning |
|--------|---------|
| `cash_profit` | Percentage gain on initial capital |
| `total_profit` | Dollar profit from closed trades |
| `win_trades` | Number of profitable trades |
| `winning_percent` | Win rate (0.0 to 1.0) |
| `total_trades` | Total completed trades |
| `total_longs` | Long trades taken |
| `total_shorts` | Short trades taken |
| `win_longs` | Win rate on long trades |
| `win_shorts` | Win rate on short trades |

**Example output:**
```python
{
    'cash_profit': '14.17%',
    'total_profit': '$2.83',
    'winning_percent': '0.80',  # 80% win rate
    'total_trades': 15,
    'total_longs': 10,
    'total_shorts': 5,
    'win_longs': '0.90',        # 90% of longs profitable
    'win_shorts': '0.60'        # 60% of shorts profitable
}
```

---

## How It Connects

The Market Environment is where the **Genetic Algorithm** tests its neural networks. Each network in the population trades in this environment, accumulating rewards that become its **fitness score**.

Networks that navigate this environment well — making profitable trades while avoiding bankruptcy — survive to reproduce. Over generations, the population evolves strategies specifically adapted to this trading world.

In the next section, we'll explore the **CNN Classifier** — a separate neural network that provides pattern recognition hints to the trading agent.

---

## Summary

The Market Environment simulates forex trading:

1. **State**: Market data + technical indicators + CNN predictions
2. **Actions**: SIT (wait), LONG (buy), SHORT (sell)
3. **Positions**: Track holdings, entry prices, and cash
4. **Rewards**: Profit-based with risk adjustment
5. **Penalties**: Inaction and bankruptcy discouraged
6. **Realism**: Cash constraints, position limits, per-share tracking

This controlled environment lets neural networks experience thousands of trading scenarios safely, learning profitable patterns through trial and error.

*Previous: [The Genetic Algorithm](./genetic-algorithm.md)*
*Next: [The CNN Classifier](./cnn-classifier.md)*
