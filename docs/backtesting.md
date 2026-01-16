# Backtesting

## Why Backtesting is Critical

A trading strategy that looks good on paper means nothing until it's tested on real historical data.

Backtesting was **critical validation** in developing this system. It answered the fundamental question: *"Does this actually work?"*

Without backtesting:
- You don't know if the model overfitted to training data
- You can't measure real performance metrics (win rate, drawdown, Sharpe ratio)
- You're gambling, not trading

With backtesting:
- You see how the strategy performs on data it's never seen
- You measure risk-adjusted returns
- You identify weaknesses before risking real money

This project uses **Backtrader**, a professional-grade Python backtesting framework, to validate the trained GA agent across multiple time periods and market conditions.

---

## How It Works

### The Testing Pipeline

```
Trained GA Agent (saved_models/DeepNeuro_Gen_best_100.pkl)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    RLStrategy Class                          │
│  • Loads pre-trained agent                                  │
│  • Prepares 15-dimensional state vector                     │
│  • Executes agent decisions as real trades                  │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backtrader Engine                         │
│  • Simulates broker execution                               │
│  • Tracks portfolio value                                   │
│  • Collects performance metrics                             │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Analysis & Visualization                  │
│  • Win rate, returns, drawdown                              │
│  • Sharpe ratio, Sortino ratio                              │
│  • Trade-by-trade breakdown                                 │
│  • Equity curve charts                                      │
└─────────────────────────────────────────────────────────────┘
```

### The RLStrategy

The core of backtesting is the `RLStrategy` class that wraps the trained agent:

```python
class RLStrategy(BaseStrategy):
    def __init__(self):
        # Load the trained GA agent
        self.ga = GeneticNetworks(architecture=(15, 16, 32, 64, 32, 16, 3), ...)
        self.ga.load_weights('saved_models/DeepNeuro_Gen_best_100.pkl')
        self.agent = self.ga.best_network

    def next(self):
        # Called on each new candle

        # 1. Prepare state vector (same as training)
        state = self.prepare_state()

        # 2. Get agent's decision
        action = self.agent.act(state)

        # 3. Execute the trade
        if action == 1:  # LONG
            self.close_shorts()
            self.buy()
        elif action == 2:  # SHORT
            self.close_longs()
            self.sell()
        # action == 0: HOLD (do nothing)
```

The key insight: **the backtesting state preparation must exactly match the training state preparation**. Any difference would invalidate the results.

---

## Results

### Performance Across Different Periods

The trained model was tested on multiple time periods to validate robustness:

#### 2017 - 30 Minute Candles (EURUSD)

| Metric | Value |
|--------|-------|
| **Duration** | 1,521 days |
| **Total Bars** | 10,018 |
| **Return** | **+9.99%** |
| **Buy & Hold Return** | +13.86% |
| **Win Rate** | **74.30%** |
| **Max Drawdown** | -3.55% |
| **Total Trades** | 3,160 |
| **Best Trade** | +0.12% |
| **Worst Trade** | -0.08% |
| **SQN** | 2.35 |

#### 2018 - 30 Minute Candles (EURUSD)

| Metric | Value |
|--------|-------|
| **Duration** | 1,521 days |
| **Total Bars** | 10,593 |
| **Return** | **+10.06%** |
| **Buy & Hold Return** | -5.01% |
| **Win Rate** | **75.39%** |
| **Max Drawdown** | -2.72% |
| **Total Trades** | 3,318 |
| **Best Trade** | +0.10% |
| **Worst Trade** | -0.07% |
| **SQN** | 2.41 |

#### 2016 - 30 Minute Candles (EURUSD)

| Metric | Value |
|--------|-------|
| **Duration** | 1,521 days |
| **Total Bars** | 10,000+ |
| **Return** | **+24.87%** |
| **Buy & Hold Return** | -3.14% |
| **Win Rate** | **81.40%** |
| **Max Drawdown** | -2.58% |
| **Total Trades** | 7,860 |

### Summary Statistics

| Year | Strategy Return | Buy & Hold | Outperformance | Win Rate |
|------|-----------------|------------|----------------|----------|
| 2016 | +24.87% | -3.14% | **+28.01%** | 81.40% |
| 2017 | +9.99% | +13.86% | -3.87% | 74.30% |
| 2018 | +10.06% | -5.01% | **+15.07%** | 75.39% |
| **Average** | **+14.97%** | **+1.90%** | **+13.07%** | **77.03%** |

**Key observations:**
- Consistent positive returns across all test periods
- Significantly outperforms buy-and-hold in 2 of 3 years
- Win rate remains stable at 74-81%
- Maximum drawdown contained under 4%

---

## Metrics Explained

### What We Measure

| Metric | What It Tells You |
|--------|-------------------|
| **Return %** | Total profit/loss over the period |
| **Buy & Hold Return** | What you'd get just holding — the benchmark |
| **Win Rate** | Percentage of profitable trades |
| **Max Drawdown** | Worst peak-to-trough decline — your worst moment |
| **Avg Drawdown** | Typical decline severity |
| **SQN (System Quality Number)** | Overall system quality (>2 is good, >3 is excellent) |
| **Sharpe Ratio** | Return per unit of total risk |
| **Sortino Ratio** | Return per unit of downside risk |
| **Best/Worst Trade** | Extremes — shows consistency |

### How They're Calculated

```python
def print_analysis(cerebro):
    # Get analyzers
    returns = cerebro.analyzers.returns.get_analysis()
    trades = cerebro.analyzers.trades.get_analysis()
    drawdown = cerebro.analyzers.drawdown.get_analysis()
    sqn = cerebro.analyzers.sqn.get_analysis()
    sharpe = cerebro.analyzers.sharpe.get_analysis()

    # Calculate metrics
    total_return = returns['rtot'] * 100
    win_rate = trades['won']['total'] / trades['total']['total'] * 100
    max_drawdown = drawdown['max']['drawdown']

    # Sortino ratio (manual calculation)
    returns_series = get_returns_series()
    downside_returns = returns_series[returns_series < 0]
    sortino = returns_series.mean() / downside_returns.std()
```

---

## Backtrader Integration

### Setting Up a Backtest

```python
import backtrader as bt
from backtesting.base_strategy import BaseStrategy
from backtesting.extended_pandas_data import ExtendedPandasData

# 1. Create the engine
cerebro = bt.Cerebro()

# 2. Add data feed
data = ExtendedPandasData(
    dataname=df,
    datetime='timestamp',
    open='Open', high='High', low='Low', close='Close',
    volume='Volume'
)
cerebro.adddata(data)

# 3. Add strategy
cerebro.addstrategy(RLStrategy)

# 4. Configure broker
cerebro.broker.setcash(10000)
cerebro.broker.setcommission(commission=0.0)

# 5. Add analyzers
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

# 6. Run backtest
results = cerebro.run()

# 7. Analyze and plot
print_analysis(cerebro)
cerebro.plot()
```

### Extended Data Feed

The `ExtendedPandasData` class handles technical indicators:

```python
class ExtendedPandasData(bt.feeds.PandasData):
    # Map 300+ technical indicator columns
    lines = ('SMA_2', 'SMA_3', ..., 'RSI_99', 'MACDhist_14', ...)

    params = (
        ('SMA_2', -1),
        ('SMA_3', -1),
        # ... all indicator mappings
    )
```

This allows Backtrader to access any pre-computed indicator directly.

---

## Lessons Learned

### 1. Train/Test Split Matters

**The problem:** If you test on the same data you trained on, you're measuring memorization, not generalization.

**The solution:** This project uses strict temporal separation:
- **Training data:** 2004-2015 (11 years)
- **Test data:** 2016-2018 (3 years, completely separate)

The model never saw 2016-2018 data during evolution. Every test result represents true out-of-sample performance.

```
Timeline:
├─────────────────────────────────────┼───────────────────┤
2004                                 2015              2018
         TRAINING DATA                    TEST DATA
         (GA evolution)                (Backtesting)
```

### 2. Different Market Regimes

**The reality:** Markets change character over time.

| Year | Market Character | Strategy Performance |
|------|------------------|---------------------|
| 2016 | Volatile, trending | Excellent (+24.87%) |
| 2017 | Strong uptrend | Good (+9.99%), but buy-and-hold won |
| 2018 | Choppy, declining | Excellent (+10.06%), beat buy-and-hold by 15% |

**The lesson:** The strategy works best in volatile/declining markets where active trading has an edge. In strong trending markets (2017), simple buy-and-hold can outperform.

A robust strategy should:
- Perform reasonably in all conditions
- Not catastrophically fail in any condition
- Have an edge in specific conditions

This strategy meets all three criteria.

### 3. Overfitting Risks

**The danger:** A strategy that perfectly fits historical data often fails on new data.

**Signs of overfitting:**
- Amazing training results, poor test results
- Performance degrades sharply on new data
- Strategy only works on specific date ranges

**How this project avoids overfitting:**

1. **Genetic algorithm regularization**: Mutation and crossover prevent networks from memorizing specific patterns

2. **Multiple episode evaluation**: Each network is tested on 10 different episodes during training

3. **Simple architecture**: The network is small (15→16→32→64→32→16→3), limiting its ability to memorize

4. **Consistent test performance**: Similar results across 2016, 2017, 2018 suggest real generalization

**Evidence against overfitting:**
- Win rate stays at 74-81% across all test years
- Drawdown stays under 4% consistently
- Strategy adapts to different market conditions

---

## Visualization

Backtrader generates comprehensive charts:

```python
cerebro.plot(width=1000, stdstats=False)
```

**What you see:**
- **Price chart**: OHLC candles with entry/exit markers
- **Equity curve**: Portfolio value over time
- **Trade markers**: Green triangles (buy), red triangles (sell)
- **Drawdown overlay**: Shaded areas showing decline periods

These visuals help identify:
- Where the strategy performs well/poorly
- Clustering of trades around certain events
- Recovery patterns after drawdowns

---

## Running Your Own Backtest

### Quick Start

```python
# In Jupyter notebook
from backtesting.utils import print_analysis

# Load your data
df = pd.read_csv('data/EURUSD_30M.csv')
df = add_technical_indicators_with_intervals(df, indicators, intervals)

# Run backtest
cerebro = bt.Cerebro()
cerebro.adddata(ExtendedPandasData(dataname=df))
cerebro.addstrategy(RLStrategy)
cerebro.broker.setcash(10000)

# Add all analyzers
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

# Execute
results = cerebro.run()

# Analyze
print_analysis(cerebro)
cerebro.plot()
```

### Testing Different Periods

```python
# Test on 2019 data
df_2019 = df[(df['timestamp'] >= '2019-01-01') & (df['timestamp'] < '2020-01-01')]

# Test on 2020 data
df_2020 = df[(df['timestamp'] >= '2020-01-01') & (df['timestamp'] < '2021-01-01')]

# Compare results across periods
```

---

## Summary

Backtesting validated that the trained GA agent works on real historical data:

| Key Finding | Evidence |
|-------------|----------|
| **Consistent returns** | 10-25% annually across test periods |
| **High win rate** | 74-81% of trades profitable |
| **Controlled risk** | Max drawdown <4% |
| **Beats benchmark** | Outperforms buy-and-hold in 2/3 years |
| **Generalizes** | Similar performance on unseen 2016-2018 data |

**Lessons learned:**
1. Train/test split is non-negotiable
2. Test across different market regimes
3. Watch for overfitting — consistent results are the antidote

Backtesting doesn't guarantee future performance, but it proves the strategy has real predictive power on historical data. That's the minimum bar for any serious trading system.

*Previous: [Flask API & Deployment](./deployment.md)*
*Next: [Configuration System](./configuration.md)*
