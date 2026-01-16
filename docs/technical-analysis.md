# Technical Analysis

## Introduction

### What is Technical Analysis?

Technical analysis is the art of reading market data to predict future price movements. Instead of analyzing company fundamentals (earnings, revenue, etc.), technical analysts study **price patterns**, **trends**, and **momentum** — the collective psychology of millions of traders encoded in price charts.

This project uses **15 technical indicators** — mathematical formulas applied to price data — each calculated at **15 different time intervals**. This creates 225 features that capture market behavior from multiple perspectives simultaneously.

### Why Technical Indicators?

Raw prices alone don't tell the full story:

| Raw Data | What It Misses |
|----------|----------------|
| "Price is $1.20" | Is that high or low? Relative to what? |
| "Price went up $0.01" | Is that a big move? Is it accelerating? |
| "Today's high was $1.22" | Is that unusual? Is volatility increasing? |

Technical indicators answer these questions by **transforming** raw data into meaningful signals:

- **Is price above or below its average?** → Moving averages
- **Is momentum increasing or decreasing?** → Momentum oscillators
- **Is the market overbought or oversold?** → RSI, Williams %R
- **How volatile is the market?** → ATR

---

## The 15 Indicators

### Overview

The system uses 15 carefully chosen indicators covering four categories:

| Category | Indicators | What They Measure |
|----------|------------|-------------------|
| **Trend** | SMA, KAMA, EMA, WMA, MIDPRICE | Direction of the market |
| **Momentum** | MOM, ROCP, TRIX, CMO | Speed of price changes |
| **Oscillators** | RSI, CCI, WILLR, AROONOSC | Overbought/oversold conditions |
| **Volatility/Strength** | ATR, ADX | Market energy and trend power |

---

### Trend Indicators

These indicators smooth out price noise to reveal the underlying direction.

#### SMA (Simple Moving Average)

**What it measures:** The average closing price over the last N periods.

**Intuition:** If a stock has been trading around $100 for the past 20 days, the 20-period SMA is about $100. If today's price is $105, the stock is "above its average" — potentially bullish.

```python
SMA_5 = average of last 5 closing prices
```

**Signal:** Price above SMA = bullish, Price below SMA = bearish

---

#### EMA (Exponential Moving Average)

**What it measures:** A weighted average that gives more importance to recent prices.

**Intuition:** EMA reacts faster than SMA to recent price changes. If the market suddenly drops, EMA will reflect that sooner.

```python
EMA = (Today's Price × Weight) + (Yesterday's EMA × (1 - Weight))
```

**Signal:** Faster trend changes than SMA; crossovers indicate momentum shifts.

---

#### KAMA (Kaufman Adaptive Moving Average)

**What it measures:** An intelligent moving average that adapts to market conditions.

**Intuition:** In choppy, sideways markets, KAMA slows down to avoid false signals. In trending markets, KAMA speeds up to capture the move.

**Signal:** Reduces whipsaws in ranging markets while staying responsive in trends.

---

#### WMA (Weighted Moving Average)

**What it measures:** A moving average where recent prices have linearly higher weights.

**Intuition:** A compromise between SMA (all weights equal) and EMA (exponential weights). Moderately responsive to recent changes.

---

#### MIDPRICE

**What it measures:** The midpoint between the highest high and lowest low over N periods.

**Intuition:** Represents the "center of gravity" of recent trading range. Prices tend to oscillate around this level.

```python
MIDPRICE = (Highest High + Lowest Low) / 2
```

---

### Momentum Indicators

These indicators measure the **speed** and **strength** of price movements.

#### MOM (Momentum)

**What it measures:** The raw change in price over N periods.

```python
MOM = Today's Close - Close N periods ago
```

**Intuition:** If price was $100 five days ago and is $105 today, MOM = +5. Positive momentum means price is rising; negative means falling.

---

#### ROCP (Rate of Change Percentage)

**What it measures:** The percentage change in price over N periods.

```python
ROCP = (Today's Close - Close N periods ago) / Close N periods ago × 100
```

**Intuition:** Normalizes momentum as a percentage, making it comparable across different price levels. A 5% move is significant whether the stock is at $10 or $100.

---

#### TRIX (Triple Exponential Moving Average)

**What it measures:** The rate of change of a triple-smoothed EMA.

**Intuition:** Extreme smoothing filters out almost all noise, showing only significant trend changes. When TRIX crosses zero, it signals a major trend shift.

---

#### CMO (Chande Momentum Oscillator)

**What it measures:** The balance between upward and downward price movements.

```python
CMO = (Sum of Up Days - Sum of Down Days) / (Sum of Up Days + Sum of Down Days)
```

**Intuition:** Ranges from -100 to +100. High positive values mean most recent moves were up; high negative values mean most were down.

---

### Oscillators

These indicators identify **overbought** and **oversold** conditions — when prices have moved too far, too fast.

#### RSI (Relative Strength Index)

**What it measures:** The ratio of average gains to average losses.

```python
RSI = 100 - (100 / (1 + Average Gain / Average Loss))
```

**Intuition:** Ranges from 0 to 100.
- **Above 70**: Overbought — price may have risen too fast, pullback likely
- **Below 30**: Oversold — price may have fallen too fast, bounce likely

This is one of the most widely used indicators in trading.

---

#### CCI (Commodity Channel Index)

**What it measures:** How far price has deviated from its statistical mean.

**Intuition:** When CCI is above +100, price is unusually high relative to recent history. When below -100, price is unusually low. Mean reversion traders look for these extremes.

---

#### WILLR (Williams %R)

**What it measures:** Where today's close is relative to the recent high-low range.

```python
Williams %R = -100 × (Highest High - Close) / (Highest High - Lowest Low)
```

**Intuition:** Ranges from -100 to 0.
- **Above -20**: Overbought (price near recent highs)
- **Below -80**: Oversold (price near recent lows)

---

#### AROONOSC (Aroon Oscillator)

**What it measures:** How recently the highest high and lowest low occurred.

**Intuition:** If the highest high was yesterday and the lowest low was 10 days ago, we're in an uptrend. AROONOSC captures this as a single number from -100 to +100.

---

### Volatility and Strength

#### ATR (Average True Range)

**What it measures:** The average size of price swings over N periods.

```python
True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
ATR = Average of True Range over N periods
```

**Intuition:** High ATR means the market is volatile — big moves are normal. Low ATR means the market is calm. Useful for setting stop-loss distances proportional to current volatility.

---

#### ADX (Average Directional Index)

**What it measures:** The **strength** of a trend, regardless of direction.

**Intuition:** Ranges from 0 to 100.
- **Below 20**: Weak trend or ranging market
- **Above 25**: Strong trend in progress
- **Above 50**: Very strong trend

ADX doesn't tell you if the trend is up or down — just how strong it is.

---

## The 15 Intervals

### Multi-Timeframe Analysis

Each indicator is calculated at **15 different time intervals**:

```
[2, 3, 4, 5, 7, 9, 11, 13, 17, 21, 34, 41, 55, 77, 99]
```

This creates a **multi-timeframe view** of the market:

| Range | Intervals | What It Captures |
|-------|-----------|------------------|
| **Ultra-short** | 2, 3, 4, 5 | Immediate price reactions, noise |
| **Short-term** | 7, 9, 11, 13 | Intraday trends, quick momentum |
| **Medium-term** | 17, 21 | Multi-hour patterns |
| **Long-term** | 34, 41, 55, 77, 99 | Daily and multi-day trends |

### Why Multiple Timeframes?

A professional trader doesn't just look at one chart — they check the 5-minute, hourly, daily, and weekly views. Our system does the same automatically:

```
RSI_5  = 65 → Short-term: slightly overbought
RSI_14 = 55 → Medium-term: neutral
RSI_99 = 42 → Long-term: slightly oversold
```

The neural network sees all three perspectives simultaneously and learns which combinations predict profitable trades.

### Fibonacci-Inspired Spacing

The intervals approximate Fibonacci numbers (2, 3, 5, 8, 13, 21, 34, 55, 89...). This isn't mystical — Fibonacci spacing naturally captures different market cycles without redundancy. Each interval adds new information rather than duplicating what shorter intervals already show.

---

## The Detrending Approach

### The Problem with Raw Prices

Raw prices are **non-stationary** — they trend over time. A neural network trained on EUR/USD at 1.10 won't work well when EUR/USD moves to 1.20.

**Solution:** Instead of using raw indicator values, we use **the difference from the current price**:

```python
# Instead of: "SMA is 1.1980"
# We compute: "Price is 0.0020 above SMA"

feature = Close - SMA
```

### Why This Works

1. **Stationarity**: The difference oscillates around zero regardless of absolute price level
2. **Meaningful signal**: Positive = price above average (bullish), Negative = price below (bearish)
3. **Comparable scale**: Works whether EUR/USD is at 1.10 or 1.30

### Code Example

```python
def add_indicator(df, indicator, interval, true_value=False):
    if not true_value:
        # DETRENDED: Difference from close price
        df[f"SMA_{interval}"] = df["Close"] - talib.SMA(df["Close"], timeperiod=interval)
    else:
        # RAW: Actual indicator value
        df[f"SMA_{interval}"] = talib.SMA(df["Close"], timeperiod=interval)
```

### Visual Example

```
Raw SMA values over time:
    1.2020, 1.2015, 1.2025, 1.2030, 1.2018...  (trending, non-stationary)

Detrended (Close - SMA):
    +0.0005, -0.0010, +0.0008, -0.0003, +0.0012...  (oscillating around 0)
```

The detrended version is much easier for neural networks to learn from.

---

## Feature Generation Pipeline

### The Complete Flow

```
           Raw OHLC Data
                │
                ▼
        ┌───────────────┐
        │   Detrend     │  Close - Close.shift()
        │   OHLC        │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │   Calculate   │  15 indicators × 15 intervals
        │   225 Features│  = 225 features
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │   Normalize   │  RSI/100, ADX/100, etc.
        │   Scales      │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │   Remove NaN  │  First ~99 rows (warmup)
        │   Rows        │
        └───────┬───────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
    CNN Input       Agent State
   (225 features)   (15 features)
```

### CNN Feature Set (Full 225)

The CNN receives **all 225 features** arranged as a 15×15 "image":

```python
indicators = ["SMA", "KAMA", "MIDPRICE", "EMA", "WMA",
              "MOM", "ROCP", "TRIX", "ATR", "RSI",
              "ADX", "CCI", "WILLR", "CMO", "AROONOSC"]

intervals = [2, 3, 4, 5, 7, 9, 11, 13, 17, 21, 34, 41, 55, 77, 99]

# Generate column names
cnn_cols = [f"{ind}_{intv}" for ind in indicators for intv in intervals]
# Result: ["SMA_2", "SMA_3", ..., "AROONOSC_99"] — 225 columns
```

### Agent Feature Set (Selective 15)

The GA agent receives a **curated subset** of 15 features:

```python
agent_state = [
    # Detrended OHLC (4 features)
    Close_detrend, Open_detrend, High_detrend, Low_detrend,

    # CNN predictions (3 features)
    CNN_hold_prob, CNN_buy_prob, CNN_sell_prob,

    # Selected indicators (8 features)
    MACDhist_3, MACDhist_5, MACDhist_9, MACDhist_14,
    RSI_3, RSI_5, RSI_9, RSI_14
]
```

**Why the difference?**
- CNN needs all features to learn visual patterns
- Agent needs fast, interpretable decisions with key signals

---

## Normalization Details

### Keeping Features Comparable

Different indicators have different ranges. We normalize them:

| Indicator | Raw Range | After Normalization |
|-----------|-----------|---------------------|
| RSI | 0 to 100 | 0 to 1 (÷100) |
| ADX | 0 to 100 | 0 to 1 (÷100) |
| CCI | -300 to +300 | -3 to +3 (÷100) |
| Williams %R | -100 to 0 | -1 to 0 (÷100) |
| CMO | -100 to +100 | -1 to +1 (÷100) |
| AROONOSC | -100 to +100 | -1 to +1 (÷100) |
| SMA, EMA, etc. | Detrended | Small decimals (~0.001) |

```python
# Normalization in code
df["RSI_5"] = talib.RSI(df["Close"], timeperiod=5) / 100.0
df["ADX_5"] = talib.ADX(df["High"], df["Low"], df["Close"], timeperiod=5) / 100.0
df["CCI_5"] = talib.CCI(df["High"], df["Low"], df["Close"], timeperiod=5) / 100.0
```

This ensures no single indicator dominates the neural network's learning just because it has larger numbers.

---

## TA-Lib: The Engine

### What is TA-Lib?

TA-Lib (Technical Analysis Library) is an industry-standard open-source library used by professional traders and quant firms. It provides:

- **200+ technical indicators**
- **Implemented in C** for speed
- **Battle-tested** accuracy
- **Easy Python interface**

### Usage Pattern

```python
import talib
import numpy as np

# Input: NumPy arrays of price data
close = np.array([1.1990, 1.2005, 1.2010, 1.1995, ...])
high = np.array([1.2020, 1.2030, 1.2025, 1.2010, ...])
low = np.array([1.1980, 1.1990, 1.2000, 1.1985, ...])

# Calculate indicators
sma = talib.SMA(close, timeperiod=14)
rsi = talib.RSI(close, timeperiod=14)
atr = talib.ATR(high, low, close, timeperiod=14)
macd, signal, hist = talib.MACD(close)
```

### Docker Setup

TA-Lib requires compilation. The project includes Docker configuration to handle this:

```dockerfile
# From docker/Dockerfile
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install
```

---

## Putting It All Together

### From Raw Data to Trading Decision

```python
# 1. Load 30-minute EURUSD candles
df = pd.read_csv("EURUSD_30M.csv")

# 2. Generate 225 technical features
df = add_technical_indicators_with_intervals(
    df,
    indicators=["SMA", "KAMA", "MIDPRICE", "EMA", "WMA",
                "MOM", "ROCP", "TRIX", "ATR", "RSI",
                "ADX", "CCI", "WILLR", "CMO", "AROONOSC"],
    intervals=[2, 3, 4, 5, 7, 9, 11, 13, 17, 21, 34, 41, 55, 77, 99]
)

# 3. CNN sees the 225 features as a 15×15 image
cnn_input = df[cnn_cols].values.reshape(-1, 15, 15, 3)
cnn_prediction = cnn_model.predict(cnn_input)  # [hold, buy, sell]

# 4. Agent combines CNN output with key indicators
state = [
    df["Close_detrend"].iloc[-1],
    df["RSI_14"].iloc[-1],
    df["MACDhist_9"].iloc[-1],
    cnn_prediction[0],  # hold probability
    cnn_prediction[1],  # buy probability
    cnn_prediction[2],  # sell probability
    # ... more features
]

# 5. Evolved neural network makes final decision
action = agent.act(state)  # 0=SIT, 1=LONG, 2=SHORT
```

---

## Summary

Technical Analysis provides the **language** that the trading system uses to understand markets:

1. **15 Indicators** covering trend, momentum, oscillators, and volatility
2. **15 Intervals** from ultra-short (2) to long-term (99) for multi-timeframe analysis
3. **225 Features** (15 × 15) that capture market state comprehensively
4. **Detrending** normalizes features for stable neural network training
5. **TA-Lib** provides fast, accurate indicator calculations

These features feed into both the CNN (for pattern recognition) and the evolved trading agent (for decision making), creating a system that sees the market through the lens of professional technical analysis.

*Previous: [The CNN Classifier](./cnn-classifier.md)*
*Next: [Reward Functions](./reward-functions.md)*
