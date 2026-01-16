# Configuration System

## Introduction

The configuration system is the **central nervous system** of this project. Two JSON files control everything:

- How the genetic algorithm trains
- What the CNN architecture looks like
- Which technical indicators to use
- How the trading environment behaves
- Where models are saved and loaded

Change a few numbers in these files, and you get a completely different trading system.

### The Two Config Files

| File | Purpose |
|------|---------|
| `config-EURUSD-30M_v3.0141.json` | **Master config** — controls GA training, production API, and ties everything together |
| `predict-cnn-buy-sell-EURUSD-config.json` | **CNN config** — controls CNN architecture and training specifically |

### Versioning

The `v3.0141` in filenames indicates:
- **v3**: Major version (significant architectural changes)
- **.0141**: Iteration number (parameter tuning)

This helps track which configuration produced which results.

---

## The Master Config

**File:** `config-EURUSD-30M_v3.0141.json`

This config has three main sections:

### 1. Agent Section

Controls the Genetic Algorithm and state vector:

```json
{
  "agent": {
    "data": "data/duka/EURUSD_30M-2004_01_01-2015-12-31.csv",
    "test_data": "data/duka/EURUSD-2018_01_01-2018_12_31.csv",
    "best_agent": "saved_models/DeepNeuro_Gen_best_100.pkl",
    "detrend": true,

    "cols": {
      "Close": {"detrend": true},
      "Open": {"detrend": true},
      "High": {"detrend": true},
      "Low": {"detrend": true},
      "CNNClassifier_hold": {"detrend": false},
      "CNNClassifier_buy": {"detrend": false},
      "CNNClassifier_sell": {"detrend": false}
    },

    "technical_indicators": ["MACDhist", "RSI"],
    "intervals": [3, 5, 9, 14],

    "layers": [16, 32, 64, 32, 16],

    "params": {
      "population_size": 256,
      "generations": 100,
      "episodes": 10,
      "mutation_variance": 0.005,
      "survival_ratio": 0.15,
      "reward_function": "SimpleProfit",
      "initial_cash": 5.0,
      "max_env_steps": 500,
      "max_possible_holdings": 10
    }
  }
}
```

**What each part controls:**

| Parameter | What It Does |
|-----------|--------------|
| `data` | Training dataset path |
| `test_data` | Testing dataset path |
| `best_agent` | Where to save/load the best network |
| `cols` | Which columns to detrend (normalize) |
| `technical_indicators` | Which TA indicators to include in state |
| `intervals` | Lookback periods for each indicator |
| `layers` | Hidden layer sizes for the neural network |
| `params` | GA hyperparameters (see tuning guide below) |

### 2. Model Section

Controls the CNN classifier:

```json
{
  "model": {
    "pre_trained_model": "cnn-EURUSD_buy_sell_w10-ep500-2020_06_11.model",
    "pre_trained_weights": "cnn-EURUSD_buy_sell_w10-ep500-2020_06_11.h5",

    "inputs": {
      "technical_indicators": [
        "SMA", "KAMA", "MIDPRICE", "EMA", "WMA",
        "MOM", "ROCP", "TRIX", "ATR", "RSI",
        "ADX", "CCI", "WILLR", "CMO", "AROONOSC"
      ],
      "intervals": [2, 3, 4, 5, 7, 9, 11, 13, 17, 21, 34, 41, 55, 77, 99]
    }
  }
}
```

**Key insight:** The CNN uses **15 indicators × 15 intervals = 225 features** (shaped as a 15×15 "image"), while the GA agent uses only **2 indicators × 4 intervals = 8 features** plus OHLC and CNN outputs.

### 3. Backtesting Section

```json
{
  "backtesting": {
    "initial_cash": 5.0,
    "commission": 0,
    "start_date": "2004-01-01",
    "end_date": "2008-03-01"
  }
}
```

---

## The CNN Config

**File:** `predict-cnn-buy-sell-EURUSD-config.json`

Used only when training the CNN classifier:

```json
{
  "data": {
    "filename": "EURUSD_30M-2004_01_01-2015-12-31.csv",
    "technical_indicators": ["SMA", "KAMA", ...],
    "intervals": [2, 3, 4, 5, 7, 9, 11, 13, 17, 21, 34, 41, 55, 77, 99]
  },

  "model": {
    "batch_size": 32,
    "epochs": 200,
    "lr": 0.001,
    "optimizer": "adam",
    "loss": "sparse_categorical_crossentropy",

    "layers": [
      {"type": "Conv2D", "neurons": 32, "kernel_size": 3},
      {"type": "BatchNormalization"},
      {"type": "MaxPool2D", "pool_size": 2},
      {"type": "dropout", "rate": 0.5},
      {"type": "Conv2D", "neurons": 64, "kernel_size": 3},
      {"type": "BatchNormalization"},
      {"type": "MaxPool2D", "pool_size": 2},
      {"type": "dropout", "rate": 0.5},
      {"type": "flatten"},
      {"type": "dense", "neurons": 128},
      {"type": "dense", "neurons": 64},
      {"type": "dense", "neurons": 3, "activation": "softmax"}
    ]
  }
}
```

The layer-by-layer CNN architecture is fully configurable — you can add/remove layers, change neuron counts, etc.

---

## How Configs Connect the System

```
┌─────────────────────────────────────────────────────────────────┐
│                    config-EURUSD-30M_v3.0141.json               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   agent.data ──────────────► Training Data                      │
│        │                                                        │
│        ▼                                                        │
│   agent.cols ──────────────► Detrend OHLC (4 features)         │
│        │                                                        │
│        │   model.inputs ───► CNN receives 225 features         │
│        │        │                                               │
│        │        ▼                                               │
│        │   model.pre_trained_* ► CNN outputs 3 probabilities   │
│        │        │                                               │
│        │        ▼                                               │
│        │   agent.cols (CNN*) ► Add to state (3 features)       │
│        │        │                                               │
│        ▼        ▼                                               │
│   agent.technical_indicators + agent.intervals                  │
│        │                                                        │
│        ▼                                                        │
│   STATE VECTOR: 4 + 3 + 8 = 15 dimensions                      │
│        │                                                        │
│        ▼                                                        │
│   agent.layers ─────────────► Neural Network [16,32,64,32,16]  │
│        │                                                        │
│        ▼                                                        │
│   agent.params ─────────────► GA Evolution                      │
│        │                                                        │
│        ▼                                                        │
│   agent.best_agent ─────────► Saved Weights                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### State Vector Construction

The 15-dimensional state is built from config:

```python
# From config, construct state vector:
state = [
    # 4 detrended OHLC (from agent.cols where detrend=true)
    Close_detrend, Open_detrend, High_detrend, Low_detrend,

    # 3 CNN outputs (from agent.cols where detrend=false)
    CNNClassifier_hold, CNNClassifier_buy, CNNClassifier_sell,

    # 8 technical indicators (agent.technical_indicators × agent.intervals)
    MACDhist_3, MACDhist_5, MACDhist_9, MACDhist_14,
    RSI_3, RSI_5, RSI_9, RSI_14
]
```

---

## Tuning Guide

### For Faster Training

| Parameter | Default | Faster Setting | Trade-off |
|-----------|---------|----------------|-----------|
| `population_size` | 256 | 128 | Less diversity, may miss good solutions |
| `generations` | 100 | 50 | Less evolution time, may converge early |
| `episodes` | 10 | 5 | Noisier fitness estimates |
| `max_env_steps` | 500 | 250 | Shorter episodes, less context |

**Example — quick experiment:**
```json
"params": {
  "population_size": 128,
  "generations": 50,
  "episodes": 5
}
```

### For Better Results

| Parameter | Default | Better Setting | Trade-off |
|-----------|---------|----------------|-----------|
| `population_size` | 256 | 512 | Slower but more thorough search |
| `generations` | 100 | 200 | More evolution, better convergence |
| `mutation_variance` | 0.005 | 0.01 | More exploration, might destabilize |
| `survival_ratio` | 0.15 | 0.10 | Stronger selection pressure |

**Example — thorough training:**
```json
"params": {
  "population_size": 512,
  "generations": 200,
  "mutation_variance": 0.008,
  "survival_ratio": 0.10
}
```

### For Risk Management

| Parameter | Default | Conservative | Effect |
|-----------|---------|--------------|--------|
| `max_possible_holdings` | 10 | 5 | Smaller position sizes |
| `initial_cash` | 5.0 | 10.0 | Larger portfolio, proportionally smaller trades |
| `lost_all_cash_penalty` | -100 | -200 | Stronger aversion to bankruptcy |

### For CNN Training

| Parameter | Default | Alternative | Effect |
|-----------|---------|-------------|--------|
| `epochs` | 200 | 500 | Longer training, risk of overfitting |
| `lr` | 0.001 | 0.0005 | Slower learning, more stable |
| `batch_size` | 32 | 64 | Faster training, less stable gradients |
| `dropout rate` | 0.5 | 0.3 | Less regularization, more capacity |

---

## Critical Constraints (Don't Break These)

### 1. State Size Must Match

The first layer of the neural network must accept exactly as many inputs as the state vector has dimensions.

**Current state:** 4 (OHLC) + 3 (CNN) + 8 (indicators) = **15**

If you change `technical_indicators` or `intervals`, update `layers` accordingly:

```python
# If you use 3 indicators × 5 intervals:
state_size = 4 + 3 + (3 × 5) = 22

# Then layers should be:
"layers": [22, 32, 64, 32, 16]  # First hidden receives 22 inputs
```

### 2. CNN Indicators Must Match Training

The `model.inputs` in the master config must match what the CNN was trained on. If you train a new CNN with different indicators, update both configs.

### 3. Action Size is Fixed

The output is always 3 actions (HOLD, LONG, SHORT). Don't change the final layer.

### 4. File Paths Must Exist

These paths must point to real files:
- `agent.data`
- `agent.best_agent`
- `model.pre_trained_model`
- `model.pre_trained_weights`

---

## Common Configurations

### Default (Production)

The shipped configuration — balanced for good results:

```json
{
  "population_size": 256,
  "generations": 100,
  "episodes": 10,
  "mutation_variance": 0.005,
  "survival_ratio": 0.15,
  "reward_function": "SimpleProfit"
}
```

### Experimental (Quick Tests)

For rapid iteration during development:

```json
{
  "population_size": 64,
  "generations": 20,
  "episodes": 3,
  "mutation_variance": 0.02
}
```

### Production (Maximum Quality)

When training the final model:

```json
{
  "population_size": 512,
  "generations": 200,
  "episodes": 20,
  "mutation_variance": 0.003,
  "survival_ratio": 0.10
}
```

---

## Adding New Technical Indicators

To add a new indicator to the GA agent:

**1. Update `agent.technical_indicators`:**
```json
"technical_indicators": ["MACDhist", "RSI", "ATR"]  // Added ATR
```

**2. Update state size calculation:**
```
New state = 4 + 3 + (3 indicators × 4 intervals) = 19
```

**3. Update `agent.layers`:**
```json
"layers": [24, 32, 64, 32, 16]  // First layer now larger
```

**4. Retrain the GA agent** — old weights won't work with new state size.

---

## Adding New Intervals

To add more timeframes:

**1. Update `agent.intervals`:**
```json
"intervals": [3, 5, 9, 14, 21, 34]  // Added 21, 34
```

**2. Recalculate state size:**
```
New state = 4 + 3 + (2 × 6) = 19
```

**3. Update layers and retrain.**

---

## Summary

The configuration system controls everything:

| Config File | Controls |
|-------------|----------|
| **Master config** | GA training, state vector, production API |
| **CNN config** | CNN architecture and training |

**Key sections:**
- `agent`: GA parameters, state features, neural network architecture
- `model`: CNN paths and feature inputs
- `params`: Hyperparameters for evolution

**Tuning rules:**
- Faster training: Reduce population, generations, episodes
- Better results: Increase population, generations; tune mutation
- Risk management: Adjust holdings limits and penalties
- **Never break:** State size must match network input size

The current configuration (v3.0141) represents a well-tuned balance that achieves ~14% returns with 80% win rate. Use it as a baseline for your experiments.

*Previous: [Backtesting](./backtesting.md)*
*Next: [Training Pipeline](./training-pipeline.md)*
