# Training Pipeline

## Overview

Training this system is a **two-stage process**:

1. **Stage 1: Train the CNN** — Learns to recognize buy/sell patterns
2. **Stage 2: Train the GA Agent** — Learns when to act on those patterns

The CNN must be trained first because its predictions become input features for the GA agent.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   STAGE 1: CNN TRAINING                                         │
│   ─────────────────────                                         │
│   Input:  Raw OHLC data (2004-2015)                            │
│   Output: Trained CNN model (.h5)                              │
│   Time:   ~30-60 minutes                                        │
│                                                                 │
│                         ↓                                       │
│                                                                 │
│   STAGE 2: GA TRAINING                                          │
│   ────────────────────                                          │
│   Input:  OHLC + CNN predictions + Technical indicators        │
│   Output: Evolved trading network (.pkl)                       │
│   Time:   ~2-8 hours                                           │
│                                                                 │
│                         ↓                                       │
│                                                                 │
│   READY FOR: Backtesting → Live Trading                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: CNN Training

### Purpose

The CNN learns to predict whether the current moment is a good time to buy, sell, or hold. It looks at 225 technical indicators arranged as a 15×15 "image" and outputs three probabilities.

### Notebook

**File:** `CNN_buy_sell_classifier.ipynb`

### Step-by-Step Process

#### 1. Load Configuration

```python
configs = json.load(open('predict-cnn-buy-sell-EURUSD-config.json'))
```

#### 2. Prepare Data

```python
from cnn_prediction.cnn_data_loader_buy_sell import CNNDataLoaderBuySell

dataloader = CNNDataLoaderBuySell(
    df_train,
    indicators=configs['data']['technical_indicators'],  # 15 indicators
    intervals=configs['data']['intervals'],              # 15 intervals
    window_size=10
)

x_train, y_train = dataloader.get_train_data(
    cols=cnn_cols,      # 225 feature columns
    window_size=15      # Reshape to 15×15
)
```

**What happens:**
- 225 technical indicators are calculated (15 indicators × 15 intervals)
- Data is reshaped to 15×15×3 "images"
- Labels are generated: 1=BUY (price will rise), 2=SELL (price will fall), 0=HOLD

#### 3. Create Model

```python
from cnn_prediction.cnn_model import CNNModel

model = CNNModel()
model.create_model_cnn(configs['model'], input_shape=(15, 15, 3))
```

**Architecture:**
```
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.5)
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.5)
Flatten → Dense(128) → Dense(64) → Dense(3, softmax)
```

#### 4. Train

```python
history = model.train(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    x_val=x_test,
    y_val=y_test
)
```

**Training output:**
```
Epoch 1/200 - loss: 1.10 - accuracy: 0.36 - val_loss: 1.09
Epoch 2/200 - loss: 1.08 - accuracy: 0.37 - val_loss: 1.08
...
Epoch 200/200 - loss: 0.95 - accuracy: 0.45 - val_loss: 0.98
```

#### 5. Save Model

```python
model.save('saved_models/cnn-EURUSD_buy_sell_w10-ep200.h5')
```

### Expected Duration

| Hardware | Time per Epoch | Total (200 epochs) |
|----------|----------------|-------------------|
| CPU only | ~60 seconds | ~3 hours |
| GPU | ~10-15 seconds | ~30-50 minutes |

### Output Files

```
saved_models/
├── cnn-EURUSD_buy_sell_w10-ep200.model  # Architecture
└── cnn-EURUSD_buy_sell_w10-ep200.h5     # Weights
```

---

## Stage 2: GA Training

### Purpose

The GA evolves a population of 256 neural networks, selecting the most profitable traders over 100 generations. Each network receives:
- Detrended OHLC prices (4 features)
- CNN predictions (3 features)
- Technical indicators (8 features)

Total: **15 input features → 3 output actions**

### Notebook

**File:** `Stock_trading_v3.0141.ipynb`

### Step-by-Step Process

#### 1. Load Configuration

```python
configs = json.load(open('config-EURUSD-30M_v3.0141.json'))
```

#### 2. Prepare Data

```python
# Load raw data
df_train = pd.read_csv(configs['agent']['data'])

# Add technical indicators
df_train = add_technical_indicators_with_intervals(
    df_train,
    indicators=['MACDhist', 'RSI'],
    intervals=[3, 5, 9, 14]
)

# Detrend OHLC
for col in ['Close', 'Open', 'High', 'Low']:
    df_train[col + '_detrend'] = df_train[col] - df_train[col].shift()
```

#### 3. Load Pre-trained CNN

```python
from cnn_prediction.cnn_model import CNNModel

cnn_model = CNNModel('saved_models')
cnn_model.load([
    configs['model']['pre_trained_model'],
    configs['model']['pre_trained_weights']
])
```

#### 4. Generate CNN Predictions

```python
# Prepare CNN input (225 features → 15×15×3)
x_cnn = prepare_cnn_input(df_train)

# Get predictions
predictions = cnn_model.predict(x_cnn)

# Add to dataframe
df_train['CNNClassifier_hold'] = predictions[:, 0]
df_train['CNNClassifier_buy'] = predictions[:, 1]
df_train['CNNClassifier_sell'] = predictions[:, 2]
```

#### 5. Build Feature Matrix

```python
cols = [
    'Close_detrend', 'Open_detrend', 'High_detrend', 'Low_detrend',
    'CNNClassifier_hold', 'CNNClassifier_buy', 'CNNClassifier_sell',
    'MACDhist_3', 'MACDhist_5', 'MACDhist_9', 'MACDhist_14',
    'RSI_3', 'RSI_5', 'RSI_9', 'RSI_14'
]

x_train = df_train[cols].values  # Shape: (n_samples, 15)
```

#### 6. Initialize GA Population

```python
from GA.ga import GeneticNetworks

ga = GeneticNetworks(
    architecture=(15, 16, 32, 64, 32, 16, 3),  # Network shape
    population_size=256,
    generations=100,
    episodes=10,
    mutation_variance=0.005,
    survival_ratio=0.15
)
```

#### 7. Configure Environment

```python
env_args = {
    'x': x_train,
    'actions': [[1,0,0], [0,1,0], [0,0,1]],
    'state_size': 15,
    'initial_cash': 5.0,
    'reward_function': 'SimpleProfit',
    'max_possible_holdings': 10
}
```

#### 8. Run Evolution

```python
# Setup TensorBoard logging
summary_writer = tf.summary.FileWriter('GA/tensorboard_Market_v3.0141/')

# Start training
ga.fit(
    env=None,
    summary_writer=summary_writer,
    is_market=True,
    env_args=env_args,
    num_cpus=4
)
```

**Training output:**
```
Generation:1   | Max: 0.05 | Mean: -0.12 | Stagnation: 1
Generation:2   | Max: 0.08 | Mean: -0.08 | Stagnation: 2
Generation:3   | Max: 0.12 | Mean: -0.05 | Stagnation: 3
...
Generation:50  | Max: 0.45 | Mean: 0.22  | Stagnation: 1
...
Generation:100 | Max: 0.52 | Mean: 0.31  | Stagnation: 5
```

### What Happens Each Generation

```
Generation N:
│
├── 1. EVALUATE (parallel on 4 CPUs)
│   └── Each of 256 networks runs 10 episodes
│       └── Each episode: up to 500 trading steps
│       └── Fitness = average profit across episodes
│
├── 2. SELECT
│   └── Keep top 15% (39 networks)
│
├── 3. REPRODUCE (create 217 new networks)
│   ├── 80% (174): Crossover from 2 parents + mutation
│   ├── 10% (22):  Clone 1 parent + mutation
│   └── 10% (21):  Clone 1 parent (no mutation)
│
├── 4. LOG
│   └── TensorBoard: max, mean, std fitness
│
└── 5. CHECKPOINT
    └── Save best and last network
```

### Expected Duration

| Hardware | Time per Generation | Total (100 generations) |
|----------|--------------------|-----------------------|
| 4 CPU cores | ~2-5 minutes | ~3-8 hours |
| 8 CPU cores | ~1-3 minutes | ~2-5 hours |

### Output Files

```
saved_models/
├── DeepNeuro_Gen_best_1.pkl    # Best at generation 1
├── DeepNeuro_Gen_last_1.pkl    # Last at generation 1
├── DeepNeuro_Gen_best_2.pkl
├── DeepNeuro_Gen_last_2.pkl
...
├── DeepNeuro_Gen_best_100.pkl  # Final best
└── DeepNeuro_Gen_last_100.pkl  # Final last
```

---

## TensorBoard Monitoring

During GA training, TensorBoard logs are written for real-time monitoring:

```bash
# Start TensorBoard
tensorboard --logdir GA/tensorboard_Market_v3.0141/

# Open in browser
http://localhost:6006
```

**Metrics tracked:**
- **Max Reward**: Best fitness in each generation (should increase)
- **Mean Reward**: Average fitness (should increase)
- **Std Reward**: Diversity of population (should decrease as it converges)

---

## Complete Execution Guide

### Prerequisites

1. **Docker environment running** (for TA-Lib)
2. **Data files in place:**
   - `data/duka/EURUSD_30M-2004_01_01-2015-12-31.csv` (training)
   - `data/duka/EURUSD-2018_01_01-2018_12_31.csv` (testing)

### Step 1: Train CNN

```bash
# Start Jupyter
docker run -p 8888:8888 -v "$PWD":/home/jovyan/work trading-notebook

# Open CNN_buy_sell_classifier.ipynb
# Run all cells
# Wait ~30-60 minutes
# Verify: saved_models/cnn-*.h5 exists
```

### Step 2: Train GA

```bash
# Open Stock_trading_v3.0141.ipynb
# Run all cells
# Wait ~2-8 hours
# Monitor: TensorBoard at localhost:6006
# Verify: saved_models/DeepNeuro_Gen_best_100.pkl exists
```

### Step 3: Select Best Model

```python
# In Stock_trading_v3.0141.ipynb, cells [41-43]

# Test all 200 checkpoints
best = None
for gen in range(1, 101):
    for mode in ['best', 'last']:
        ga.load_weights(f'saved_models/DeepNeuro_Gen_{mode}_{gen}.pkl')
        profit = evaluate_on_test_data(ga.best_network)
        if best is None or profit > best[0]:
            best = (profit, gen, mode)

print(f"Best model: Generation {best[1]}, {best[2]} with profit {best[0]}")
```

### Step 4: Validate

```bash
# Open Backtesting_rl_model_v3.0141.ipynb
# Run all cells
# Check: Win rate > 70%, positive returns
```

---

## Troubleshooting

### CNN Training Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Accuracy stuck at 33% | Class imbalance | Enable class balancing in data loader |
| Out of memory | Dataset too large | Use generator-based training |
| Loss not decreasing | Learning rate too high | Reduce `lr` to 0.0001 |

### GA Training Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Fitness not improving | Stuck in local optimum | Increase `mutation_variance` |
| All networks same fitness | Diversity lost | Increase `survival_ratio` |
| Training too slow | CPU bottleneck | Increase `num_cpus` |
| Early stopping | Stagnation detected | Set `stagnation_end=False` |

### Common Errors

**"CNN model not found"**
```
Solution: Run CNN training notebook first, verify saved_models/ contains .h5 file
```

**"State size mismatch"**
```
Solution: Ensure agent.layers[0] matches number of features (15)
Check: 4 OHLC + 3 CNN + (indicators × intervals) = input size
```

**"Not enough candles"**
```
Solution: Data needs warmup period for indicators
Use start_index > 100 to skip warmup
```

---

## Timeline Summary

| Phase | Duration | Output |
|-------|----------|--------|
| CNN Training | 30-60 min | `cnn-*.h5` |
| GA Training | 2-8 hours | `DeepNeuro_Gen_best_100.pkl` |
| Model Selection | 10-30 min | Best checkpoint identified |
| Backtesting | 5-10 min | Performance validated |
| **Total** | **3-10 hours** | **Production-ready model** |

---

## Summary

The training pipeline has two sequential stages:

**Stage 1 (CNN):**
- Transforms 225 indicators into buy/sell probabilities
- Trained with supervised learning on labeled data
- Output becomes input feature for Stage 2

**Stage 2 (GA):**
- Evolves 256 trading strategies over 100 generations
- Uses CNN predictions + technical indicators
- Selects strategies by actual trading profit

The result is a trading system that combines deep learning pattern recognition (CNN) with evolutionary optimization (GA) — each component doing what it does best.

*Previous: [Configuration System](./configuration.md)*

---

## Documentation Complete

This concludes the NeuroEvolutionMarketTrader documentation series:

1. [Introduction](./introduction.md) — Project overview and journey
2. [Genetic Algorithm](./genetic-algorithm.md) — Evolution-based learning
3. [Market Environment](./market-environment.md) — Trading simulation
4. [CNN Classifier](./cnn-classifier.md) — Pattern recognition (features as images)
5. [Technical Analysis](./technical-analysis.md) — 15 indicators × 15 intervals
6. [Reward Functions](./reward-functions.md) — What "good trading" means
7. [Flask API & Deployment](./deployment.md) — Live trading integration
8. [Backtesting](./backtesting.md) — Historical validation
9. [Configuration](./configuration.md) — System settings and tuning
10. [Training Pipeline](./training-pipeline.md) — End-to-end workflow
