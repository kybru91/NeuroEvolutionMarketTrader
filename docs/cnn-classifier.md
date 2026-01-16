# The CNN Classifier

## The Innovation: Features as Images

### A Simple Yet Powerful Idea

What if we could use **image recognition** to recognize **trading patterns**?

That was the breakthrough insight behind this component. Convolutional Neural Networks (CNNs) have achieved remarkable success in image recognition — they can identify faces, objects, and complex visual patterns. But market data isn't images... or is it?

The innovation: **transform financial indicators into a 2D grid that CNNs can "see" as an image**.

Here's the key insight:
- We have **15 technical indicators** (RSI, MACD, SMA, etc.)
- Each calculated at **15 different intervals** (2, 3, 5, 9, 14... up to 99 periods)
- That's **15 × 15 = 225 features** — exactly enough to form a **15×15 pixel image**

```
Traditional approach:
  [feat_1, feat_2, feat_3, ..., feat_225] → Dense Network
  Problem: Loses relationships between features

Image approach:
  ┌─────────────────────────────────────┐
  │ SMA_2   SMA_3   SMA_5  ... SMA_99  │  ← Row 1: SMA at all intervals
  │ KAMA_2  KAMA_3  KAMA_5 ... KAMA_99 │  ← Row 2: KAMA at all intervals
  │ EMA_2   EMA_3   EMA_5  ... EMA_99  │  ← Row 3: EMA at all intervals
  │  ...     ...     ...   ...  ...    │
  │ RSI_2   RSI_3   RSI_5  ... RSI_99  │  ← Row 15: RSI at all intervals
  └─────────────────────────────────────┘
        ↓
  CNN learns spatial patterns
```

By arranging indicators this way, **related features become neighbors**:
- Same indicator, different timeframes → adjacent horizontally
- Different indicators, same timeframe → adjacent vertically
- CNNs naturally learn correlations between neighbors

This transforms decades of computer vision research into a tool for market analysis.

---

## How It Works

### Step 1: Calculate 225 Features

For each moment in time, we calculate 15 indicators at 15 intervals:

**Indicators (15 total):**
| Category | Indicators |
|----------|------------|
| Trend | SMA, KAMA, MIDPRICE, EMA, WMA |
| Momentum | MOM, ROCP, TRIX, RSI, CMO, AROONOSC |
| Volatility | ATR |
| Strength | ADX, CCI, WILLR |

**Intervals (15 total):**
```
[2, 3, 4, 5, 7, 9, 11, 13, 17, 21, 34, 41, 55, 77, 99]
```

This creates a rich feature set capturing market behavior across multiple timeframes — from ultra-short (2 periods) to long-term (99 periods).

### Step 2: Reshape to Image

The magic transformation:

```python
def get_train_data(self, cols, window_size=15):
    # Extract 225 features as 1D array
    x_train = dataframe.get(cols).values  # Shape: (samples, 225)

    # THE KEY: Reshape to 15x15 grid
    x_train = x_train.reshape((samples, 15, 15))

    # Convert to RGB (3 channels) - CNNs expect color images
    x_train = np.stack((x_train,) * 3, axis=-1)  # Shape: (samples, 15, 15, 3)

    return x_train
```

**Before:** `[0.52, 0.48, 0.61, ..., 0.55]` — 225 numbers in a row

**After:**
```
┌──────────────────────────┐
│ 0.52  0.48  0.61  ...    │
│ 0.44  0.51  0.58  ...    │
│ 0.39  0.42  0.55  ...    │
│  ...   ...   ...  ...    │
└──────────────────────────┘
A 15×15 "image" where each pixel is an indicator value
```

### Step 3: CNN Learns Patterns

The CNN processes this "image" through convolutional filters that detect patterns:

```
15×15 Input Image
       ↓
┌─────────────────┐
│ Conv2D Filter   │  Slides across the image looking for patterns
│ (3×3 kernel)    │  like "RSI rising while MACD falling"
└─────────────────┘
       ↓
Pattern detected? → Activate!
       ↓
Multiple layers build up to complex pattern recognition
       ↓
Output: [Hold, Buy, Sell] probabilities
```

---

## Why This Works

### 1. Spatial Relationships Have Meaning

When you place related indicators next to each other, the CNN's convolutional filters can learn their relationships:

```
Example 3×3 filter scanning the grid:

┌─────────────────┐
│ SMA_5  SMA_7    │  Filter detects: "Short-term SMA crossing
│ SMA_9  SMA_11   │   above medium-term SMA" (bullish signal)
│ SMA_13 SMA_17   │
└─────────────────┘
```

The filter learns that when values in the top-left exceed values in the bottom-right, it's a bullish crossover pattern.

### 2. Translation Invariance

A pattern is recognized regardless of where it appears in the grid:

```
┌─────────────────┐      ┌─────────────────┐
│ PATTERN  ...    │  =   │ ...    ...      │
│ ...      ...    │      │ ...  PATTERN    │
│ ...      ...    │      │ ...    ...      │
└─────────────────┘      └─────────────────┘
Same bullish signal detected in both positions
```

### 3. Hierarchical Learning

Each CNN layer learns more abstract patterns:

- **Layer 1**: Simple patterns (single indicator trends)
- **Layer 2**: Combinations (multiple indicators agreeing)
- **Dense layers**: Decision logic (when to buy/sell)

### 4. Parameter Efficiency

A 3×3 convolutional filter has only 9 weights but scans the entire 15×15 grid. Compare to a dense network that would need 225 × hidden_size weights for the first layer alone.

---

## The CNN Architecture

```
Input: (15, 15, 3) — The "indicator image"
           │
           ▼
┌─────────────────────────────────────────────────┐
│ Conv2D: 32 filters, 3×3 kernel                  │
│ → Learns 32 different indicator patterns        │
├─────────────────────────────────────────────────┤
│ BatchNormalization → Stabilizes learning        │
├─────────────────────────────────────────────────┤
│ MaxPool2D: 2×2 → Reduces to (7, 7, 32)         │
├─────────────────────────────────────────────────┤
│ Dropout: 50% → Prevents overfitting            │
└─────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ Conv2D: 64 filters, 3×3 kernel                  │
│ → Combines patterns into higher-level features  │
├─────────────────────────────────────────────────┤
│ BatchNormalization                              │
├─────────────────────────────────────────────────┤
│ MaxPool2D: 2×2 → Reduces to (3, 3, 64)         │
├─────────────────────────────────────────────────┤
│ Dropout: 50%                                    │
└─────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ Flatten → 576 features                          │
├─────────────────────────────────────────────────┤
│ Dense: 128 neurons + BatchNorm + Dropout        │
├─────────────────────────────────────────────────┤
│ Dense: 64 neurons + BatchNorm + Dropout         │
├─────────────────────────────────────────────────┤
│ Dense: 3 neurons + Softmax                      │
│ → Output: [P(Hold), P(Buy), P(Sell)]           │
└─────────────────────────────────────────────────┘
```

### Key Design Choices

| Choice | Reasoning |
|--------|-----------|
| **Two Conv layers** | Enough depth for pattern hierarchy without overfitting |
| **Heavy dropout (50%)** | Financial data is noisy; prevents memorization |
| **Batch normalization** | Stabilizes training, allows higher learning rates |
| **L2 regularization** | Additional overfitting prevention |
| **3-class softmax** | Clear probability distribution over actions |

---

## Generating Labels: What is a "Buy Signal"?

The CNN needs to learn what patterns precede good buying/selling opportunities. Labels are generated by looking at **what actually happened next**:

```python
def _generate_buy_sell_labels(self, close_prices, window_size=10):
    """Look 10 candles into the future to create labels"""
    labels = np.zeros(len(close_prices))

    for i in range(len(close_prices) - window_size):
        future_prices = close_prices[i+1 : i+window_size]

        # BUY signal: current price is lower than ALL future prices
        # (we're at a local bottom)
        if close_prices[i] < future_prices.min():
            labels[i] = 1  # BUY

        # SELL signal: current price is higher than ALL future prices
        # (we're at a local top)
        elif close_prices[i] > future_prices.max():
            labels[i] = 2  # SELL

        # Otherwise: HOLD (label stays 0)

    return labels
```

### Visual Example

```
Price chart:
                    ╱╲
                   ╱  ╲         SELL signal here (local top)
                  ╱    ╲        ↓
            ╱╲   ╱      ╲      ╱╲
           ╱  ╲ ╱        ╲    ╱  ╲
          ╱    ╳          ╲  ╱    ╲
         ╱                 ╲╱
        ╱
       ╱
      ↑
   BUY signal here (local bottom)
```

**Label 1 (BUY):** Current price is the lowest point in the next 10 candles
**Label 2 (SELL):** Current price is the highest point in the next 10 candles
**Label 0 (HOLD):** Neither — don't trade here

This creates a **hindsight-labeled dataset**: we know what the "right" answer was because we can see the future. The CNN learns to recognize the patterns that preceded these optimal entry/exit points.

---

## Training the CNN

### Training Configuration

```python
epochs = 200
batch_size = 64
learning_rate = 0.001
optimizer = Adam
loss = sparse_categorical_crossentropy
```

### Handling Class Imbalance

Markets spend most of their time in "HOLD" territory — clear buy/sell signals are rare. The data loader balances classes:

```python
def _handle_class_unbalancing(self, x, y):
    """Ensure equal samples of each class"""
    # Count samples per class
    hold_count = sum(y == 0)
    buy_count = sum(y == 1)
    sell_count = sum(y == 2)

    # Use the minimum count
    min_count = min(hold_count, buy_count, sell_count)

    # Sample equally from each class
    balanced_x, balanced_y = sample_equally(x, y, min_count)

    return balanced_x, balanced_y
```

If we have 10,000 HOLDs, 500 BUYs, and 400 SELLs, we keep 400 of each class. This prevents the CNN from just predicting "HOLD" all the time.

### Training Metrics

The model tracks multiple metrics during training:

```python
metrics = ['accuracy', 'f1_score', 'precision', 'recall']
```

**F1-score** is particularly important for imbalanced classification — it balances precision (not giving false signals) with recall (not missing real signals).

---

## Integration with the Trading Agent

The CNN doesn't make final trading decisions — it provides **advisory signals** to the genetic algorithm agent.

### The Two-Stage Architecture

```
Stage 1: CNN Pattern Recognition
┌─────────────────────────────────────────────────┐
│  Market Data → 225 Features → 15×15 Image      │
│                     ↓                           │
│              CNN Forward Pass                   │
│                     ↓                           │
│  Output: [0.1, 0.6, 0.3] (Hold, Buy, Sell)     │
└─────────────────────────────────────────────────┘
                      │
                      ▼
Stage 2: GA Agent Final Decision
┌─────────────────────────────────────────────────┐
│  State = [OHLC, Indicators, CNN_predictions]    │
│                     ↓                           │
│           Evolved Neural Network                │
│                     ↓                           │
│       Final Action: LONG / SHORT / SIT          │
└─────────────────────────────────────────────────┘
```

### Why Two Stages?

The CNN excels at pattern recognition but doesn't understand:
- Current portfolio state (already holding positions?)
- Risk management (how much to bet?)
- When to take profits vs let winners run

The GA agent learns these higher-level decisions, using CNN signals as one of many inputs.

### Code Integration

```python
# In the Flask API endpoint:

# 1. Get CNN prediction
cnn_input = reshape_features_to_image(market_data)  # (1, 15, 15, 3)
cnn_prediction = model.predict(cnn_input)[-1]        # [0.1, 0.6, 0.3]

# 2. Add CNN outputs to state vector
state = [
    close_price,      # Current price
    rsi_14,           # Technical indicator
    macd_histogram,   # Technical indicator
    # ... more indicators ...
    cnn_prediction[0],  # CNN's hold probability
    cnn_prediction[1],  # CNN's buy probability
    cnn_prediction[2],  # CNN's sell probability
]

# 3. GA agent makes final decision
action = ga_agent.act(state)  # Returns 0, 1, or 2

# 4. Execute trade
return {'action': action}
```

---

## Transfer Learning Variant

For scenarios with limited market data, a transfer learning approach is available using VGG16 (pre-trained on ImageNet):

```python
class CNNModelTransfer():
    def __init__(self, input_shape):
        # Load VGG16 without its classification head
        self.base_model = keras.applications.VGG16(
            weights='imagenet',      # Pre-trained weights
            include_top=False,       # Remove classification layers
            input_tensor=Input(shape=input_shape)
        )

        # Add custom layers for market classification
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(3, activation='softmax')(x)  # Hold/Buy/Sell

        self.model = Model(inputs=self.base_model.input, outputs=x)

    def freeze_base(self):
        """Freeze pre-trained layers during initial training"""
        for layer in self.base_model.layers:
            layer.trainable = False
```

**Why this works:** VGG16 learned to detect edges, textures, and patterns from millions of images. These low-level features transfer surprisingly well to other domains — including financial "images."

---

## Results and Performance

The trained CNN model provides:

- **Pattern recognition** that complements technical indicator rules
- **Probabilistic confidence** rather than binary signals
- **Multi-timeframe awareness** through the 15-interval feature grid

When integrated with the GA agent, the combined system achieved:
- **80% win rate** on trades
- **14.17% average return** on test data
- Consistent performance across different market conditions

---

## The Bigger Picture

This CNN represents a **creative domain transfer**:

| Traditional ML | This Approach |
|----------------|---------------|
| Treat indicators as independent features | Treat indicators as spatial relationships |
| Use dense networks | Use convolutional networks |
| Learn feature weights | Learn visual patterns |
| Manual feature engineering | Automatic pattern discovery |

By "seeing" market data as images, we leverage decades of computer vision research for financial pattern recognition — finding relationships that traditional approaches might miss.

---

## Summary

The CNN Classifier transforms market analysis into image recognition:

1. **225 features** (15 indicators × 15 intervals) become a **15×15 image**
2. **CNN architecture** learns spatial patterns between indicators
3. **Retrospective labels** teach the network what precedes good trades
4. **Class balancing** ensures the model doesn't just predict "HOLD"
5. **Two-stage integration** combines pattern recognition with trading logic

The key innovation: treating numerical features as spatial pixels, enabling CNNs to discover patterns invisible to traditional machine learning approaches.

*Previous: [The Market Environment](./market-environment.md)*
*Next: [Technical Analysis](./technical-analysis.md)*
