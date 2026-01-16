# Flask API & Deployment

## Why This Matters for Live Trading

Everything documented so far — the genetic algorithm, market environment, CNN classifier, technical analysis, reward functions — exists for one purpose: **making real trading decisions in real time**.

This API is the bridge between research and production. It takes all the trained models and exposes them through a simple REST endpoint that can be called from any trading system, in any language, on any platform.

**Without this API, you have:**
- Trained models sitting in files
- Jupyter notebooks for backtesting
- No way to act on live market data

**With this API, you have:**
- Real-time trading signals in <500ms
- Integration with any broker or trading platform
- The ability to automate your trading strategy

This is where the research becomes actionable.

---

## API Overview

### The Single Endpoint

```
POST /action/<market_code>
```

That's it. One endpoint that does everything:

1. Receives OHLC market data
2. Processes it through the full pipeline
3. Returns a trading action

### Request Format

```json
{
  "Open":  [1.1200, 1.1210, 1.1230, ...],
  "High":  [1.1250, 1.1260, 1.1270, ...],
  "Low":   [1.1190, 1.1200, 1.1210, ...],
  "Close": [1.1234, 1.1245, 1.1256, ...]
}
```

**Requirements:**
- All four arrays (Open, High, Low, Close) must be present
- Each array must have **at least 347 values**
- Values should be the most recent candles, oldest first
- Currently supports: `EURUSD` (30-minute candles)

### Response Format

```json
{
  "action": "1"
}
```

**Action meanings:**
| Action | Meaning | What to Do |
|--------|---------|------------|
| `"0"` | HOLD | Do nothing, stay out of market |
| `"1"` | LONG | Buy EUR/USD |
| `"2"` | SHORT | Sell EUR/USD |

### Error Responses

```json
{
  "message": "Not enough candles to predict, need at least 347"
}
```

Common errors:
- Insufficient data (need 347+ candles)
- Missing OHLC fields
- Unsupported market code

---

## The Inference Pipeline

When you call the API, your data flows through five stages:

```
OHLC Data (347+ candles)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: PREPROCESSING                                       │
│ • Detrend prices (Close - Previous Close)                   │
│ • Normalize for neural network input                        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: CNN PREDICTION                                      │
│ • Calculate 225 technical indicators (15 × 15)              │
│ • Reshape to 15×15 "image"                                  │
│ • CNN outputs: [P(hold), P(buy), P(sell)]                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: TECHNICAL ANALYSIS                                  │
│ • RSI at intervals [3, 5, 9, 14]                            │
│ • MACD histogram at intervals [3, 5, 9, 14]                 │
│ • 8 additional features                                      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: STATE VECTOR                                        │
│ • Combine: Detrended OHLC (4)                               │
│          + CNN predictions (3)                               │
│          + Technical indicators (8)                          │
│ • Result: 15-dimensional state vector                        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: GA AGENT DECISION                                   │
│ • Neural network: 15 → 16 → 32 → 64 → 32 → 16 → 3          │
│ • Forward pass with ReLU + Softmax                          │
│ • Return: argmax(probabilities)                              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
    Action: 0, 1, or 2
```

**Total processing time:** ~50-200ms (depending on hardware)

---

## Model Loading

### On Startup

When the Flask server starts, it loads everything into memory:

```python
if __name__ == '__main__':
    load_config()           # JSON configuration
    load_hyperparameters()  # state_size=15, action_size=3
    load_ga_agent()         # Evolved neural network weights
    load_cnn()              # Pre-trained CNN model

    app.run(host='0.0.0.0', port=5000)
```

### Configuration File

The system reads from `config-EURUSD-30M_v3.0141.json`:

```json
{
  "agent": {
    "best_agent": "saved_models/DeepNeuro_Gen_best_100.pkl",
    "layers": [16, 32, 64, 32, 16],
    "technical_indicators": ["MACDhist", "RSI"],
    "intervals": [3, 5, 9, 14]
  },
  "model": {
    "pre_trained_model": "cnn-EURUSD_buy_sell_w10-ep500-2020_06_11.model",
    "pre_trained_weights": "cnn-EURUSD_buy_sell_w10-ep500-2020_06_11.h5"
  }
}
```

### Loaded Models

| Model | File | Purpose |
|-------|------|---------|
| GA Agent | `DeepNeuro_Gen_best_100.pkl` | Final trading decisions |
| CNN | `cnn-EURUSD_buy_sell_w10-*.h5` | Pattern recognition signals |

---

## Docker Deployment

### Why Docker?

The system requires specific versions of TensorFlow (1.15), Keras (2.2.4), and TA-Lib (compiled from source). Docker ensures consistent behavior across any machine.

### Production Server

**Build the image:**

```bash
docker build -t ga-server -f docker/server/Dockerfile .
```

**Run the server:**

```bash
docker run --rm -p 5000:5000 ga-server
```

Or use the convenience script:

```bash
./run_server.sh
```

### What's in the Dockerfile

```dockerfile
FROM python:3.7

# Install TA-Lib (requires compilation)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvf ta-lib-0.4.0-src.tar.gz && cd ta-lib && \
    ./configure --prefix=/usr && make && make install

# Python dependencies
RUN pip install Flask keras==2.2.4 tensorflow==1.15 TA-lib pandas numpy

# Copy application
WORKDIR /app
ADD ./app.py /app/
ADD ./GA /app/GA
ADD ./core /app/core
ADD ./cnn_prediction /app/cnn_prediction
ADD ./saved_models /app/saved_models
ADD ./config* /app/

# Start server
CMD ["python", "app.py"]
```

### Development Environment

For Jupyter notebooks and model training:

```bash
docker build -t trading-notebook -f docker/Dockerfile .
docker run -p 8888:8888 -v "$PWD":/home/jovyan/work trading-notebook
```

---

## Live Trading Integration

### Client Implementation

Here's a Python client that integrates with any broker:

```python
import requests
import pandas as pd

class TradingClient:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        self.min_candles = 347

    def get_signal(self, ohlc_data):
        """
        ohlc_data: DataFrame with Open, High, Low, Close columns
                   Must have at least 347 rows
        """
        if len(ohlc_data) < self.min_candles:
            raise ValueError(f"Need at least {self.min_candles} candles")

        payload = {
            'Open': ohlc_data['Open'].tolist(),
            'High': ohlc_data['High'].tolist(),
            'Low': ohlc_data['Low'].tolist(),
            'Close': ohlc_data['Close'].tolist()
        }

        response = requests.post(
            f"{self.api_url}/action/EURUSD",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            return int(response.json()['action'])
        else:
            raise Exception(response.json().get('message', 'API Error'))

    def interpret_action(self, action):
        return {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}[action]
```

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR TRADING SYSTEM                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Broker API   │───►│ Data Buffer  │───►│ API Client   │  │
│  │ (MT4/MT5,    │    │ (347+ candles│    │ (HTTP POST)  │  │
│  │  OANDA, etc) │    │  rolling)    │    │              │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                  │          │
│                                                  ▼          │
│                                         ┌──────────────┐   │
│                                         │ Flask API    │   │
│                                         │ (localhost   │   │
│                                         │  or remote)  │   │
│                                         └──────┬───────┘   │
│                                                  │          │
│                                                  ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Order        │◄───│ Risk         │◄───│ Signal       │  │
│  │ Execution    │    │ Management   │    │ (0, 1, or 2) │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Integration Points

**1. Data Buffer**
- Maintain a rolling window of 347+ candles
- Update on each new candle close
- Trigger API call when new candle completes

**2. Risk Management** (implement yourself)
- Position sizing based on account balance
- Stop-loss and take-profit levels
- Maximum drawdown limits

**3. Order Execution**
- Map API signals to broker orders
- Handle order failures gracefully
- Log all trades for analysis

### Latency Considerations

| Component | Typical Latency |
|-----------|-----------------|
| API Processing | 50-200ms |
| Network (local) | 1-5ms |
| Network (cloud) | 20-100ms |
| **Total** | **<500ms** |

For 30-minute candles, you have plenty of time. The signal should arrive well before the next candle opens.

---

## Security Considerations

### Current Limitations

The API is designed for local/trusted network use. It currently lacks:

- **Authentication**: Anyone who can reach the endpoint can call it
- **Rate limiting**: No protection against abuse
- **HTTPS**: Traffic is unencrypted

### For Production Use

If deploying to a public server, add:

**1. API Key Authentication**
```python
@app.before_request
def check_api_key():
    if request.endpoint == 'signal':
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            return jsonify({'message': 'Unauthorized'}), 401
```

**2. HTTPS via Reverse Proxy (nginx)**
```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:5000;
    }
}
```

**3. Rate Limiting**
```python
from flask_limiter import Limiter
limiter = Limiter(app, default_limits=["60 per minute"])
```

---

## Quick Start Guide

### 1. Build and Run

```bash
# Clone the repository
git clone <repo-url>
cd NeuroEvolutionMarketTrader

# Build Docker image
docker build -t ga-server -f docker/server/Dockerfile .

# Run the server
docker run --rm -p 5000:5000 ga-server
```

### 2. Test the API

```bash
# Health check (server running?)
curl http://localhost:5000/

# Get trading signal (replace with real data)
curl -X POST http://localhost:5000/action/EURUSD \
  -H "Content-Type: application/json" \
  -d '{
    "Open": [1.1200, 1.1210, ... 347+ values ...],
    "High": [1.1250, 1.1260, ... 347+ values ...],
    "Low": [1.1190, 1.1200, ... 347+ values ...],
    "Close": [1.1234, 1.1245, ... 347+ values ...]
  }'
```

### 3. Expected Response

```json
{"action": "1"}
```

This means: **LONG** — the model recommends buying EUR/USD.

---

## Performance in Live Conditions

The model was trained on historical data (2004-2015) and tested on out-of-sample data:

| Metric | Value |
|--------|-------|
| Average Return | 14.17% |
| Win Rate | 80% |
| Max Drawdown | ~3.5% |
| Trades per Year | ~4,000-8,000 |

**Important caveats:**
- Past performance doesn't guarantee future results
- Markets change; the model may need retraining
- Always start with paper trading before live capital
- Use proper position sizing and risk management

---

## Summary

The Flask API transforms trained models into a live trading system:

| Component | Purpose |
|-----------|---------|
| **`app.py`** | REST endpoint serving predictions |
| **Docker** | Reproducible deployment environment |
| **Pipeline** | 5-stage inference from OHLC to action |
| **Response** | Simple JSON: `{"action": "0/1/2"}` |

**To go live:**
1. Deploy the Docker container
2. Connect your broker's data feed
3. Maintain a 347+ candle buffer
4. Call the API on each new candle
5. Execute the recommended action

This API is the final piece that makes the entire system practical for real-world trading.

*Previous: [Reward Functions](./reward-functions.md)*
*Next: [Backtesting](./backtesting.md)*
