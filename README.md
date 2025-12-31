# XAU/USD AI Trading System

An AI-powered gold (XAU/USD) trading system with real-time signals, automated execution, and self-improving machine learning models.

## 🎯 Project Goals

- **80%+ prediction accuracy** through ensemble ML models
- **Self-improving system** without human intervention
- **Support for volatility and investment trading styles**
- **Real-time market data and signal generation**
- **Comprehensive risk management**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Flutter PWA Frontend                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │Dashboard │ │ Signals  │ │ Trading  │ │Portfolio │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────┬───────────────────────────────────────┘
                          │ WebSocket / REST API
┌─────────────────────────┴───────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    API Routes                              │   │
│  │  auth | market | signals | trades | portfolio | ai | news │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │ ML Engine  │ │ Strategies │ │    Risk    │ │   Data     │   │
│  │ LSTM+XGB   │ │ Vol+Trend  │ │ Management │ │ Collectors │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────┐
│              Background Services (Celery)                        │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │   Data     │ │ Prediction │ │  Trading   │ │ Monitoring │   │
│  │ Collection │ │   Tasks    │ │   Tasks    │ │   Tasks    │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────┐
│                     Data Storage                                 │
│  ┌──────────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │ PostgreSQL/      │  │   Redis     │  │   TimescaleDB   │    │
│  │ TimescaleDB      │  │   Cache     │  │   Time-series   │    │
│  └──────────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
├── backend/
│   └── app/
│       ├── api/routes/          # FastAPI route handlers
│       ├── core/                # Security, logging, exceptions
│       ├── data_collectors/     # OANDA, news, sentiment
│       ├── database/            # SQLAlchemy models, sessions
│       ├── ml_engine/           # LSTM, XGBoost, ensemble models
│       ├── risk_management/     # Position sizing, drawdown control
│       ├── services/            # Business logic services
│       ├── strategies/          # Trading strategies
│       ├── tasks/               # Celery background tasks
│       ├── technical_analysis/  # 50+ technical indicators
│       └── websocket/           # Real-time communication
│
├── frontend/
│   └── lib/
│       ├── core/                # Theme, DI, routing, widgets
│       └── features/            # Auth, market, signals, trades, etc.
│
└── docker/                      # Docker configuration
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Flutter 3.0+
- Docker & Docker Compose
- PostgreSQL with TimescaleDB
- Redis
- OANDA API account (for market data)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Get dependencies
flutter pub get

# Run code generation
flutter pub run build_runner build

# Run in development
flutter run -d chrome

# Build for production
flutter build web --release
```

### Docker Setup (Recommended)

```bash
cd docker

# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📊 Features

### Machine Learning Engine

- **LSTM Neural Network**: Deep learning for sequence prediction
- **XGBoost Model**: Gradient boosting for tabular features
- **Ensemble Model**: Weighted voting with confidence scoring
- **100+ Feature Engineering**: Technical, fundamental, sentiment features
- **Walk-Forward Validation**: Time-series aware model evaluation
- **Self-Improvement System**: Automatic retraining on performance degradation

### Trading Strategies

- **Volatility Strategy**: Optimized for high-volatility conditions
- **Trend Following**: Captures major market trends
- **Swing Trading**: Medium-term position management
- **ML Strategy**: AI-driven signal generation

### Risk Management

- **Dynamic Position Sizing**: Kelly criterion with risk adjustments
- **Drawdown Control**: Maximum drawdown limits with circuit breakers
- **Correlation Analysis**: Multi-position risk assessment
- **VaR Calculation**: Value at Risk monitoring

### Technical Analysis

50+ indicators including:
- Trend: SMA, EMA, MACD, ADX, Parabolic SAR, Ichimoku
- Momentum: RSI, Stochastic, CCI, Williams %R, ROC
- Volatility: Bollinger Bands, ATR, Keltner Channel
- Volume: OBV, MFI, VWAP, A/D Line

## 🔧 Configuration

### Environment Variables

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/trading
REDIS_URL=redis://localhost:6379

# OANDA API
OANDA_API_KEY=your_api_key
OANDA_ACCOUNT_ID=your_account_id
OANDA_ENVIRONMENT=practice  # or live

# Security
JWT_SECRET_KEY=your_secret_key
JWT_ALGORITHM=HS256

# ML Settings
MODEL_RETRAIN_THRESHOLD=0.75
PREDICTION_CONFIDENCE_MIN=70.0
```

## 📱 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login
- `POST /api/v1/auth/refresh` - Refresh token

### Market Data
- `GET /api/v1/market/price/{symbol}` - Current price
- `GET /api/v1/market/candles/{symbol}` - OHLCV data
- `GET /api/v1/market/indicators/{symbol}` - Technical indicators

### Trading Signals
- `GET /api/v1/signals/active` - Active signals
- `GET /api/v1/signals/history` - Signal history
- `POST /api/v1/signals/{id}/execute` - Execute signal

### Trades
- `GET /api/v1/trades/open` - Open trades
- `POST /api/v1/trades/order` - Place order
- `PUT /api/v1/trades/{id}` - Modify trade
- `POST /api/v1/trades/{id}/close` - Close trade

### Portfolio
- `GET /api/v1/portfolio` - Portfolio overview
- `GET /api/v1/portfolio/performance` - Performance metrics
- `GET /api/v1/portfolio/equity-curve` - Equity curve data

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum peak-to-trough decline
- **Expectancy**: Average expected profit per trade
- **Model Accuracy**: ML prediction accuracy

## 🔒 Security

- JWT-based authentication
- Password hashing with bcrypt
- API rate limiting
- Input validation
- CORS configuration
- Secure WebSocket connections

## 📝 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading financial instruments carries significant risk of loss. Past performance is not indicative of future results. Always trade with caution and never invest more than you can afford to lose.
