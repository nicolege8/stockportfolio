# Sentiment Analysis for Stock Price Prediction

A comprehensive machine learning pipeline for analyzing news sentiment and predicting stock price movements using advanced feature engineering and multiple ML models.

## 🎯 Project Overview

This project transforms raw sentiment analysis data from news articles into actionable trading signals. It processes sentiment JSON files from Alpha Vantage API and creates sophisticated features for machine learning models to predict stock price directions.

## 📊 Features

### Data Processing
- **Multi-source sentiment aggregation**: Combines sentiment from multiple news sources
- **Time-series feature engineering**: Creates daily, weekly, and monthly sentiment metrics
- **Technical indicators**: RSI, MACD, Bollinger Bands applied to sentiment data
- **Topic-based features**: Extracts sentiment by news topics (earnings, product launches, etc.)
- **Source credibility weighting**: Weights sentiment by source reliability

### Machine Learning Models
- **Classification models**: Predict sentiment direction (bullish/bearish)
- **Regression models**: Predict sentiment change magnitude
- **Multiple algorithms**: Random Forest, Gradient Boosting, SVM, Logistic/Linear Regression
- **Time series cross-validation**: Proper validation for financial data
- **Feature importance analysis**: Identifies most predictive features

### Trading Signals
- **Confidence-based signals**: BUY/SELL/HOLD based on model confidence
- **Threshold optimization**: Adjustable confidence thresholds
- **Performance metrics**: Signal accuracy and distribution analysis

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Demo
```bash
python demo_sentiment_analysis.py
```

This will:
- Load and explore your sentiment data
- Process and engineer features
- Train multiple ML models
- Generate visualizations and reports
- Create trading signals

## 📁 Project Structure

```
stockportfolio/
├── news_sentiment/                 # Raw sentiment data
│   ├── msft_sentiment.json
│   ├── v_sentiment.json
│   ├── vz_sentiment.json
│   ├── wbd_sentiment.json
│   ├── sny_sentiment.json
│   └── polygon_article_info/       # Additional news data
├── sentiment_preprocessor.py       # Data preprocessing pipeline
├── sentiment_ml_pipeline.py        # Machine learning pipeline
├── demo_sentiment_analysis.py      # Complete demo script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Core Components

### SentimentPreprocessor
The main preprocessing class that handles:

```python
from sentiment_preprocessor import SentimentPreprocessor

# Initialize preprocessor
preprocessor = SentimentPreprocessor()

# Load and process data
processed_data = preprocessor.process_all_stocks()

# Save processed features
preprocessor.save_processed_data()
```

**Key Features Created:**
- `sentiment_mean`: Daily average sentiment score
- `sentiment_std`: Sentiment volatility
- `sentiment_ma_7d`: 7-day moving average
- `sentiment_rsi`: Sentiment RSI indicator
- `sentiment_macd`: Sentiment MACD
- `article_count`: Number of articles per day
- `weighted_sentiment`: Relevance-weighted sentiment
- Topic-specific features (earnings, technology, etc.)

### SentimentMLPipeline
The machine learning pipeline that provides:

```python
from sentiment_ml_pipeline import SentimentMLPipeline

# Initialize ML pipeline
pipeline = SentimentMLPipeline()

# Train models for classification and regression
pipeline.run_complete_analysis(
    stocks=['MSFT', 'V'],
    target_types=['classification', 'regression']
)

# Generate trading signals
signals = pipeline.create_trading_signals('MSFT', 'classification')
```

## 📈 Feature Engineering Details

### Daily Aggregation
- **Sentiment Statistics**: Mean, std, min, max, count
- **Relevance Metrics**: Average and sum of relevance scores
- **Source Diversity**: Number of unique news sources
- **Polarity Classification**: Bullish/Bearish/Neutral labels

### Rolling Features
- **Moving Averages**: 3-day, 7-day, 14-day, 30-day windows
- **Trend Analysis**: Linear regression slopes over windows
- **Volatility Measures**: Rolling standard deviations

### Technical Indicators
- **Sentiment RSI**: Relative Strength Index for sentiment
- **Sentiment MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Upper and lower sentiment bands
- **Momentum Indicators**: Rate of change and acceleration

### Topic-Based Features
- **Topic Sentiment**: Sentiment scores by news topic
- **Topic Relevance**: Relevance scores by topic
- **Weighted Topic Sentiment**: Sentiment × relevance

## 🤖 Machine Learning Models

### Classification Models
Predict whether sentiment will be bullish (1) or bearish (0):
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential boosting algorithm
- **Logistic Regression**: Linear classification model
- **SVM**: Support Vector Machine with probability estimates

### Regression Models
Predict the magnitude of sentiment change:
- **Random Forest Regressor**: Ensemble regression
- **Gradient Boosting Regressor**: Sequential regression
- **Linear Regression**: Simple linear model
- **SVR**: Support Vector Regression

### Model Evaluation
- **Time Series Cross-Validation**: 5-fold time series splits
- **Performance Metrics**: Accuracy, R², RMSE, CV scores
- **Feature Importance**: Model-specific importance rankings
- **Confusion Matrices**: Classification performance details

## 📊 Trading Signals

### Signal Generation
```python
# Classification signals (BUY/SELL/HOLD)
signals = pipeline.create_trading_signals('MSFT', 'classification', threshold=0.7)

# Regression signals (based on predicted sentiment change)
signals = pipeline.create_trading_signals('MSFT', 'regression')
```

### Signal Types
- **BUY**: High confidence bullish prediction
- **SELL**: High confidence bearish prediction  
- **HOLD**: Low confidence or neutral prediction

### Performance Metrics
- **Signal Accuracy**: Percentage of correct signals
- **Signal Distribution**: Count of BUY/SELL/HOLD signals
- **Directional Accuracy**: Separate accuracy for BUY/SELL signals

## 📈 Integration with Stock Prices

### Next Steps for Price Integration

1. **Data Collection**
   ```python
   # Add stock price data collection
   import yfinance as yf
   
   def get_stock_prices(symbol, start_date, end_date):
       stock = yf.Ticker(symbol)
       prices = stock.history(start=start_date, end=end_date)
       return prices
   ```

2. **Feature Alignment**
   ```python
   # Align sentiment features with price data
   def align_features(sentiment_df, price_df):
       # Merge on date
       combined_df = sentiment_df.merge(price_df, on='date', how='inner')
       
       # Create price-based targets
       combined_df['price_change'] = combined_df['Close'].pct_change()
       combined_df['price_direction'] = np.where(combined_df['price_change'] > 0, 1, 0)
       
       return combined_df
   ```

3. **Enhanced Models**
   ```python
   # Add price-based features
   def add_price_features(df):
       df['price_ma_20'] = df['Close'].rolling(20).mean()
       df['price_volatility'] = df['Close'].rolling(20).std()
       df['rsi'] = calculate_rsi(df['Close'])
       return df
   ```

## 🎯 Advanced Use Cases

### Multi-Stock Analysis
```python
# Compare sentiment across related stocks
def sector_sentiment_analysis(stocks):
    sector_sentiment = {}
    for stock in stocks:
        sentiment = preprocessor.process_all_stocks()[stock]
        sector_sentiment[stock] = sentiment['sentiment_mean'].mean()
    return sector_sentiment
```

### Real-Time Monitoring
```python
# Set up real-time sentiment monitoring
def real_time_sentiment_monitor(symbol):
    # Fetch latest news and sentiment
    latest_sentiment = fetch_latest_sentiment(symbol)
    
    # Make prediction using trained model
    prediction = model.predict(latest_sentiment)
    
    # Generate trading signal
    signal = generate_signal(prediction)
    
    return signal
```

### Portfolio Optimization
```python
# Use sentiment for portfolio weighting
def sentiment_weighted_portfolio(stocks, sentiment_scores):
    # Weight stocks based on sentiment
    weights = softmax(sentiment_scores)
    
    # Calculate portfolio allocation
    portfolio = dict(zip(stocks, weights))
    
    return portfolio
```

## 📊 Output Files

After running the demo, you'll get:

- **`processed_sentiment/`**: CSV files with engineered features
- **`sentiment_dashboard.png`**: Comprehensive visualization dashboard
- **`feature_importance.png`**: Feature importance analysis
- **`sentiment_analysis_report.md`**: Detailed analysis report
- **`[stock]_[type]_results.png`**: Individual model result plots

## 🔧 Customization

### Adding New Features
```python
def add_custom_features(df):
    # Add your custom features here
    df['custom_sentiment_ratio'] = df['bullish_count'] / df['total_count']
    df['sentiment_momentum'] = df['sentiment_mean'].diff(5)
    return df
```

### Modifying Models
```python
# Add custom models to the pipeline
from xgboost import XGBClassifier

custom_models = {
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
}
pipeline.models['classification'].update(custom_models)
```

### Adjusting Parameters
```python
# Modify preprocessing parameters
preprocessor = SentimentPreprocessor()
preprocessor.rolling_windows = [5, 10, 20]  # Custom windows
preprocessor.source_weights = {'Reuters': 1.2, 'Bloomberg': 1.1}  # Custom weights
```

## 🚨 Important Considerations

### Data Quality
- Ensure sentiment data covers sufficient time period
- Check for data gaps and handle missing values
- Validate sentiment scores are within expected ranges

### Model Limitations
- Sentiment analysis is not a crystal ball
- Always combine with technical and fundamental analysis
- Use proper risk management and position sizing
- Past performance doesn't guarantee future results

### Trading Risks
- Sentiment can be manipulated or delayed
- News sentiment may not immediately affect prices
- Market conditions can change rapidly
- Always use stop-losses and proper risk management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with your local regulations when using for actual trading.

## 📞 Support

For questions or issues:
1. Check the generated reports for insights
2. Review the code comments for implementation details
3. Test with different parameters and stocks
4. Consider integrating with additional data sources

---

**Disclaimer**: This tool is for educational purposes only. Trading involves substantial risk and is not suitable for all investors. Always do your own research and consider consulting with a financial advisor before making investment decisions.


