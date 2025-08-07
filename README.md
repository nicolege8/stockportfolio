# Stock Portfolio Sentiment Analysis & Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## Project Overview

### Problem Statement
**Stock Portfolio Sentiment Analysis & Prediction System** is a comprehensive machine learning pipeline that transforms financial news sentiment into actionable trading signals. This project leverages advanced natural language processing and time-series analysis to predict stock price movements based on market sentiment, providing investors with data-driven insights for portfolio optimization.

### Key Objectives
- **Real-time Sentiment Analysis**: Process financial news from multiple sources to extract sentiment signals
- **Advanced Feature Engineering**: Create sophisticated technical indicators from sentiment data
- **Multi-Model Prediction**: Employ machine learning models for robust price direction forecasting
- **Trading Signal Generation**: Generate BUY/SELL/HOLD signals with confidence scores
- **Portfolio Optimization**: Enable sentiment-weighted portfolio allocation strategies

### Expected Outcomes
- **Improved Trading Decisions**: Data-driven approach to stock selection and timing
- **Risk Management**: Sentiment-based risk assessment and position sizing
- **Market Timing**: Identify optimal entry and exit points based on sentiment trends
- **Sector Analysis**: Compare sentiment across related stocks and sectors

## Table of Contents

- [Technologies & Prerequisites](#-technologies--prerequisites)
- [Key Features](#-key-features)
- [Installation Guide](#-installation-guide)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Code Examples](#-code-examples)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Advanced Features](#-advanced-features)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

## Technologies & Prerequisites

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **Requests**: API integration
- **JSON**: Data serialization

### External APIs & Services
- **Alpha Vantage API**: Financial news sentiment data
- **Polygon.io**: Stock data

### Prerequisites
- Python 3.8 or higher
- pip package manager
- API keys for Alpha Vantage and Polygon.io
- 4GB+ RAM for processing large datasets
- Stable internet connection for API calls

## Key Features

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

## Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/stockportfolio.git
cd stockportfolio
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn requests python-dotenv

# Or install from requirements.txt (if available)
pip install -r requirements.txt
```

### Step 4: Set Up API Keys
Create a `.env` file in the project root:
```bash
# Alpha Vantage API
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Polygon.io API (optional)
POLYGON_API_KEY=your_polygon_key_here
```

### Step 5: Verify Installation
```bash
python -c "import pandas, numpy, sklearn; print('Installation successful!')"
```

## Quick Start

### Basic Usage Example
```python
from sentiment_preprocessor import SentimentPreprocessor
from filter_sentiment_variables import filter_sentiment_variables

# Initialize preprocessor
preprocessor = SentimentPreprocessor()

# Process sentiment data for all stocks
processed_data = preprocessor.process_all_stocks()

# Filter to most relevant variables
filter_sentiment_variables()

print("Sentiment analysis pipeline completed!")
```

### Run Complete Demo
```bash
# Process sentiment data
python sentiment_preprocessor.py

# Filter variables
python filter_sentiment_variables.py

# Update sentiment CSVs
python update_sentiment_csvs.py
```

## 📚 API Documentation

### SentimentPreprocessor Class

The main preprocessing class for sentiment data analysis.

#### Constructor
```python
SentimentPreprocessor(sentiment_dir: str = "news_jsons")
```

**Parameters:**
- `sentiment_dir` (str): Directory containing sentiment JSON files

#### Key Methods

##### `load_sentiment_data() -> Dict[str, pd.DataFrame]`
Loads all sentiment JSON files and converts to DataFrames.

**Returns:**
- Dictionary mapping stock symbols to sentiment DataFrames

**Example:**
```python
preprocessor = SentimentPreprocessor()
sentiment_data = preprocessor.load_sentiment_data()
print(f"Loaded data for: {list(sentiment_data.keys())}")
```

##### `process_all_stocks() -> Dict[str, pd.DataFrame]`
Processes sentiment data for all stocks with feature engineering.

**Returns:**
- Dictionary of processed DataFrames with engineered features

**Example:**
```python
processed_data = preprocessor.process_all_stocks()
msft_data = processed_data['MSFT']
print(f"MSFT features: {list(msft_data.columns)}")
```

##### `save_processed_data(output_dir: str = "processed_news")`
Saves processed data to CSV files.

**Parameters:**
- `output_dir` (str): Directory to save processed CSV files

### Alpha Vantage API Integration

#### News Sentiment Endpoint
```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_news_sentiment(ticker: str, topics: str = "technology", 
                      time_from: str = "20240101T0000", 
                      time_to: str = "20241231T2359", 
                      limit: int = 1000):
    """
    Fetch news sentiment data from Alpha Vantage API.
    
    Args:
        ticker (str): Stock symbol (e.g., 'MSFT')
        topics (str): News topics filter
        time_from (str): Start time in YYYYMMDDTHHMM format
        time_to (str): End time in YYYYMMDDTHHMM format
        limit (int): Maximum number of articles to fetch
    
    Returns:
        dict: JSON response from Alpha Vantage API
    """
    API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    base_url = "https://www.alphavantage.co/query"
    
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "topics": topics,
        "time_from": time_from,
        "time_to": time_to,
        "limit": limit,
        "apikey": API_KEY,
    }
    
    response = requests.get(base_url, params=params)
    return response.json()
```

## 💻 Code Examples

### Example 1: Basic Sentiment Analysis
```python
from sentiment_preprocessor import SentimentPreprocessor
import pandas as pd

# Initialize preprocessor
preprocessor = SentimentPreprocessor()

# Load and process MSFT sentiment data
msft_data = preprocessor.process_all_stocks()['MSFT']

# Display key metrics
print("MSFT Sentiment Analysis:")
print(f"Date range: {msft_data['date'].min()} to {msft_data['date'].max()}")
print(f"Average sentiment: {msft_data['sentiment_mean'].mean():.3f}")
print(f"Sentiment volatility: {msft_data['sentiment_std'].mean():.3f}")
print(f"Total articles: {msft_data['article_count'].sum()}")
```

### Example 2: Feature Engineering
```python
# Create technical indicators
def create_sentiment_indicators(df):
    """Create technical indicators from sentiment data."""
    
    # RSI for sentiment
    df['sentiment_rsi'] = calculate_rsi(df['sentiment_mean'], period=14)
    
    # MACD for sentiment
    df['sentiment_macd'], df['sentiment_macd_signal'] = calculate_macd(df['sentiment_mean'])
    
    # Moving averages
    df['sentiment_ma_7d'] = df['sentiment_mean'].rolling(7).mean()
    df['sentiment_ma_14d'] = df['sentiment_mean'].rolling(14).mean()
    
    # Momentum indicators
    df['sentiment_momentum'] = df['sentiment_mean'].diff(5)
    df['sentiment_acceleration'] = df['sentiment_momentum'].diff(5)
    
    return df

# Apply to your data
msft_data = create_sentiment_indicators(msft_data)
```

### Example 3: Trading Signal Generation
```python
def generate_trading_signals(df, confidence_threshold=0.7):
    """
    Generate trading signals based on sentiment analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment features
        confidence_threshold (float): Minimum confidence for signal generation
    
    Returns:
        pd.DataFrame: DataFrame with trading signals
    """
    signals = df.copy()
    
    # Calculate sentiment strength
    signals['sentiment_strength'] = abs(signals['sentiment_mean'])
    
    # Generate signals based on sentiment direction and strength
    signals['signal'] = 'HOLD'
    signals.loc[(signals['sentiment_mean'] > 0) & 
                (signals['sentiment_strength'] > confidence_threshold), 'signal'] = 'BUY'
    signals.loc[(signals['sentiment_mean'] < 0) & 
                (signals['sentiment_strength'] > confidence_threshold), 'signal'] = 'SELL'
    
    return signals

# Generate signals for MSFT
msft_signals = generate_trading_signals(msft_data)
print(f"Signal distribution: {msft_signals['signal'].value_counts()}")
```

### Example 4: Multi-Stock Analysis
```python
def compare_stock_sentiments(stocks=['MSFT', 'V', 'VZ', 'WBD', 'SNY']):
    """Compare sentiment across multiple stocks."""
    
    preprocessor = SentimentPreprocessor()
    processed_data = preprocessor.process_all_stocks()
    
    comparison = {}
    for stock in stocks:
        if stock in processed_data:
            data = processed_data[stock]
            comparison[stock] = {
                'avg_sentiment': data['sentiment_mean'].mean(),
                'sentiment_volatility': data['sentiment_std'].mean(),
                'article_count': data['article_count'].sum(),
                'positive_days': (data['sentiment_mean'] > 0).sum(),
                'negative_days': (data['sentiment_mean'] < 0).sum()
            }
    
    return pd.DataFrame(comparison).T

# Compare all stocks
comparison_df = compare_stock_sentiments()
print(comparison_df)
```

## 📁 Project Structure

```
stockportfolio/
├── api_call_code/                    # API integration scripts
│   ├── alphaverse.py                    # Alpha Vantage API calls
│   ├── alphaverse2.py                   # Alternative API implementation
│   ├── polygon_news.py                  # Polygon.io news API
│   └── polygon_article_info/            # Polygon news data
├── earnings_csv/                     # Earnings call transcripts
│   ├── MSFT/                           # Microsoft earnings data
│   ├── SNY/                            # Sanofi earnings data
│   ├── V/                              # Visa earnings data
│   ├── VZ/                             # Verizon earnings data
│   └── WBD/                            # Warner Bros Discovery earnings data
├── earnings_jsons/                   # Processed earnings data
├── news_jsons/                       # Raw news sentiment data
├── processed_news/                   # Processed sentiment features
├── Stock_Data/                       # Stock price data with price differences
├── sentiment_preprocessor.py         # Main sentiment processing pipeline
├── filter_sentiment_variables.py     # Variable filtering utility
├── update_sentiment_csvs.py          # CSV update utility
├── SENTIMENT_VARIABLE_FILTERING_SUMMARY.md  # Filtering documentation
├── README.md                         # This file
└── stocks_predictor (1).ipynb        # Jupyter notebook for analysis
```

## Usage Examples

### Real-World Scenario: Portfolio Sentiment Analysis

```python
# Scenario: Analyze sentiment for a tech-focused portfolio
portfolio_stocks = ['MSFT', 'V', 'VZ']

# 1. Load and process sentiment data
preprocessor = SentimentPreprocessor()
processed_data = preprocessor.process_all_stocks()

# 2. Calculate portfolio sentiment
portfolio_sentiment = {}
for stock in portfolio_stocks:
    if stock in processed_data:
        data = processed_data[stock]
        portfolio_sentiment[stock] = {
            'current_sentiment': data['sentiment_mean'].iloc[-1],
            'sentiment_trend': data['sentiment_trend_7d'].iloc[-1],
            'sentiment_strength': abs(data['sentiment_mean'].iloc[-1]),
            'risk_level': data['sentiment_volatility'].iloc[-1]
        }

# 3. Generate portfolio recommendations
for stock, metrics in portfolio_sentiment.items():
    if metrics['sentiment_strength'] > 0.6:
        if metrics['current_sentiment'] > 0:
            print(f"{stock}: STRONG BUY - High positive sentiment")
        else:
            print(f"{stock}: STRONG SELL - High negative sentiment")
    elif metrics['sentiment_strength'] > 0.3:
        if metrics['current_sentiment'] > 0:
            print(f"{stock}: BUY - Moderate positive sentiment")
        else:
            print(f"{stock}: SELL - Moderate negative sentiment")
    else:
        print(f"{stock}: HOLD - Low sentiment signal")
```

### Advanced Feature: Sector Rotation Analysis

```python
def sector_sentiment_analysis():
    """Analyze sentiment across different sectors for rotation opportunities."""
    
    # Define sector mappings
    sectors = {
        'Technology': ['MSFT'],
        'Financial': ['V'],
        'Telecommunications': ['VZ'],
        'Healthcare': ['SNY'],
        'Media': ['WBD']
    }
    
    preprocessor = SentimentPreprocessor()
    processed_data = preprocessor.process_all_stocks()
    
    sector_analysis = {}
    for sector, stocks in sectors.items():
        sector_sentiment = []
        for stock in stocks:
            if stock in processed_data:
                data = processed_data[stock]
                sector_sentiment.append(data['sentiment_mean'].iloc[-1])
        
        if sector_sentiment:
            sector_analysis[sector] = {
                'avg_sentiment': np.mean(sector_sentiment),
                'sentiment_rank': len([s for s in sector_analysis.values() 
                                     if s['avg_sentiment'] > np.mean(sector_sentiment)]) + 1
            }
    
    # Sort by sentiment
    sorted_sectors = sorted(sector_analysis.items(), 
                           key=lambda x: x[1]['avg_sentiment'], reverse=True)
    
    print("Sector Sentiment Ranking (Best to Worst):")
    for i, (sector, metrics) in enumerate(sorted_sectors, 1):
        print(f"{i}. {sector}: {metrics['avg_sentiment']:.3f}")
    
    return sector_analysis
```

## Advanced Features

### Custom Feature Engineering
```python
def add_custom_features(df):
    """Add custom sentiment features for enhanced prediction."""
    
    # Sentiment momentum indicators
    df['sentiment_momentum_3d'] = df['sentiment_mean'].diff(3)
    df['sentiment_momentum_7d'] = df['sentiment_mean'].diff(7)
    
    # Sentiment volatility measures
    df['sentiment_volatility_5d'] = df['sentiment_mean'].rolling(5).std()
    df['sentiment_volatility_10d'] = df['sentiment_mean'].rolling(10).std()
    
    # Sentiment extremes
    df['sentiment_extreme_positive'] = (df['sentiment_mean'] > df['sentiment_mean'].quantile(0.9)).astype(int)
    df['sentiment_extreme_negative'] = (df['sentiment_mean'] < df['sentiment_mean'].quantile(0.1)).astype(int)
    
    # Sentiment reversal signals
    df['sentiment_reversal'] = ((df['sentiment_mean'] > 0) & 
                               (df['sentiment_mean'].shift(1) < 0)).astype(int)
    
    return df
```

### Model Performance Monitoring
```python
def monitor_model_performance(predictions, actual_values, window=30):
    """Monitor model performance over time."""
    
    performance_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for i in range(window, len(predictions)):
        window_preds = predictions[i-window:i]
        window_actuals = actual_values[i-window:i]
        
        # Calculate metrics for this window
        accuracy = (window_preds == window_actuals).mean()
        # Add other metrics as needed
        
        performance_metrics['accuracy'].append(accuracy)
    
    return performance_metrics
```

## Contributing

We welcome contributions from the community! Here's how you can help improve this project:

### Contribution Guidelines

#### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/stockportfolio.git
cd stockportfolio
```

#### 2. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

#### 3. Make Changes
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include type hints where appropriate
- Write tests for new functionality

#### 4. Test Your Changes
```bash
# Run existing tests
python -m pytest tests/

# Test your new functionality
python your_new_script.py
```

#### 5. Commit and Push
```bash
git add .
git commit -m "Add: brief description of your changes"
git push origin feature/your-feature-name
```

#### 6. Create Pull Request
- Provide clear description of changes
- Include any relevant issue numbers
- Add screenshots for UI changes

### Code Style Guidelines

#### Python Style
- Follow PEP 8 conventions
- Use meaningful variable names
- Keep functions under 50 lines
- Add comprehensive docstrings

#### Documentation
- Update README.md for new features
- Add inline comments for complex logic
- Include usage examples

#### Testing
- Write unit tests for new functions
- Test edge cases and error conditions
- Maintain test coverage above 80%

### Reporting Issues

When reporting issues, please include:
1. **Environment details**: Python version, OS, dependencies
2. **Error messages**: Full traceback and error logs
3. **Steps to reproduce**: Clear, step-by-step instructions
4. **Expected vs actual behavior**: What you expected vs what happened
5. **Screenshots**: If applicable

### Feature Requests

For feature requests:
1. Check existing issues to avoid duplicates
2. Provide clear use case and benefits
3. Include mockups or examples if applicable
4. Consider implementation complexity

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: API Rate Limits
**Problem**: Getting 429 (Too Many Requests) errors
**Solution**:
```python
import time

def api_call_with_retry(url, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            if response.status_code == 429:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
```

#### Issue 2: Memory Issues with Large Datasets
**Problem**: Out of memory errors when processing large files
**Solution**:
```python
# Process data in chunks
def process_large_file(file_path, chunk_size=1000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process each chunk
        processed_chunk = process_chunk(chunk)
        yield processed_chunk
```

#### Issue 3: Missing Dependencies
**Problem**: Import errors for required packages
**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually
pip install pandas numpy scikit-learn matplotlib seaborn requests python-dotenv
```

#### Issue 4: API Key Issues
**Problem**: Authentication errors with APIs
**Solution**:
```python
# Check API key configuration
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

if not api_key:
    raise ValueError("API key not found. Please check your .env file.")
```

### Performance Optimization

#### Optimize Data Processing
```python
# Use vectorized operations instead of loops
# Instead of:
for i in range(len(df)):
    df.loc[i, 'new_col'] = some_calculation(df.loc[i, 'old_col'])

# Use:
df['new_col'] = df['old_col'].apply(some_calculation)
```

#### Memory Management
```python
# Clear memory when processing large datasets
import gc

def process_with_memory_management(data):
    # Process data
    result = heavy_processing(data)
    
    # Clear memory
    del data
    gc.collect()
    
    return result
```

## Acknowledgments

### Data Sources
- **[Alpha Vantage](https://www.alphavantage.co/)**: Financial news sentiment data and market information
- **[Polygon.io](https://polygon.io/)**: Real-time market data and news articles
- **[Yahoo Finance](https://finance.yahoo.com/)**: Historical stock price data via yfinance library

### Open Source Libraries
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[NumPy](https://numpy.org/)**: Numerical computing
- **[Scikit-learn](https://scikit-learn.org/)**: Machine learning algorithms
- **[Matplotlib](https://matplotlib.org/)**: Data visualization
- **[Seaborn](https://seaborn.pydata.org/)**: Statistical data visualization
- **[Requests](https://requests.readthedocs.io/)**: HTTP library for API calls

### Research and Inspiration
- Financial sentiment analysis research papers
- Technical analysis literature
- Machine learning for quantitative finance resources

### Community Contributors
- Thanks to all contributors who have helped improve this project
- Special thanks to the open-source community for providing excellent tools and libraries

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### License Terms
- **Commercial Use**: Allowed
- **Modification**: Allowed
- **Distribution**: Allowed
- **Private Use**: Allowed
- **Liability**: Limited
- **Warranty**: None

### Disclaimer
This software is for educational and research purposes only. Trading involves substantial risk and is not suitable for all investors. Always do your own research and consider consulting with a financial advisor before making investment decisions.

---

**Important Notice**: This tool is designed for educational purposes. Past performance does not guarantee future results. Always use proper risk management and never invest more than you can afford to lose.

**Support**: For questions, issues, or contributions, please open an issue on GitHub or contact the maintainers.

**Star this repository** if you find it helpful!


