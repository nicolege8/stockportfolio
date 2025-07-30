import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SentimentPreprocessor:
    """
    Comprehensive sentiment preprocessing pipeline for stock price prediction.
    
    This class handles:
    - Loading and parsing sentiment JSON files
    - Time-based aggregation of sentiment scores
    - Feature engineering for machine learning
    - Data alignment with stock price timeframes
    """
    
    def __init__(self, sentiment_dir: str = "news_sentiment"):
        """
        Initialize the preprocessor.
        
        Args:
            sentiment_dir: Directory containing sentiment JSON files
        """
        self.sentiment_dir = sentiment_dir
        self.stocks = ['MSFT', 'V', 'VZ', 'WBD', 'SNY']
        self.sentiment_data = {}
        self.processed_data = {}
        
    def load_sentiment_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all sentiment JSON files and convert to DataFrames.
        
        Returns:
            Dictionary mapping stock symbols to sentiment DataFrames
        """
        print("Loading sentiment data...")
        
        for stock in self.stocks:
            file_path = os.path.join(self.sentiment_dir, f"{stock.lower()}_sentiment.json")
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract articles from feed
                articles = data.get('feed', [])
                
                # Convert to DataFrame
                df = pd.DataFrame(articles)
                
                # Parse timestamp
                df['time_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S')
                df['date'] = df['time_published'].dt.date
                
                # Extract ticker-specific sentiment
                ticker_sentiments = []
                for _, row in df.iterrows():
                    ticker_data = row.get('ticker_sentiment', [])
                    for ticker_info in ticker_data:
                        if ticker_info.get('ticker') == stock:
                            ticker_sentiments.append({
                                'time_published': row['time_published'],
                                'date': row['date'],
                                'title': row.get('title', ''),
                                'summary': row.get('summary', ''),
                                'source': row.get('source', ''),
                                'overall_sentiment_score': row.get('overall_sentiment_score', 0),
                                'overall_sentiment_label': row.get('overall_sentiment_label', ''),
                                'ticker_sentiment_score': float(ticker_info.get('ticker_sentiment_score', 0)),
                                'ticker_sentiment_label': ticker_info.get('ticker_sentiment_label', ''),
                                'relevance_score': float(ticker_info.get('relevance_score', 0)),
                                'topics': row.get('topics', [])
                            })
                            break
                    else:
                        # If no specific ticker sentiment found, use overall sentiment
                        ticker_sentiments.append({
                            'time_published': row['time_published'],
                            'date': row['date'],
                            'title': row.get('title', ''),
                            'summary': row.get('summary', ''),
                            'source': row.get('source', ''),
                            'overall_sentiment_score': row.get('overall_sentiment_score', 0),
                            'overall_sentiment_label': row.get('overall_sentiment_label', ''),
                            'ticker_sentiment_score': row.get('overall_sentiment_score', 0),
                            'ticker_sentiment_label': row.get('overall_sentiment_label', ''),
                            'relevance_score': 0.5,  # Default relevance
                            'topics': row.get('topics', [])
                        })
                
                stock_df = pd.DataFrame(ticker_sentiments)
                stock_df = stock_df.sort_values('time_published')
                
                self.sentiment_data[stock] = stock_df
                print(f"Loaded {len(stock_df)} articles for {stock}")
            else:
                print(f"Warning: No sentiment file found for {stock}")
        
        return self.sentiment_data
    
    def create_daily_features(self, stock: str) -> pd.DataFrame:
        """
        Create daily aggregated sentiment features for a specific stock.
        
        Args:
            stock: Stock symbol
            
        Returns:
            DataFrame with daily sentiment features
        """
        if stock not in self.sentiment_data:
            print(f"No data found for {stock}")
            return pd.DataFrame()
        
        df = self.sentiment_data[stock].copy()
        
        # Group by date and aggregate
        daily_features = df.groupby('date').agg({
            'ticker_sentiment_score': ['mean', 'std', 'count', 'min', 'max'],
            'relevance_score': ['mean', 'sum'],
            'overall_sentiment_score': ['mean', 'std'],
            'title': 'count',  # Article count
            'source': 'nunique'  # Unique sources
        }).reset_index()
        
        # Flatten MultiIndex columns
        daily_features.columns = [
            'date', 'sentiment_mean', 'sentiment_std', 'sentiment_count', 
            'sentiment_min', 'sentiment_max', 'relevance_mean', 'relevance_sum',
            'overall_sentiment_mean', 'overall_sentiment_std', 'source_count', 'article_count'
        ]
        
        # Add sentiment polarity features
        daily_features['sentiment_polarity'] = np.where(
            daily_features['sentiment_mean'] > 0.15, 'bullish',
            np.where(daily_features['sentiment_mean'] < -0.15, 'bearish', 'neutral')
        )
        
        # Add sentiment momentum (change from previous day)
        daily_features['sentiment_momentum'] = daily_features['sentiment_mean'].diff()
        
        # Add volatility (rolling standard deviation)
        daily_features['sentiment_volatility'] = daily_features['sentiment_mean'].rolling(7).std()
        
        # Add weighted sentiment (by relevance)
        daily_features['weighted_sentiment'] = (
            daily_features['sentiment_mean'] * daily_features['relevance_mean']
        )
        
        return daily_features
    
    def create_rolling_features(self, daily_df: pd.DataFrame, windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling window features for sentiment analysis.
        
        Args:
            daily_df: Daily sentiment DataFrame
            windows: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with rolling features added
        """
        df = daily_df.copy()
        
        for window in windows:
            # Rolling sentiment averages
            df[f'sentiment_ma_{window}d'] = df['sentiment_mean'].rolling(window).mean()
            df[f'sentiment_std_{window}d'] = df['sentiment_mean'].rolling(window).std()
            
            # Rolling article counts
            df[f'article_count_ma_{window}d'] = df['article_count'].rolling(window).mean()
            
            # Rolling relevance
            df[f'relevance_ma_{window}d'] = df['relevance_mean'].rolling(window).mean()
            
            # Sentiment trend (slope of linear regression over window)
            df[f'sentiment_trend_{window}d'] = self._calculate_trend(df['sentiment_mean'], window)
        
        return df
    
    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend (slope) over rolling window using linear regression."""
        def slope(x):
            if len(x) < 2:
                return np.nan
            return np.polyfit(range(len(x)), x, 1)[0]
        
        return series.rolling(window).apply(slope)
    
    def create_sentiment_indicators(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical-style sentiment indicators.
        
        Args:
            daily_df: Daily sentiment DataFrame
            
        Returns:
            DataFrame with sentiment indicators added
        """
        df = daily_df.copy()
        
        # Sentiment RSI (Relative Strength Index)
        df['sentiment_rsi'] = self._calculate_rsi(df['sentiment_mean'], period=14)
        
        # Sentiment MACD
        df['sentiment_macd'], df['sentiment_macd_signal'] = self._calculate_macd(df['sentiment_mean'])
        
        # Sentiment Bollinger Bands
        df['sentiment_bb_upper'], df['sentiment_bb_lower'] = self._calculate_bollinger_bands(df['sentiment_mean'])
        
        # Sentiment momentum indicators
        df['sentiment_rate_of_change'] = df['sentiment_mean'].pct_change()
        df['sentiment_acceleration'] = df['sentiment_rate_of_change'].diff()
        
        return df
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI for sentiment series."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD for sentiment series."""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands for sentiment series."""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def create_topic_features(self, stock: str) -> pd.DataFrame:
        """
        Extract topic-based sentiment features.
        
        Args:
            stock: Stock symbol
            
        Returns:
            DataFrame with topic features
        """
        if stock not in self.sentiment_data:
            return pd.DataFrame()
        
        df = self.sentiment_data[stock].copy()
        
        # Extract topics and their relevance scores
        topic_sentiments = []
        
        for _, row in df.iterrows():
            topics = row.get('topics', [])
            sentiment_score = row['ticker_sentiment_score']
            
            for topic in topics:
                topic_name = topic.get('topic', '')
                relevance = float(topic.get('relevance_score', 0))
                
                topic_sentiments.append({
                    'date': row['date'],
                    'topic': topic_name,
                    'sentiment_score': sentiment_score,
                    'relevance_score': relevance,
                    'weighted_sentiment': sentiment_score * relevance
                })
        
        topic_df = pd.DataFrame(topic_sentiments)
        
        if topic_df.empty:
            return pd.DataFrame()
        
        # Aggregate by date and topic
        topic_features = topic_df.groupby(['date', 'topic']).agg({
            'sentiment_score': 'mean',
            'relevance_score': 'mean',
            'weighted_sentiment': 'sum'
        }).reset_index()
        
        # Pivot to create topic columns
        topic_pivot = topic_features.pivot_table(
            index='date',
            columns='topic',
            values=['sentiment_score', 'relevance_score', 'weighted_sentiment'],
            fill_value=0
        )
        
        # Flatten column names
        topic_pivot.columns = [f"{col[1]}_{col[0]}" for col in topic_pivot.columns]
        topic_pivot = topic_pivot.reset_index()
        
        return topic_pivot
    
    def create_source_features(self, stock: str) -> pd.DataFrame:
        """
        Create features based on news sources.
        
        Args:
            stock: Stock symbol
            
        Returns:
            DataFrame with source-based features
        """
        if stock not in self.sentiment_data:
            return pd.DataFrame()
        
        df = self.sentiment_data[stock].copy()
        
        # Define source credibility weights (you can customize these)
        source_weights = {
            'Reuters': 1.0,
            'Bloomberg': 1.0,
            'CNBC': 0.9,
            'MarketWatch': 0.8,
            'Seeking Alpha': 0.7,
            'Motley Fool': 0.7,
            'Yahoo Finance': 0.6,
            'default': 0.5
        }
        
        # Add source weight
        df['source_weight'] = df['source'].map(lambda x: source_weights.get(x, source_weights['default']))
        df['weighted_sentiment'] = df['ticker_sentiment_score'] * df['source_weight']
        
        # Aggregate by date
        source_features = df.groupby('date').agg({
            'weighted_sentiment': ['mean', 'sum'],
            'source_weight': 'mean',
            'source': 'nunique'
        }).reset_index()
        
        source_features.columns = ['date', 'weighted_sentiment_mean', 'weighted_sentiment_sum', 'avg_source_weight', 'unique_sources']
        
        return source_features
    
    def process_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """
        Process sentiment data for all stocks and create comprehensive features.
        
        Returns:
            Dictionary mapping stock symbols to processed feature DataFrames
        """
        print("Processing sentiment data for all stocks...")
        
        # Load data if not already loaded
        if not self.sentiment_data:
            self.load_sentiment_data()
        
        for stock in self.stocks:
            if stock not in self.sentiment_data:
                continue
                
            print(f"Processing {stock}...")
            
            # Create daily features
            daily_features = self.create_daily_features(stock)
            
            if daily_features.empty:
                continue
            
            # Add rolling features
            daily_features = self.create_rolling_features(daily_features)
            
            # Add sentiment indicators
            daily_features = self.create_sentiment_indicators(daily_features)
            
            # Add topic features
            topic_features = self.create_topic_features(stock)
            if not topic_features.empty:
                daily_features = daily_features.merge(topic_features, on='date', how='left')
            
            # Add source features
            source_features = self.create_source_features(stock)
            if not source_features.empty:
                daily_features = daily_features.merge(source_features, on='date', how='left')
            
            # Fill NaN values
            daily_features = daily_features.fillna(method='ffill').fillna(0)
            
            # Convert date to datetime for easier merging with price data
            daily_features['date'] = pd.to_datetime(daily_features['date'])
            
            self.processed_data[stock] = daily_features
            
            print(f"Created {len(daily_features)} daily records with {len(daily_features.columns)} features for {stock}")
        
        return self.processed_data
    
    def save_processed_data(self, output_dir: str = "processed_sentiment"):
        """
        Save processed sentiment data to CSV files.
        
        Args:
            output_dir: Directory to save processed data
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for stock, df in self.processed_data.items():
            output_path = os.path.join(output_dir, f"{stock.lower()}_sentiment_features.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {stock} features to {output_path}")
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get a summary of all features created.
        
        Returns:
            DataFrame with feature information
        """
        feature_info = []
        
        for stock, df in self.processed_data.items():
            for col in df.columns:
                if col != 'date':
                    feature_info.append({
                        'stock': stock,
                        'feature': col,
                        'type': df[col].dtype,
                        'non_null_count': df[col].count(),
                        'mean': df[col].mean() if df[col].dtype in ['float64', 'int64'] else None,
                        'std': df[col].std() if df[col].dtype in ['float64', 'int64'] else None
                    })
        
        return pd.DataFrame(feature_info)
    
    def create_training_dataset(self, target_stock: str, prediction_horizon: int = 1) -> pd.DataFrame:
        """
        Create a dataset ready for machine learning training.
        
        Args:
            target_stock: Stock to predict
            prediction_horizon: Number of days ahead to predict (1 = next day)
            
        Returns:
            DataFrame ready for ML training
        """
        if target_stock not in self.processed_data:
            print(f"No processed data found for {target_stock}")
            return pd.DataFrame()
        
        df = self.processed_data[target_stock].copy()
        
        # Create target variable (future sentiment change)
        df['target_sentiment_change'] = df['sentiment_mean'].shift(-prediction_horizon) - df['sentiment_mean']
        df['target_sentiment_direction'] = np.where(df['target_sentiment_change'] > 0, 1, 0)
        
        # Remove rows with NaN targets (last few days)
        df = df.dropna(subset=['target_sentiment_change', 'target_sentiment_direction'])
        
        # Select feature columns (exclude date and target columns)
        feature_cols = [col for col in df.columns if col not in ['date', 'target_sentiment_change', 'target_sentiment_direction']]
        
        # Create feature matrix
        X = df[feature_cols]
        y_classification = df['target_sentiment_direction']
        y_regression = df['target_sentiment_change']
        
        # Add target columns to feature matrix for easy access
        X['target_sentiment_direction'] = y_classification
        X['target_sentiment_change'] = y_regression
        X['date'] = df['date']
        
        return X

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = SentimentPreprocessor()
    
    # Process all stocks
    processed_data = preprocessor.process_all_stocks()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    # Get feature summary
    feature_summary = preprocessor.get_feature_summary()
    print("\nFeature Summary:")
    print(feature_summary.head(10))
    
    # Create training dataset for MSFT
    training_data = preprocessor.create_training_dataset('MSFT', prediction_horizon=1)
    print(f"\nTraining dataset shape: {training_data.shape}")
    print(f"Features: {list(training_data.columns)}")
    
    # Show sample of processed data
    if 'MSFT' in processed_data:
        print("\nSample MSFT sentiment features:")
        print(processed_data['MSFT'].head()) 