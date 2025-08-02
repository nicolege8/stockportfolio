import json
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
from collections import defaultdict

def parse_date(date_str):
    """Parse date from format like '20250604T224742'"""
    try:
        return datetime.strptime(date_str, '%Y%m%dT%H%M%S').strftime('%Y-%m-%d')
    except:
        return None

def extract_sentiment_data(json_file_path, target_ticker):
    """Extract sentiment data for a specific ticker from a JSON file"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        sentiment_data = []
        
        for article in data.get('feed', []):
            # Find the ticker sentiment for our target ticker
            ticker_sentiment = None
            for ticker_data in article.get('ticker_sentiment', []):
                if ticker_data.get('ticker') == target_ticker:
                    ticker_sentiment = ticker_data
                    break
            
            if ticker_sentiment:
                date = parse_date(article.get('time_published', ''))
                if date:
                    sentiment_data.append({
                        'date': date,
                        'ticker_sentiment_score': float(ticker_sentiment.get('ticker_sentiment_score', 0)),
                        'relevance_score': float(ticker_sentiment.get('relevance_score', 0)),
                        'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                        'source': article.get('source', ''),
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'topics': article.get('topics', [])
                    })
        
        return sentiment_data
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return []

def calculate_features(df):
    """Calculate sentiment features similar to the existing CSV structure"""
    if df.empty:
        return pd.DataFrame()
    
    # Group by date
    daily_data = df.groupby('date').agg({
        'ticker_sentiment_score': ['mean', 'std', 'count', 'min', 'max'],
        'relevance_score': ['mean', 'sum'],
        'overall_sentiment_score': ['mean', 'std'],
        'source': 'nunique',
        'title': 'count'
    }).reset_index()
    
    # Flatten column names
    daily_data.columns = ['date', 'sentiment_mean', 'sentiment_std', 'sentiment_count', 
                         'sentiment_min', 'sentiment_max', 'relevance_mean', 'relevance_sum',
                         'overall_sentiment_mean', 'overall_sentiment_std', 'source_count', 'article_count']
    
    # Calculate additional features
    daily_data['sentiment_polarity'] = daily_data['sentiment_mean'].apply(
        lambda x: 'bullish' if x >= 0.15 else 'bearish' if x <= -0.15 else 'neutral'
    )
    
    # Calculate moving averages and trends
    daily_data = daily_data.sort_values('date')
    
    # 3-day moving averages
    daily_data['sentiment_ma_3d'] = daily_data['sentiment_mean'].rolling(3, min_periods=1).mean()
    daily_data['sentiment_std_3d'] = daily_data['sentiment_std'].rolling(3, min_periods=1).mean()
    daily_data['article_count_ma_3d'] = daily_data['article_count'].rolling(3, min_periods=1).mean()
    daily_data['relevance_ma_3d'] = daily_data['relevance_mean'].rolling(3, min_periods=1).mean()
    daily_data['sentiment_trend_3d'] = daily_data['sentiment_mean'].diff(3)
    
    # 7-day moving averages
    daily_data['sentiment_ma_7d'] = daily_data['sentiment_mean'].rolling(7, min_periods=1).mean()
    daily_data['sentiment_std_7d'] = daily_data['sentiment_std'].rolling(7, min_periods=1).mean()
    daily_data['article_count_ma_7d'] = daily_data['article_count'].rolling(7, min_periods=1).mean()
    daily_data['relevance_ma_7d'] = daily_data['relevance_mean'].rolling(7, min_periods=1).mean()
    daily_data['sentiment_trend_7d'] = daily_data['sentiment_mean'].diff(7)
    
    # 14-day moving averages
    daily_data['sentiment_ma_14d'] = daily_data['sentiment_mean'].rolling(14, min_periods=1).mean()
    daily_data['sentiment_std_14d'] = daily_data['sentiment_std'].rolling(14, min_periods=1).mean()
    daily_data['article_count_ma_14d'] = daily_data['article_count'].rolling(14, min_periods=1).mean()
    daily_data['relevance_ma_14d'] = daily_data['relevance_mean'].rolling(14, min_periods=1).mean()
    daily_data['sentiment_trend_14d'] = daily_data['sentiment_mean'].diff(14)
    
    # 30-day moving averages
    daily_data['sentiment_ma_30d'] = daily_data['sentiment_mean'].rolling(30, min_periods=1).mean()
    daily_data['sentiment_std_30d'] = daily_data['sentiment_std'].rolling(30, min_periods=1).mean()
    daily_data['article_count_ma_30d'] = daily_data['article_count'].rolling(30, min_periods=1).mean()
    daily_data['relevance_ma_30d'] = daily_data['relevance_mean'].rolling(30, min_periods=1).mean()
    daily_data['sentiment_trend_30d'] = daily_data['sentiment_mean'].diff(30)
    
    # Calculate momentum and volatility
    daily_data['sentiment_momentum'] = daily_data['sentiment_mean'].diff()
    daily_data['sentiment_volatility'] = daily_data['sentiment_std'].rolling(5, min_periods=1).mean()
    
    # Weighted sentiment
    daily_data['weighted_sentiment'] = daily_data['sentiment_mean'] * daily_data['relevance_mean']
    
    # Technical indicators (simplified)
    daily_data['sentiment_rsi'] = 50  # Placeholder
    daily_data['sentiment_macd'] = daily_data['sentiment_mean'].diff(12) - daily_data['sentiment_mean'].diff(26)
    daily_data['sentiment_macd_signal'] = daily_data['sentiment_macd'].rolling(9, min_periods=1).mean()
    
    # Bollinger Bands (using 20-day MA)
    sentiment_ma_20d = daily_data['sentiment_mean'].rolling(20, min_periods=1).mean()
    sentiment_std_20d = daily_data['sentiment_std'].rolling(20, min_periods=1).mean()
    daily_data['sentiment_bb_upper'] = sentiment_ma_20d + (2 * sentiment_std_20d)
    daily_data['sentiment_bb_lower'] = sentiment_ma_20d - (2 * sentiment_std_20d)
    
    daily_data['sentiment_rate_of_change'] = daily_data['sentiment_mean'].pct_change()
    daily_data['sentiment_acceleration'] = daily_data['sentiment_momentum'].diff()
    
    # Topic relevance scores (simplified - would need more complex logic for actual topic extraction)
    topic_columns = [
        'Blockchain_relevance_score', 'Earnings_relevance_score', 'Economy - Fiscal_relevance_score',
        'Economy - Macro_relevance_score', 'Economy - Monetary_relevance_score', 'Energy & Transportation_relevance_score',
        'Finance_relevance_score', 'Financial Markets_relevance_score', 'IPO_relevance_score', 'Life Sciences_relevance_score',
        'Manufacturing_relevance_score', 'Mergers & Acquisitions_relevance_score', 'Real Estate & Construction_relevance_score',
        'Retail & Wholesale_relevance_score', 'Technology_relevance_score'
    ]
    
    for col in topic_columns:
        daily_data[col] = 0.0  # Placeholder values
    
    # Topic sentiment scores (simplified)
    topic_sentiment_columns = [col.replace('_relevance_score', '_sentiment_score') for col in topic_columns]
    for col in topic_sentiment_columns:
        daily_data[col] = 0.0  # Placeholder values
    
    # Topic weighted sentiment scores (simplified)
    topic_weighted_columns = [col.replace('_relevance_score', '_weighted_sentiment') for col in topic_columns]
    for col in topic_weighted_columns:
        daily_data[col] = 0.0  # Placeholder values
    
    # Additional features
    daily_data['weighted_sentiment_mean'] = daily_data['weighted_sentiment']
    daily_data['weighted_sentiment_sum'] = daily_data['weighted_sentiment'] * daily_data['article_count']
    daily_data['avg_source_weight'] = daily_data['relevance_mean']
    daily_data['unique_sources'] = daily_data['source_count']
    
    # Fill NaN values
    daily_data = daily_data.fillna(0)
    
    return daily_data

def process_stock_data(stock_ticker, json_pattern, csv_file_path):
    """Process all JSON files for a stock and update the CSV file"""
    print(f"Processing {stock_ticker}...")
    
    # Find all JSON files for this stock
    json_files = glob.glob(json_pattern)
    print(f"Found {len(json_files)} JSON files for {stock_ticker}")
    
    # Extract data from all JSON files
    all_sentiment_data = []
    for json_file in json_files:
        sentiment_data = extract_sentiment_data(json_file, stock_ticker)
        all_sentiment_data.extend(sentiment_data)
    
    if not all_sentiment_data:
        print(f"No sentiment data found for {stock_ticker}")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_sentiment_data)
    print(f"Total articles found for {stock_ticker}: {len(df)}")
    
    # Calculate features
    features_df = calculate_features(df)
    
    if features_df.empty:
        print(f"No features calculated for {stock_ticker}")
        return
    
    # Read existing CSV if it exists
    existing_df = pd.DataFrame()
    if os.path.exists(csv_file_path):
        existing_df = pd.read_csv(csv_file_path)
        print(f"Existing CSV has {len(existing_df)} rows")
    
    # Combine existing and new data
    if not existing_df.empty:
        # Remove duplicates based on date
        combined_df = pd.concat([existing_df, features_df]).drop_duplicates(subset=['date'], keep='last')
        combined_df = combined_df.sort_values('date')
    else:
        combined_df = features_df
    
    # Save updated CSV
    combined_df.to_csv(csv_file_path, index=False)
    print(f"Updated {csv_file_path} with {len(combined_df)} rows")
    print(f"Added {len(features_df)} new rows for {stock_ticker}")
    print("-" * 50)

def main():
    """Main function to process all stocks"""
    print("Starting sentiment data processing...")
    print("=" * 50)
    
    # Define stock mappings
    stock_mappings = {
        'MSFT': {
            'json_pattern': 'msft_sentiment*.json',
            'csv_file': 'processed_sentiment/msft_sentiment_features.csv'
        },
        'WBD': {
            'json_pattern': 'wbd_sentiment*.json',
            'csv_file': 'processed_sentiment/wbd_sentiment_features.csv'
        },
        'V': {
            'json_pattern': 'visa_sentiment*.json',
            'csv_file': 'processed_sentiment/v_sentiment_features.csv'
        },
        'VZ': {
            'json_pattern': 'verizon_sentiment*.json',
            'csv_file': 'processed_sentiment/vz_sentiment_features.csv'
        },
        'SNY': {
            'json_pattern': 'sanofi_sentiment*.json',
            'csv_file': 'processed_sentiment/sny_sentiment_features.csv'
        }
    }
    
    # Process each stock
    for ticker, config in stock_mappings.items():
        process_stock_data(ticker, config['json_pattern'], config['csv_file'])
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 