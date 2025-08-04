import pandas as pd
import os

def filter_sentiment_variables():
    """
    Filter sentiment CSV files to keep only the most relevant variables for stock prediction.
    """
    
    # Define the most relevant variables for stock prediction
    relevant_variables = [
        # Core sentiment metrics
        'date',
        'sentiment_mean',
        'sentiment_std', 
        'sentiment_polarity',
        'weighted_sentiment',
        
        # Technical indicators
        'sentiment_momentum',
        'sentiment_volatility',
        'sentiment_rsi',
        'sentiment_macd',
        'sentiment_rate_of_change',
        'sentiment_acceleration',
        
        # Moving averages (most predictive)
        'sentiment_ma_7d',
        'sentiment_ma_14d',
        'sentiment_trend_7d',
        'sentiment_trend_14d',
        
        # Volume & activity metrics
        'article_count',
        'article_count_ma_7d',
        'relevance_mean',
        'source_count',
        
        # Sector-specific sentiment (most relevant sectors)
        'Technology_sentiment_score',
        'Finance_sentiment_score', 
        'Financial Markets_sentiment_score',
        'Earnings_sentiment_score',
        'Economy - Macro_sentiment_score',
        'Manufacturing_sentiment_score'
    ]
    
    # Process each CSV file
    csv_files = [
        'processed_sentiment/msft_sentiment_features.csv',
        'processed_sentiment/wbd_sentiment_features.csv',
        'processed_sentiment/v_sentiment_features.csv',
        'processed_sentiment/vz_sentiment_features.csv',
        'processed_sentiment/sny_sentiment_features.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"Processing {csv_file}...")
            
            # Read the original CSV
            df = pd.read_csv(csv_file)
            
            # Get available columns (some might not exist in all files)
            available_columns = [col for col in relevant_variables if col in df.columns]
            
            # Filter to keep only relevant columns
            df_filtered = df[available_columns]
            
            # Create backup of original file
            backup_file = csv_file.replace('.csv', '_full.csv')
            df.to_csv(backup_file, index=False)
            print(f"  Backup created: {backup_file}")
            
            # Save filtered file
            df_filtered.to_csv(csv_file, index=False)
            print(f"  Filtered file saved: {csv_file}")
            print(f"  Kept {len(available_columns)} out of {len(df.columns)} variables")
            print(f"  Removed {len(df.columns) - len(available_columns)} variables")
            
            # Show summary of kept variables
            print(f"  Kept variables: {', '.join(available_columns)}")
            print()
        else:
            print(f"File not found: {csv_file}")
    
    print("Filtering complete! Original files backed up with '_full' suffix.")

if __name__ == "__main__":
    filter_sentiment_variables() 