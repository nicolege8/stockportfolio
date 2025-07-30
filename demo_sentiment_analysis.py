#!/usr/bin/env python3
"""
Comprehensive Sentiment Analysis Demo for Stock Price Prediction

This script demonstrates the complete pipeline:
1. Preprocessing sentiment data from JSON files
2. Feature engineering and aggregation
3. Machine learning model training
4. Model evaluation and comparison
5. Trading signal generation

Usage:
    python demo_sentiment_analysis.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import our custom modules
from sentiment_preprocessor import SentimentPreprocessor
from sentiment_ml_pipeline import SentimentMLPipeline

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")

def check_data_files():
    """Check if sentiment data files exist."""
    sentiment_dir = "news_sentiment"
    required_files = [
        "msft_sentiment.json",
        "v_sentiment.json", 
        "vz_sentiment.json",
        "wbd_sentiment.json",
        "sny_sentiment.json"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(sentiment_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: Missing sentiment data files: {missing_files}")
        print("Please ensure all sentiment JSON files are in the news_sentiment/ directory")
        return False
    
    print("✓ All sentiment data files found")
    return True

def explore_sentiment_data():
    """Explore and display basic statistics about the sentiment data."""
    print_section("Exploring Sentiment Data")
    
    preprocessor = SentimentPreprocessor()
    sentiment_data = preprocessor.load_sentiment_data()
    
    if not sentiment_data:
        print("No sentiment data loaded")
        return
    
    # Display basic statistics for each stock
    for stock, df in sentiment_data.items():
        print(f"\n{stock} Sentiment Data:")
        print(f"  Total articles: {len(df)}")
        print(f"  Date range: {df['time_published'].min()} to {df['time_published'].max()}")
        print(f"  Average sentiment score: {df['ticker_sentiment_score'].mean():.4f}")
        print(f"  Sentiment std: {df['ticker_sentiment_score'].std():.4f}")
        print(f"  Unique sources: {df['source'].nunique()}")
        
        # Sentiment distribution
        sentiment_labels = df['ticker_sentiment_label'].value_counts()
        print(f"  Sentiment distribution:")
        for label, count in sentiment_labels.items():
            print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")

def process_sentiment_features():
    """Process sentiment data and create features."""
    print_section("Processing Sentiment Features")
    
    preprocessor = SentimentPreprocessor()
    
    # Process all stocks
    processed_data = preprocessor.process_all_stocks()
    
    if not processed_data:
        print("No data processed")
        return
    
    # Display feature summary
    feature_summary = preprocessor.get_feature_summary()
    print(f"\nFeature Summary:")
    print(f"Total features created: {len(feature_summary)}")
    
    # Show feature types
    feature_types = feature_summary['type'].value_counts()
    print(f"Feature types:")
    for dtype, count in feature_types.items():
        print(f"  {dtype}: {count}")
    
    # Save processed data
    preprocessor.save_processed_data()
    
    return processed_data

def analyze_feature_importance(processed_data):
    """Analyze and visualize feature importance across stocks."""
    print_section("Feature Importance Analysis")
    
    # Create a summary of most important features across all stocks
    all_features = []
    
    for stock, df in processed_data.items():
        # Calculate correlation with sentiment mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['sentiment_mean'].abs().sort_values(ascending=False)
        
        # Get top 10 features
        top_features = correlations.head(10)
        for feature, corr in top_features.items():
            all_features.append({
                'stock': stock,
                'feature': feature,
                'correlation': corr
            })
    
    # Create summary DataFrame
    feature_df = pd.DataFrame(all_features)
    
    # Find most important features across all stocks
    feature_importance = feature_df.groupby('feature')['correlation'].mean().sort_values(ascending=False)
    
    print("Top 10 Most Important Features Across All Stocks:")
    for feature, importance in feature_importance.head(10).items():
        print(f"  {feature}: {importance:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    feature_importance.head(15).plot(kind='barh')
    plt.title('Top 15 Most Important Sentiment Features')
    plt.xlabel('Average Correlation with Sentiment')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_machine_learning_analysis():
    """Run the complete machine learning analysis."""
    print_section("Machine Learning Analysis")
    
    # Initialize ML pipeline
    pipeline = SentimentMLPipeline()
    
    # Run analysis for a subset of stocks first (for demo purposes)
    demo_stocks = ['MSFT', 'V']  # Start with 2 stocks for faster execution
    
    print(f"Running ML analysis for: {demo_stocks}")
    
    # Run complete analysis
    pipeline.run_complete_analysis(
        stocks=demo_stocks,
        target_types=['classification', 'regression']
    )
    
    return pipeline

def create_trading_strategy_demo(pipeline):
    """Demonstrate trading strategy creation."""
    print_section("Trading Strategy Demo")
    
    # Create trading signals for MSFT
    print("Creating trading signals for MSFT...")
    
    # Classification signals
    class_signals = pipeline.create_trading_signals('MSFT', 'classification', threshold=0.7)
    
    if not class_signals.empty:
        print(f"\nClassification Trading Signals (MSFT):")
        print(f"Total signals: {len(class_signals)}")
        print(f"Signal distribution:")
        print(class_signals['signal'].value_counts())
        
        # Calculate basic performance metrics
        buy_signals = class_signals[class_signals['signal'] == 'BUY']
        sell_signals = class_signals[class_signals['signal'] == 'SELL']
        
        if len(buy_signals) > 0:
            buy_accuracy = (buy_signals['actual'] == 1).mean()
            print(f"BUY signal accuracy: {buy_accuracy:.4f}")
        
        if len(sell_signals) > 0:
            sell_accuracy = (sell_signals['actual'] == 0).mean()
            print(f"SELL signal accuracy: {sell_accuracy:.4f}")
    
    # Regression signals
    reg_signals = pipeline.create_trading_signals('MSFT', 'regression')
    
    if not reg_signals.empty:
        print(f"\nRegression Trading Signals (MSFT):")
        print(f"Total signals: {len(reg_signals)}")
        print(f"Signal distribution:")
        print(reg_signals['signal'].value_counts())

def create_visualization_dashboard(processed_data):
    """Create a comprehensive visualization dashboard."""
    print_section("Creating Visualization Dashboard")
    
    # Create a multi-panel visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sentiment Analysis Dashboard', fontsize=16)
    
    # 1. Sentiment trends over time
    ax1 = axes[0, 0]
    for stock, df in processed_data.items():
        ax1.plot(df['date'], df['sentiment_mean'], label=stock, alpha=0.7)
    ax1.set_title('Sentiment Trends Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Average Sentiment Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sentiment volatility comparison
    ax2 = axes[0, 1]
    volatility_data = []
    for stock, df in processed_data.items():
        volatility_data.append({
            'Stock': stock,
            'Volatility': df['sentiment_std'].mean()
        })
    vol_df = pd.DataFrame(volatility_data)
    vol_df.plot(x='Stock', y='Volatility', kind='bar', ax=ax2)
    ax2.set_title('Average Sentiment Volatility by Stock')
    ax2.set_ylabel('Sentiment Standard Deviation')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Article volume comparison
    ax3 = axes[1, 0]
    volume_data = []
    for stock, df in processed_data.items():
        volume_data.append({
            'Stock': stock,
            'Articles': df['article_count'].sum()
        })
    vol_df = pd.DataFrame(volume_data)
    vol_df.plot(x='Stock', y='Articles', kind='bar', ax=ax3)
    ax3.set_title('Total Articles by Stock')
    ax3.set_ylabel('Number of Articles')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Sentiment distribution
    ax4 = axes[1, 1]
    all_sentiments = []
    for stock, df in processed_data.items():
        all_sentiments.extend(df['sentiment_mean'].dropna())
    
    ax4.hist(all_sentiments, bins=30, alpha=0.7, edgecolor='black')
    ax4.set_title('Overall Sentiment Distribution')
    ax4.set_xlabel('Sentiment Score')
    ax4.set_ylabel('Frequency')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('sentiment_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_report(processed_data, pipeline):
    """Generate a comprehensive analysis report."""
    print_section("Generating Analysis Report")
    
    report = []
    report.append("# Sentiment Analysis for Stock Price Prediction - Analysis Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data Summary
    report.append("## Data Summary")
    for stock, df in processed_data.items():
        report.append(f"### {stock}")
        report.append(f"- Date range: {df['date'].min()} to {df['date'].max()}")
        report.append(f"- Total days: {len(df)}")
        report.append(f"- Average sentiment: {df['sentiment_mean'].mean():.4f}")
        report.append(f"- Sentiment volatility: {df['sentiment_std'].mean():.4f}")
        report.append(f"- Total articles: {df['article_count'].sum()}")
        report.append("")
    
    # Feature Summary
    report.append("## Feature Engineering Summary")
    if processed_data:
        sample_df = list(processed_data.values())[0]
        report.append(f"- Total features created: {len(sample_df.columns)}")
        report.append(f"- Feature categories:")
        report.append("  - Daily sentiment metrics")
        report.append("  - Rolling averages (3d, 7d, 14d, 30d)")
        report.append("  - Technical indicators (RSI, MACD, Bollinger Bands)")
        report.append("  - Topic-based features")
        report.append("  - Source credibility features")
        report.append("")
    
    # Model Performance Summary
    if hasattr(pipeline, 'results') and pipeline.results:
        report.append("## Model Performance Summary")
        for stock in pipeline.results:
            report.append(f"### {stock}")
            for target_type in pipeline.results[stock]:
                results = pipeline.results[stock][target_type]
                if results:
                    best_model = max(results.keys(), key=lambda x: results[x]['metrics']['cv_mean'])
                    best_score = results[best_model]['metrics']['cv_mean']
                    report.append(f"- Best {target_type} model: {best_model} (CV Score: {best_score:.4f})")
            report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("1. **Feature Engineering**: The pipeline creates comprehensive sentiment features")
    report.append("2. **Model Selection**: Tree-based models (Random Forest, Gradient Boosting) typically perform well")
    report.append("3. **Time Series Considerations**: Use time series cross-validation for financial data")
    report.append("4. **Trading Signals**: Combine sentiment signals with technical analysis for better results")
    report.append("5. **Risk Management**: Always use proper position sizing and stop-losses")
    report.append("")
    
    # Save report
    with open('sentiment_analysis_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("✓ Analysis report saved to 'sentiment_analysis_report.md'")

def main():
    """Main demo function."""
    print_header("Sentiment Analysis for Stock Price Prediction - Demo")
    
    # Check if data files exist
    if not check_data_files():
        print("Please ensure all required data files are present before running the demo.")
        return
    
    try:
        # Step 1: Explore the data
        explore_sentiment_data()
        
        # Step 2: Process sentiment features
        processed_data = process_sentiment_features()
        
        if not processed_data:
            print("No data was processed. Exiting.")
            return
        
        # Step 3: Analyze feature importance
        analyze_feature_importance(processed_data)
        
        # Step 4: Create visualization dashboard
        create_visualization_dashboard(processed_data)
        
        # Step 5: Run machine learning analysis
        pipeline = run_machine_learning_analysis()
        
        # Step 6: Demonstrate trading strategy
        create_trading_strategy_demo(pipeline)
        
        # Step 7: Generate comprehensive report
        generate_report(processed_data, pipeline)
        
        print_header("Demo Completed Successfully!")
        print("Generated files:")
        print("- processed_sentiment/ (directory with CSV files)")
        print("- sentiment_dashboard.png")
        print("- feature_importance.png")
        print("- sentiment_analysis_report.md")
        print("- [stock]_[type]_results.png (model result plots)")
        
        print("\nNext steps:")
        print("1. Review the generated visualizations")
        print("2. Examine the model performance results")
        print("3. Integrate with actual stock price data")
        print("4. Implement real-time sentiment monitoring")
        print("5. Build a live trading system")
        
    except Exception as e:
        print(f"Error during demo execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 