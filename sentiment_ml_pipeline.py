import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SentimentMLPipeline:
    """
    Machine learning pipeline for stock price prediction using sentiment analysis.
    
    This class provides:
    - Multiple ML models for classification and regression
    - Feature selection and engineering
    - Time series cross-validation
    - Model evaluation and comparison
    - Visualization of results
    """
    
    def __init__(self, sentiment_data_dir: str = "processed_sentiment"):
        """
        Initialize the ML pipeline.
        
        Args:
            sentiment_data_dir: Directory containing processed sentiment data
        """
        self.sentiment_data_dir = sentiment_data_dir
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.results = {}
        
    def load_sentiment_data(self, stock: str) -> pd.DataFrame:
        """
        Load processed sentiment data for a specific stock.
        
        Args:
            stock: Stock symbol
            
        Returns:
            DataFrame with sentiment features
        """
        file_path = f"{self.sentiment_data_dir}/{stock.lower()}_sentiment_features.csv"
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except FileNotFoundError:
            print(f"No sentiment data found for {stock}")
            return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame, target_type: str = 'classification') -> tuple:
        """
        Prepare features and targets for machine learning.
        
        Args:
            df: DataFrame with sentiment features
            target_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Create target variables first
        df = df.copy()
        
        # Create target variable (future sentiment change)
        df['target_sentiment_change'] = df['sentiment_mean'].shift(-1) - df['sentiment_mean']
        df['target_sentiment_direction'] = np.where(df['target_sentiment_change'] > 0, 1, 0)
        
        # Remove rows with NaN targets (last few days)
        df = df.dropna(subset=['target_sentiment_change', 'target_sentiment_direction'])
        
        # Remove date and target columns for features
        exclude_cols = ['date', 'target_sentiment_change', 'target_sentiment_direction']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill NaN values
        X = X.fillna(0)
        
        # Select target based on type
        if target_type == 'classification':
            y = df['target_sentiment_direction']
        else:  # regression
            y = df['target_sentiment_change']
        
        return X, y, feature_cols
    
    def create_models(self):
        """
        Initialize various machine learning models.
        """
        # Classification models
        self.models['classification'] = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Regression models
        self.models['regression'] = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR()
        }
    
    def train_models(self, stock: str, target_type: str = 'classification', test_size: float = 0.2):
        """
        Train models for a specific stock.
        
        Args:
            stock: Stock symbol
            target_type: 'classification' or 'regression'
            test_size: Proportion of data for testing
        """
        # Load data
        df = self.load_sentiment_data(stock)
        if df.empty:
            return
        
        # Prepare features
        X, y, feature_names = self.prepare_features(df, target_type)
        
        if len(X) < 20:  # Need sufficient data (reduced from 50)
            print(f"Insufficient data for {stock}: {len(X)} samples")
            return
        
        # Time series split (important for financial data)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize results storage
        self.results[stock] = {
            'classification': {},
            'regression': {}
        }
        
        # Train models
        models_to_train = self.models[target_type]
        
        for model_name, model in models_to_train.items():
            print(f"Training {model_name} for {stock} ({target_type})...")
            
            # Scale features (except for tree-based models)
            if model_name in ['Logistic Regression', 'Linear Regression', 'SVM', 'SVR']:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[f"{stock}_{model_name}"] = scaler
            else:
                X_scaled = X.values
            
            # Time series cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                
                if target_type == 'classification':
                    score = accuracy_score(y_val, model.predict(X_val))
                else:
                    score = r2_score(y_val, model.predict(X_val))
                
                cv_scores.append(score)
            
            # Final train/test split
            split_idx = int(len(X_scaled) * (1 - test_size))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train final model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            if target_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'feature_importance': self._get_feature_importance(model, feature_names)
                }
            else:
                metrics = {
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'predictions': y_pred,
                    'feature_importance': self._get_feature_importance(model, feature_names)
                }
            
            self.results[stock][target_type][model_name] = {
                'model': model,
                'metrics': metrics,
                'X_test': X_test,
                'y_test': y_test,
                'feature_names': feature_names
            }
    
    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # Handle both 1D and 2D coefficient arrays
            coef = model.coef_
            if coef.ndim == 1:
                return dict(zip(feature_names, np.abs(coef)))
            else:
                return dict(zip(feature_names, np.abs(coef[0])))
        else:
            return {}
    
    def evaluate_models(self, stock: str, target_type: str = 'classification'):
        """
        Evaluate and compare models for a specific stock.
        
        Args:
            stock: Stock symbol
            target_type: 'classification' or 'regression'
        """
        if stock not in self.results or target_type not in self.results[stock]:
            print(f"No results found for {stock} ({target_type})")
            return
        
        results = self.results[stock][target_type]
        
        print(f"\n=== Model Evaluation for {stock} ({target_type}) ===")
        
        # Compare models
        comparison_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            if target_type == 'classification':
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'CV Mean': metrics['cv_mean'],
                    'CV Std': metrics['cv_std']
                })
            else:
                comparison_data.append({
                    'Model': model_name,
                    'R²': metrics['r2'],
                    'RMSE': metrics['rmse'],
                    'CV Mean': metrics['cv_mean'],
                    'CV Std': metrics['cv_std']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.sort_values('CV Mean', ascending=False))
        
        # Detailed results for best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_result = results[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        
        if target_type == 'classification':
            print(f"Accuracy: {best_result['metrics']['accuracy']:.4f}")
            print(f"Cross-validation: {best_result['metrics']['cv_mean']:.4f} (+/- {best_result['metrics']['cv_std']*2:.4f})")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(best_result['y_test'], best_result['metrics']['predictions']))
            
            # Confusion matrix
            cm = confusion_matrix(best_result['y_test'], best_result['metrics']['predictions'])
            print("\nConfusion Matrix:")
            print(cm)
        
        else:
            print(f"R² Score: {best_result['metrics']['r2']:.4f}")
            print(f"RMSE: {best_result['metrics']['rmse']:.4f}")
            print(f"Cross-validation: {best_result['metrics']['cv_mean']:.4f} (+/- {best_result['metrics']['cv_std']*2:.4f})")
        
        # Feature importance
        if best_result['metrics']['feature_importance']:
            print("\nTop 10 Most Important Features:")
            importance_df = pd.DataFrame(
                best_result['metrics']['feature_importance'].items(),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            print(importance_df.head(10))
    
    def plot_results(self, stock: str, target_type: str = 'classification'):
        """
        Create visualizations for model results.
        
        Args:
            stock: Stock symbol
            target_type: 'classification' or 'regression'
        """
        if stock not in self.results or target_type not in self.results[stock]:
            return
        
        results = self.results[stock][target_type]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Results for {stock} ({target_type})', fontsize=16)
        
        # 1. Model comparison
        comparison_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            if target_type == 'classification':
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'CV Mean': metrics['cv_mean']
                })
            else:
                comparison_data.append({
                    'Model': model_name,
                    'R²': metrics['r2'],
                    'CV Mean': metrics['cv_mean']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        ax1 = axes[0, 0]
        metric_col = 'Accuracy' if target_type == 'classification' else 'R²'
        comparison_df.plot(x='Model', y=metric_col, kind='bar', ax=ax1)
        ax1.set_title('Model Performance Comparison')
        ax1.set_ylabel(metric_col)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Cross-validation scores
        ax2 = axes[0, 1]
        cv_data = []
        for model_name, result in results.items():
            cv_data.append({
                'Model': model_name,
                'CV Mean': result['metrics']['cv_mean'],
                'CV Std': result['metrics']['cv_std']
            })
        
        cv_df = pd.DataFrame(cv_data)
        cv_df.plot(x='Model', y='CV Mean', kind='bar', yerr='CV Std', ax=ax2)
        ax2.set_title('Cross-Validation Performance')
        ax2.set_ylabel('CV Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Feature importance (best model)
        best_model_name = comparison_df.iloc[0]['Model']
        best_result = results[best_model_name]
        
        if best_result['metrics']['feature_importance']:
            ax3 = axes[1, 0]
            importance_df = pd.DataFrame(
                best_result['metrics']['feature_importance'].items(),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False).head(10)
            
            importance_df.plot(x='Feature', y='Importance', kind='barh', ax=ax3)
            ax3.set_title(f'Top 10 Features - {best_model_name}')
            ax3.set_xlabel('Importance')
        
        # 4. Predictions vs Actual
        ax4 = axes[1, 1]
        y_test = best_result['y_test']
        y_pred = best_result['metrics']['predictions']
        
        if target_type == 'classification':
            # Confusion matrix heatmap
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
            ax4.set_title('Confusion Matrix')
            ax4.set_xlabel('Predicted')
            ax4.set_ylabel('Actual')
        else:
            # Scatter plot
            ax4.scatter(y_test, y_pred, alpha=0.6)
            ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax4.set_xlabel('Actual')
            ax4.set_ylabel('Predicted')
            ax4.set_title('Predictions vs Actual')
        
        plt.tight_layout()
        plt.savefig(f'{stock}_{target_type}_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_trading_signals(self, stock: str, target_type: str = 'classification', threshold: float = 0.6):
        """
        Create trading signals based on model predictions.
        
        Args:
            stock: Stock symbol
            target_type: 'classification' or 'regression'
            threshold: Confidence threshold for classification signals
            
        Returns:
            DataFrame with trading signals
        """
        if stock not in self.results or target_type not in self.results[stock]:
            return pd.DataFrame()
        
        # Get best model
        results = self.results[stock][target_type]
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['cv_mean'])
        best_result = results[best_model_name]
        
        model = best_result['model']
        X_test = best_result['X_test']
        y_test = best_result['y_test']
        
        # Make predictions
        if target_type == 'classification':
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Create signals
            signals = pd.DataFrame({
                'actual': y_test.values,
                'predicted': predictions,
                'confidence': probabilities[:, 1] if probabilities is not None else [0.5] * len(predictions)
            })
            
            # Apply threshold
            signals['signal'] = np.where(
                (signals['predicted'] == 1) & (signals['confidence'] >= threshold),
                'BUY',
                np.where(
                    (signals['predicted'] == 0) & (signals['confidence'] >= threshold),
                    'SELL',
                    'HOLD'
                )
            )
            
        else:  # regression
            predictions = model.predict(X_test)
            
            signals = pd.DataFrame({
                'actual': y_test.values,
                'predicted': predictions,
                'predicted_change': predictions
            })
            
            # Create signals based on predicted sentiment change
            signals['signal'] = np.where(
                signals['predicted_change'] > 0.05,  # Significant positive change
                'BUY',
                np.where(
                    signals['predicted_change'] < -0.05,  # Significant negative change
                    'SELL',
                    'HOLD'
                )
            )
        
        # Calculate signal accuracy
        if target_type == 'classification':
            correct_signals = (
                ((signals['signal'] == 'BUY') & (signals['actual'] == 1)) |
                ((signals['signal'] == 'SELL') & (signals['actual'] == 0)) |
                (signals['signal'] == 'HOLD')
            )
        else:
            correct_signals = (
                ((signals['signal'] == 'BUY') & (signals['actual'] > 0)) |
                ((signals['signal'] == 'SELL') & (signals['actual'] < 0)) |
                (signals['signal'] == 'HOLD')
            )
        
        signal_accuracy = correct_signals.mean()
        
        print(f"\nTrading Signal Analysis for {stock}:")
        print(f"Signal Accuracy: {signal_accuracy:.4f}")
        print(f"Signal Distribution:")
        print(signals['signal'].value_counts())
        
        return signals
    
    def run_complete_analysis(self, stocks: list = None, target_types: list = None):
        """
        Run complete analysis for multiple stocks and target types.
        
        Args:
            stocks: List of stock symbols to analyze
            target_types: List of target types ('classification', 'regression')
        """
        if stocks is None:
            stocks = ['MSFT', 'V', 'VZ', 'WBD', 'SNY']
        
        if target_types is None:
            target_types = ['classification', 'regression']
        
        # Initialize models
        self.create_models()
        
        # Run analysis for each stock and target type
        for stock in stocks:
            for target_type in target_types:
                print(f"\n{'='*50}")
                print(f"Analyzing {stock} - {target_type}")
                print(f"{'='*50}")
                
                # Train models
                self.train_models(stock, target_type)
                
                # Evaluate models
                self.evaluate_models(stock, target_type)
                
                # Create visualizations
                self.plot_results(stock, target_type)
                
                # Create trading signals
                signals = self.create_trading_signals(stock, target_type)
        
        # Summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a summary report of all results."""
        print(f"\n{'='*60}")
        print("SUMMARY REPORT")
        print(f"{'='*60}")
        
        summary_data = []
        
        for stock in self.results:
            for target_type in self.results[stock]:
                results = self.results[stock][target_type]
                
                for model_name, result in results.items():
                    metrics = result['metrics']
                    
                    if target_type == 'classification':
                        summary_data.append({
                            'Stock': stock,
                            'Target Type': target_type,
                            'Model': model_name,
                            'Accuracy': metrics['accuracy'],
                            'CV Score': metrics['cv_mean']
                        })
                    else:
                        summary_data.append({
                            'Stock': stock,
                            'Target Type': target_type,
                            'Model': model_name,
                            'R²': metrics['r2'],
                            'CV Score': metrics['cv_mean']
                        })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Best model per stock and target type
        best_models = summary_df.loc[summary_df.groupby(['Stock', 'Target Type'])['CV Score'].idxmax()]
        
        print("\nBest Models by Stock and Target Type:")
        print(best_models[['Stock', 'Target Type', 'Model', 'CV Score']].to_string(index=False))
        
        # Overall best models
        classification_results = summary_df[summary_df['Target Type'] == 'classification']
        regression_results = summary_df[summary_df['Target Type'] == 'regression']
        
        if not classification_results.empty:
            print(f"\nOverall Best Classification Model:")
            best_class = classification_results.loc[classification_results['CV Score'].idxmax()]
            print(f"Stock: {best_class['Stock']}, Model: {best_class['Model']}, Score: {best_class['CV Score']:.4f}")
        
        if not regression_results.empty:
            print(f"\nOverall Best Regression Model:")
            best_reg = regression_results.loc[regression_results['CV Score'].idxmax()]
            print(f"Stock: {best_reg['Stock']}, Model: {best_reg['Model']}, Score: {best_reg['CV Score']:.4f}")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SentimentMLPipeline()
    
    # Run complete analysis
    pipeline.run_complete_analysis(
        stocks=['MSFT', 'V'],  # Start with a subset for testing
        target_types=['classification', 'regression']
    ) 