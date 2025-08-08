# app.py

import streamlit as st
import pandas as pd
import torch
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn

# --- Model & Scaler Loading ---

# Define the Transformer model class again, so we can load the state_dict
class MultiStockTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, output_dim=5):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        encoded = self.transformer(x)
        out = self.decoder(encoded[-1])
        return out

@st.cache_resource
def load_models_and_scalers():
    """Load all necessary models and scalers from disk."""
    try:
        # Load the price prediction model (Transformer)
        price_model = MultiStockTransformer(input_dim=5, output_dim=5)
        price_model.load_state_dict(torch.load('multi_stock_transformer.pt'))
        price_model.eval()

        # Load the direction prediction model (Random Forest)
        direction_model = joblib.load('random_forest_classifier.joblib')
        
        # We need to recreate and fit the scalers to inverse the predictions
        # Load data to fit the scalers
        price_df = pd.read_csv('SNY.csv') # Using SNY as an example for the price scaler
        price_scaler = MinMaxScaler()
        price_scaler.fit(price_df[['close']])

    except FileNotFoundError as e:
        st.error(f"Error loading model file: {e}. Please ensure all model files are in the 'app' directory.")
        return None, None, None
        
    return price_model, direction_model, price_scaler

# --- Data Loading and Prediction Functions ---

@st.cache_data
def load_data(ticker):
    """Loads the data for a specific stock."""
    try:
        df = pd.read_csv(f'{ticker}.csv', parse_dates=['Date'])
        return df
    except FileNotFoundError:
        st.error(f"Data file for {ticker}.csv not found.")
        return None

def predict_price(model, price_data, scaler):
    """Predicts the next day's stock price."""
    # Prepare the sequence for the model
    last_sequence = price_data['close'].values[-20:] # Use last 20 days
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    
    # The model expects multiple features, we'll use the single price feature and duplicate it for the other dimensions as a placeholder
    # This part needs to be aligned perfectly with your training script's input
    model_input = np.tile(last_sequence_scaled, (1, 5))
    
    with torch.no_grad():
        input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0)
        prediction_scaled = model(input_tensor)
        
        # We only care about the prediction for the first stock (JPM in your training)
        # You'll need to adjust this if you want to select other stocks
        predicted_price = scaler.inverse_transform(prediction_scaled.numpy()[:, 0].reshape(-1, 1))
    
    return predicted_price[0][0]

# --- Streamlit UI ---

st.title("📈 Stock Price & Direction Predictor")

# Load all the models and scalers
price_model, direction_model, price_scaler = load_models_and_scalers()

if price_model and direction_model:
    st.success("Models loaded successfully!")

    # User Input
    stock_ticker = st.selectbox("Select a Stock for Prediction:", ('JPM', 'MSFT', 'SNY', 'VZ', 'WBD'))
    
    if st.button("Run Prediction"):
        # Price Prediction
        st.subheader(f"Price Prediction for {stock_ticker}")
        price_data = load_data(stock_ticker)
        
        if price_data is not None:
            predicted_price = predict_price(price_model, price_data, price_scaler)
            last_price = price_data['close'].iloc[-1]
            st.metric("Predicted Next Day Price", f"${predicted_price:.2f}", f"${predicted_price - last_price:.2f}")

        # Direction Prediction (using placeholder data for now)
        st.subheader("Market Direction Prediction")
        st.info("Note: The direction predictor uses features from the latest sentiment data, not just price.")
        
        # This part is simplified. A real app would fetch live sentiment data.
        # We'll use the last row of your sentiment data as a stand-in.
        sentiment_data = pd.read_csv('sny_sentiment_features_full.csv')
        latest_sentiment_features = sentiment_data[['sentiment_mean', 'sentiment_std', 'sentiment_count',
                                                    'sentiment_min', 'sentiment_max', 'relevance_mean',
                                                    'relevance_sum', 'overall_sentiment_mean',
                                                    'overall_sentiment_std', 'source_count']].tail(1)

        direction = direction_model.predict(latest_sentiment_features)[0]
        direction_map = {1: "Positive 🟢", 0: "Neutral ⚪", -1: "Negative 🔴"}

        st.metric("Predicted Market Direction", direction_map.get(direction, "Unknown"))
