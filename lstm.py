from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import re


class ModelConfig(BaseModel):
    forecast_horizon: int = Field(..., description="Forecasting horizon in hours")
    training_horizon: int = Field(..., description="Training horizon in hours")
    n_splits: int = Field(..., description="Number of train-test splits")
    model_name: str = Field(..., description="Name of the model")
    output_dir: str = Field(..., description="Directory to save CSV outputs")
    time: int | None = Field(..., description="Execution time for the forecast")



def preprocess_data(dataset: pd.DataFrame, config: ModelConfig):
    # Parse the timestamp column
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset.set_index('timestamp', inplace=True)

    # Define the target variable and features
    target_column = 'Actual Load [MW]'
    feature_columns = ['temperature'] #, 'humidity', 'wind_speed', 'dayofyear', 'dayofmonth', 'dayofweek', 'hour', 'holiday']

    # Extract features and target variable
    X = dataset[feature_columns]
    y = dataset[target_column]

    # Normalize the features using MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    return X_scaled, y_scaled, scaler_X, scaler_y


def train_and_forecast(dataset: pd.DataFrame, config: ModelConfig):

    X_scaled, y_scaled, scaler_X, scaler_y = preprocess_data(dataset, config)

    # Prepare the data for LSTM (supervised learning, creating sequences)
    def create_sequences(X, y, seq_length = config.training_horizon, target_length = config.forecast_horizon):
        X_seq, y_seq = [], []
        for i in reversed(range(config.n_splits)):
            start = len(X) - seq_length - target_length - i * (target_length)
            end = start + seq_length
            X_seq.append(X[start:end])
            y_seq.append(y[end:end+target_length])  # Predict a sequence of 'target_length' time steps
        return np.array(X_seq), np.array(y_seq)

    # Create sequences (use 24 hours of previous data to predict the next 24 hours)
    seq_length = config.training_horizon # change the name to train horizon
    target_length = config.forecast_horizon
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length, target_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, shuffle=False)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=target_length))  # Predict a sequence of 'target_length' values

    model.compile(optimizer=Adam(), loss='mean_squared_error')

    # Start time
    config.time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Make predictions
    predictions_scaled = model.predict(X_test)

    # Inverse transform the scaled predictions and true values
    predictions = scaler_y.inverse_transform(predictions_scaled)
    predictions = predictions.flatten()

    # Reshape y_test to 2D for inverse transformation
    y_test_reshaped = y_test.reshape(-1, target_length)
    y_test_inverse = scaler_y.inverse_transform(y_test_reshaped)

    return predictions

def clean_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def save_forecast(forecasts: np.ndarray, dataset: pd.DataFrame, config: ModelConfig):
    """
    Save the forecast results to a CSV file, including timestamps.
    """
    # Get the timestamps from the dataset, assuming the dataset is ordered and contains the 'timestamp' column.
    forecast_timestamps = dataset.iloc[-len(forecasts):].index
    splits = np.repeat(range(1, int(config.n_splits * 0.3 + 1)), config.forecast_horizon)
    
    # Create a DataFrame for the forecasts and timestamps
    forecast_df = pd.DataFrame({
        'timestamp': forecast_timestamps,
        'forecast': forecasts,  # Ensure the forecast is a 1D array
        'model_name':config.model_name,
        'split':splits,
        'training_horizon':config.training_horizon,
        'forecast_horizon':config.forecast_horizon,
    })
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Define the output file path
    start_time = clean_filename(str(config.time))
    output_path = os.path.join(config.output_dir, f"{config.model_name}_forecasts_TH_{config.training_horizon}_FH_{config.forecast_horizon}_{start_time}.csv")
    
    # Save the DataFrame to a CSV file
    forecast_df.to_csv(output_path, index=False)
    
    print(f"Forecasts saved to {output_path}")



# main.py file example
if __name__ == "__main__":
    # Load the dataset
    dataset = pd.read_csv('Enriched_data.csv')

    # Example usage:
    config = ModelConfig(
        forecast_horizon=24,
        training_horizon=720,
        n_splits=1440,
        model_name="LSTM",  # or "LSTM"
        output_dir="outputs",
        time = None
        )

    # Execute ARIMA Model
    predictions = train_and_forecast(dataset=dataset, config=config)
    save_forecast(predictions, dataset, config)
