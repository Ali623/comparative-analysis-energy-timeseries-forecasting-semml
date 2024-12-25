from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import ARIMA, RNNModel
from sklearn.model_selection import TimeSeriesSplit
import os


class ModelConfig(BaseModel):
    forecast_horizon: int = Field(..., description="Forecasting horizon in hours")
    training_horizon: int = Field(..., description="Training horizon in hours")
    n_splits: int = Field(..., description="Number of train-test splits")
    model_name: str = Field(..., description="Name of the model")
    output_dir: str = Field(..., description="Directory to save CSV outputs")


def preprocess_data(dataset: pd.DataFrame, config: ModelConfig, is_covariate: bool = False):
    """
    Preprocess the dataset into a Darts TimeSeries object.

    Parameters:
    - dataset (pd.DataFrame): The input dataset.
    - config (ModelConfig): The configuration for the model.
    - is_covariate (bool): Whether the data being processed is a covariate.
    
    Returns:
    - TimeSeries: A Darts TimeSeries object for the target or covariates.
    """
    return TimeSeries.from_dataframe(
            dataset, 
            time_col="timestamp", 
            value_cols="Actual Load [MW]"
        )



def train_and_forecast(dataset: pd.DataFrame, config: ModelConfig):
    """
    Train a model and forecast based on the provided dataset and configuration.
    """
    series = preprocess_data(dataset, config)

    # Convert TimeSeries to a Pandas DataFrame for compatibility
    series_df = series.pd_dataframe()

    forecasts = []

    
    # Custom sliding window split logic
    total_length = len(series_df)
    train_size = config.training_horizon
    test_size = config.forecast_horizon
    splits = []

    for i in reversed(range(config.n_splits)):
        train_start = total_length - train_size - test_size - i * test_size
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size

        if train_start < 0 or test_end > total_length:
            break

        train_indices = np.arange(train_start, train_end)
        test_indices = np.arange(test_start, test_end)
        splits.append((train_indices, test_indices))

    # Iterate over the splits
    for train_idx, test_idx in splits:
        train = TimeSeries.from_dataframe(series_df.iloc[train_idx])
        test = TimeSeries.from_dataframe(series_df.iloc[test_idx])


        # Dynamically select the model name
        if config.model_name == "ARIMA":
            model = ARIMA()
            # Fit the model
            model.fit(train)
            forecast = model.predict(config.forecast_horizon)
            
        else:
            raise ValueError(f"Unsupported model name: {config.model_name}")

        # Save the forecast and timestamps
        forecast_df = forecast.pd_dataframe()
        forecast_df.reset_index(inplace=True)  # Ensure the timestamp is in the DataFrame
        forecasts.append(forecast_df)

    # Concatenate all forecasts into a single DataFrame
    combined_forecasts = pd.concat(forecasts, axis=0)
    combined_forecasts.columns = ['timestamp', 'forecast']

    # Save the forecasts
    save_forecast(combined_forecasts, config)


def save_forecast(forecasts: pd.DataFrame, config: ModelConfig):
    """
    Save the forecast results to a CSV file.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, f"{config.model_name}_forecasts.csv")
    forecasts.to_csv(output_path, index=False)
    print(f"Forecasts saved to {output_path}")


# main.py file example
if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv("Enriched_data.csv")
    
    # Define ARIMA Configuration
    arima_config = ModelConfig(
        forecast_horizon=24,
        training_horizon=720,
        n_splits=432,
        model_name="ARIMA", 
        output_dir="outputs"
    )

    # Execute ARIMA Model
    train_and_forecast(dataset=dataset, config=arima_config)
