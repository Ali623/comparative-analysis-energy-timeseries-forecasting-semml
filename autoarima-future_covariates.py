from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import AutoARIMA
from sklearn.model_selection import TimeSeriesSplit
import os
from datetime import datetime
import re


class ModelConfig(BaseModel):
    forecast_horizon: int = Field(..., description="Forecasting horizon in hours")
    training_horizon: int = Field(..., description="Training horizon in hours")
    n_splits: int = Field(..., description="Number of train-test splits")
    model_name: str = Field(..., description="Name of the model")
    output_dir: str = Field(..., description="Directory to save CSV outputs")
    time: int | None = Field(..., description="Execution time for the forecast")


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
            value_cols=["Actual Load [MW]", 'temperature', 'dayofweek', 'hour']
        )



def train_and_forecast(dataset: pd.DataFrame, config: ModelConfig):
    """
    Train a model and forecast based on the provided dataset and configuration.
    """
    series = preprocess_data(dataset, config)

    # Convert TimeSeries to a Pandas DataFrame for compatibility
    series_df = series.pd_dataframe()

    # perform min-max scaling
    max = series_df["Actual Load [MW]"].max()
    min = series_df["Actual Load [MW]"].min()

    series_df = (series_df - series_df.min())/(series_df.max() - series_df.min())

    future_covariates = series_df[["temperature","dayofweek","hour"]].copy()
    series_df = series_df[["Actual Load [MW]"]].copy()

    forecasts = []

    
    # Custom sliding window split logic
    total_length = len(series_df)
    train_size = config.training_horizon
    test_size = config.forecast_horizon
    splits = []
    # Start time
    config.time = datetime.now().strftime("%Y-%m-%d_%H-%M")
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
    for split_no, idx in enumerate(splits):
        train_idx, test_idx = idx
        train = TimeSeries.from_dataframe(series_df.iloc[train_idx,:])
        test = TimeSeries.from_dataframe(series_df.iloc[test_idx,:])

        # Future covariates
        future_covariates_train = TimeSeries.from_dataframe(future_covariates.iloc[train_idx,:])
        future_covariates_test = TimeSeries.from_dataframe(future_covariates.iloc[test_idx,:])


        # Dynamically select the model name
        if config.model_name == "AutoARIMA_wFC":
            model = AutoARIMA(start_p=3, max_p=10, start_q=1)
            # Fit the model
            model.fit(train, future_covariates=future_covariates_train)
            forecast = model.predict(config.forecast_horizon, future_covariates=future_covariates_test)
            
        else:
            raise ValueError(f"Unsupported model name: {config.model_name}")

        # Save the forecast and timestamps
        forecast_df = forecast.pd_dataframe()
        # inverse transform
        forecast_df = (max - min) * (forecast_df) + min

        forecast_df.reset_index(inplace=True)  # Ensure the timestamp is in the DataFrame
        forecast_df["split"] = split_no + 1
        forecasts.append(forecast_df)

    # Concatenate all forecasts into a single DataFrame
    combined_forecasts = pd.concat(forecasts, axis=0)


    combined_forecasts["model_name"] = config.model_name
    combined_forecasts["training_horizon"]  = config.training_horizon
    combined_forecasts["forecast_horizon"]  = config.forecast_horizon
    combined_forecasts.columns = ['timestamp', 'forecast',  'split', 'model_name', 'training_horizon', 'forecast_horizon']

    # Save the forecasts
    save_forecast(combined_forecasts, config)

def clean_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def save_forecast(forecasts: pd.DataFrame, config: ModelConfig):
    """
    Save the forecast results to a CSV file.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    start_time = clean_filename(str(config.time))
    output_path = os.path.join(config.output_dir, f"{config.model_name}_forecasts_TH_{config.training_horizon}_FH_{config.forecast_horizon}_{start_time}.csv")
    forecasts.to_csv(output_path, index=False)
    print(f"Forecasts saved to {output_path}")


# main.py file example
if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv("Enriched_data.csv")
    
    # Define ARIMA Configuration

    # Experiment 1
    # arima_config = ModelConfig(
    #     forecast_horizon=24,
    #     training_horizon=2*7*24, # 2 weeks
    #     n_splits=30,
    #     model_name="AutoARIMA_wFC", 
    #     output_dir="outputs",
    #     time = None
    # )

    # Experiment 2
    # arima_config = ModelConfig(
    #     forecast_horizon=24,
    #     training_horizon=4*7*24, # 4 weeks
    #     n_splits=30,
    #     model_name="AutoARIMA_wFC", 
    #     output_dir="outputs",
    #     time = None
    # )

    # Experiment 3
    # arima_config = ModelConfig(
    #     forecast_horizon=24,
    #     training_horizon=8*7*24, # 8 weeks
    #     n_splits=30,
    #     model_name="AutoARIMA_wFC", 
    #     output_dir="outputs",
    #     time = None
    # )

    # Experiment 4
    # arima_config = ModelConfig(
    #     forecast_horizon=24,
    #     training_horizon=16*7*24, # 16 weeks
    #     n_splits=30,
    #     model_name="AutoARIMA_wFC", 
    #     output_dir="outputs",
    #     time = None
    # )

    # Experiment 5
    # arima_config = ModelConfig(
    #     forecast_horizon=24,
    #     training_horizon=32*7*24, # 32 weeks
    #     n_splits=30,
    #     model_name="AutoARIMA_wFC", 
    #     output_dir="outputs",
    #     time = None
    # )

    # Experiment 6
    arima_config = ModelConfig(
        forecast_horizon=24,
        training_horizon=48*7*24, # 48 weeks
        n_splits=30,
        model_name="AutoARIMA_wFC", 
        output_dir="outputs",
        time = None
    )

    # Execute ARIMA Model
    train_and_forecast(dataset=dataset, config=arima_config)
