from pydantic import BaseModel, Field
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA, RNNModel
from sklearn.model_selection import TimeSeriesSplit
# from darts.utils.timeseries_generation import split_series
import os


class ModelConfig(BaseModel):
    forecast_horizon: int = Field(..., description="Forecasting horizon in hours")
    training_horizon: int = Field(..., description="Training horizon in hours")
    n_splits: int = Field(..., description="Number of train-test splits")
    model_name: str = Field(..., description="Name of the model")
    output_dir: str = Field(..., description="Directory to save CSV outputs")


def preprocess_data(dataset: pd.DataFrame):
    """
    Preprocess the dataset into a Darts TimeSeries object.
    """
    return TimeSeries.from_dataframe(dataset, "timestamp", "Actual Load [MW]")


def train_and_forecast(dataset: pd.DataFrame, config: ModelConfig):
    """
    Train an ARIMA model and forecast based on the provided dataset and configuration.
    """
    series = preprocess_data(dataset)
    forecasts = []
    tscv =  TimeSeriesSplit(n_splits=config.n_splits)

     # Convert TimeSeries to a Pandas DataFrame for compatibility with TimeSeriesSplit
    series_df = series.pd_dataframe()

    for train_idx, test_idx in tscv.split(series):
    # for i in range(config.n_splits):
        # train, test = split_series(series, config.forecast_horizon)
        # train, test = series[train_idx], series[test_idx]
        train = TimeSeries.from_dataframe(series_df.iloc[train_idx])
        test = TimeSeries.from_dataframe(series_df.iloc[test_idx])
         # Dynamically select the model name
        if config.model_name == "ARIMA":
            model = ARIMA()
        elif config.model_name == "LSTM":
            model = RNNModel(
                model=config.model_name,
                input_chunk_length=config.training_horizon,
                output_chunk_length=config.forecast_horizon,
                n_epochs=100,
                random_state=42
            )
        elif config.model_name == "BiLSTM":
            model = RNNModel(
                model=config.model_name,
                input_chunk_length=config.training_horizon,
                output_chunk_length=config.forecast_horizon,
                n_epochs=100,
                random_state=42,
                bidirectional=True  # enable bidirectional LSTM
            )
        elif config.model_name == "HCNN":
            model = RNNModel(
                model=config.model_name,
                input_chunk_length=config.training_horizon,
                output_chunk_length=config.forecast_horizon,
                n_epochs=100,
                random_state=42,
                bidirectional=True  # enable bidirectional LSTM
            )
        elif config.model_name == "CRCNN":
            model = RNNModel(
                model=config.model_name,
                input_chunk_length=config.training_horizon,
                output_chunk_length=config.forecast_horizon,
                n_epochs=100,
                random_state=42,
                bidirectional=True  # enable bidirectional LSTM
            )
        else:
            raise ValueError(f"Unsupported model name: {config.model_name}")
        model.fit(train)
        forecast = model.predict(config.forecast_horizon)
        forecasts.append(forecast.pd_series())
    
    save_forecast(pd.concat(forecasts, axis=1), config)


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
        n_splits=5,
        model_name="ARIMA", #"LSTM", "BiLSTM", "HCNN", "CRCNN"
        output_dir="outputs"
    )

    # Execute ARIMA Model
    train_and_forecast(dataset=dataset, config=arima_config)
