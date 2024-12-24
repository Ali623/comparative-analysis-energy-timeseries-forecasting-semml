from pydantic import BaseModel, Field
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA, RNNModel
from darts.dataprocessing.transformers import Scaler
from sklearn.model_selection import TimeSeriesSplit
import os

class ModelConfig(BaseModel):
    forecast_horizon: int = Field(..., description="Forecasting horizon in hours")
    training_horizon: int = Field(..., description="Training horizon in hours")
    n_splits: int = Field(..., description="Number of train-test splits")
    model_name: str = Field(..., description="Name of the model")
    output_dir: str = Field(..., description="Directory to save CSV outputs")

class BaseModelExecutor(BaseModel):
    config: ModelConfig
    dataset: pd.DataFrame

    def preprocess_data(self):
        raise NotImplementedError("Must implement preprocess_data method.")

    def train_and_forecast(self):
        raise NotImplementedError("Must implement train_and_forecast method.")

    def save_forecast(self, forecasts):
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, f"{self.config.model_name}_forecasts.csv")
        forecasts.to_csv(output_path, index=False)
        print(f"Forecasts saved to {output_path}")

class ARIMAExecutor(BaseModelExecutor):

    def preprocess_data(self):
        return TimeSeries.from_dataframe(self.dataset, "timestamp", "load")

    def train_and_forecast(self):
        series = self.preprocess_data()
        forecasts = []
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        for train_idx, test_idx in tscv.split(series):
            train, test = series[train_idx], series[test_idx]
            model = ARIMA()
            model.fit(train)
            forecast = model.predict(self.config.forecast_horizon)
            forecasts.append(forecast.pd_series())
        
        self.save_forecast(pd.concat(forecasts, axis=1))

class RNNExecutor(BaseModelExecutor):
    model_type: str

    def preprocess_data(self):
        series = TimeSeries.from_dataframe(self.dataset, "timestamp", "load")
        covariates = TimeSeries.from_dataframe(self.dataset, "timestamp", ["temp", "time_feature"])
        scaler = Scaler()
        return scaler.fit_transform(series), scaler.fit_transform(covariates)

    def train_and_forecast(self):
        series, covariates = self.preprocess_data()
        forecasts = []
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)

        for train_idx, test_idx in tscv.split(series):
            train, test = series[train_idx], series[test_idx]
            cov_train, cov_test = covariates[train_idx], covariates[test_idx]

            model = RNNModel(
                model=self.model_type,
                input_chunk_length=self.config.training_horizon,
                output_chunk_length=self.config.forecast_horizon,
                n_epochs=100,
                random_state=42
            )
            model.fit(train, past_covariates=cov_train)
            forecast = model.predict(self.config.forecast_horizon, past_covariates=cov_test)
            forecasts.append(forecast.pd_series())

        self.save_forecast(pd.concat(forecasts, axis=1))

class HCNNExecutor(RNNExecutor):

    def train_and_forecast(self):
        series, covariates = self.preprocess_data()
        forecasts = []
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)

        for train_idx, test_idx in tscv.split(series):
            train, test = series[train_idx], series[test_idx]
            cov_train, cov_test = covariates[train_idx], covariates[test_idx]

            model = RNNModel(
                model="TCN",
                input_chunk_length=self.config.training_horizon,
                output_chunk_length=self.config.forecast_horizon,
                n_epochs=100,
                random_state=42,
                kernel_size=3,
                num_filters=32,
                dropout=0.2
            )
            model.fit(train, past_covariates=cov_train)
            forecast = model.predict(self.config.forecast_horizon, past_covariates=cov_test)
            forecasts.append(forecast.pd_series())

        self.save_forecast(pd.concat(forecasts, axis=1))

class CRCNNExecutor(RNNExecutor):

    def train_and_forecast(self):
        series, covariates = self.preprocess_data()
        forecasts = []
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)

        for train_idx, test_idx in tscv.split(series):
            train, test = series[train_idx], series[test_idx]
            cov_train, cov_test = covariates[train_idx], covariates[test_idx]

            model = RNNModel(
                model="LSTM",
                input_chunk_length=self.config.training_horizon,
                output_chunk_length=self.config.forecast_horizon,
                n_epochs=100,
                random_state=42,
                kernel_size=3,
                num_filters=16,
                dropout=0.3
            )
            model.fit(train, past_covariates=cov_train)
            forecast = model.predict(self.config.forecast_horizon, past_covariates=cov_test)
            forecasts.append(forecast.pd_series())

        self.save_forecast(pd.concat(forecasts, axis=1))

# main.py file example
if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv("data/electrical_load.csv")
    
    # Define ARIMA Configuration
    arima_config = ModelConfig(
        forecast_horizon=24,
        training_horizon=720,
        n_splits=50,
        model_name="ARIMA",
        output_dir="outputs"
    )

    # Execute ARIMA Model
    arima_executor = ARIMAExecutor(config=arima_config, dataset=dataset)
    arima_executor.train_and_forecast()

    # Add similar executors for LSTM, HCNN, CRCNN, and BiLSTM here
