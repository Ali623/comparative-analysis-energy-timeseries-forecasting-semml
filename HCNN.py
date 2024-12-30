import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from hcnn.hcnn.hcnn import HCNN
from hcnn.hcnn.loss_functions import LogCosh
import torch
import os
from datetime import datetime, timedelta
import re


class ModelConfig(BaseModel):
    forecast_horizon: int = Field(..., description="Forecasting horizon in hours")
    training_horizon: int = Field(..., description="Training horizon in hours")
    n_splits: int = Field(..., description="Number of train-test splits")
    model_name: str = Field(..., description="Name of the model")
    output_dir: str = Field(..., description="Directory to save CSV outputs")
    time: int | None = Field(..., description="Execution time for the forecast")


def preprocess_data(dataset: pd.DataFrame, config: ModelConfig, iteration : int):

    df = dataset[["timestamp", "Actual Load [MW]"]].copy()
    # df.set_index("timestamp", inplace = True)

    start_index = df.shape[0] - config.training_horizon - config.forecast_horizon - iteration * config.forecast_horizon
    end_index = start_index + config.training_horizon + config.forecast_horizon
    start_datetime = df["timestamp"].iloc[start_index - 1] 
    end_datetime = df["timestamp"].iloc[end_index - 1]
    df.drop("timestamp", axis = 1,  inplace = True)

    df.index = range(0, len(df))
    df = df[start_index : end_index]
    max, min = df.max(), df.min()
    df = (df - min)/(max - min)

    data_train = df[: config.training_horizon]
    test = df[config.training_horizon :]

    df = pd.DataFrame(df)
    data_train = pd.DataFrame(data_train)
    test = pd.DataFrame(test)
    
    return df, data_train, test, max, min, start_datetime, end_datetime



def train_and_forecast(dataset: pd.DataFrame, config: ModelConfig):
    """
    Train a model and forecast based on the provided dataset and configuration.
    """
    # df = dataset["Actual Load [MW]"].copy()

    combined_forecast = pd.DataFrame()

    # Start time
    config.time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    for i in reversed(range(config.n_splits)):

        df, data_train, test, max, min, start_datetime, end_datetime = preprocess_data(dataset, config, iteration = i)

        hcnn = HCNN(data_dim=data_train.shape[1], hidden_dim=200, sparsity=0, init_state_trainable=True)
        init_state = hcnn.init_state()

        train_loss = hcnn.train(
            data_train.values, 
            init_state, 
            lr=0.0005, epochs=1000, 
            criterion=LogCosh.apply, #LogCosh.apply, torch.nn.MSELoss()
            reduce_lr_epochs = 900, 
            plot_pred_train=False
        )

        y_pred_test = hcnn.sample(hcnn.forward(init_state, len(data_train)), len(test))
        y_pred = hcnn.sample(init_state, len(data_train))

        y_pred_test = pd.DataFrame(data=y_pred_test, columns=df.columns)#.rename('forecast from the end of train'])
        y_pred_test.index = df.loc[data_train.index[-1]+1:].index
        df_forecast_test = diff_inverse_transform(y_pred_test, df.loc[[data_train.index[-1]]])
        df_forecast_test.rename(columns={'Actual Load [MW]': 'forecast'}, inplace=True)
        
        # Generate the timestamp column with 1-hour resolution
        # Convert the string to a datetime object
        end_datetime = datetime.strptime(end_datetime, "%Y-%m-%d %H:%M:%S")
        end_datetime -= timedelta(hours=23)

        timestamps = pd.date_range(start=end_datetime, end=end_datetime + timedelta(hours=23) , freq='h')
        df_forecast_test["timestamp"] = timestamps
        df_forecast_test["split"] = config.n_splits - i

        combined_forecast = pd.concat([combined_forecast, df_forecast_test], ignore_index=True)
    

    combined_forecast["model_name"] = config.model_name
    combined_forecast["training_horizon"]  = config.training_horizon
    combined_forecast["forecast_horizon"]  = config.forecast_horizon
    combined_forecast.columns = ['forecast', 'timestamp',  'split', 'model_name', 'training_horizon', 'forecast_horizon']

    # Save the forecasts
    save_forecast(combined_forecast, config)

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


def diff_inverse_transform(df, first_row, max = 73442.25, min =  41132.5):
    df_ = first_row.copy()  # Copy to avoid modifying the original
    for i in range(len(df)):
        # Compute the row and ensure it's 2D
        row_values = (max - min)*(df.iloc[[i]].values) + min  # Use last row of df_ for addition
        row = pd.DataFrame(
            data=row_values,  # Ensure data is 2D
            columns=df.columns,
            index=df.iloc[[i]].index,
        )
        
        # Concatenate instead of appending
        df_ = pd.concat([df_, row])
    return df_.iloc[1:]  # Drop the first row if it was only for initialization


# main.py file example
if __name__ == "__main__":
    # Load dataset
    # dataset = pd.read_csv("Enriched_data.csv")
    
    # Experiment 1
    # config = ModelConfig(
    #     forecast_horizon=24,
    #     training_horizon=2*7*24, # 2 weeks
    #     n_splits=30,
    #     model_name="HCNN", 
    #     output_dir="outputs",
    #     time = None
    # )

    # Experiment 2
    # config = ModelConfig(
    #     forecast_horizon=24,
    #     training_horizon=4*7*24, # 4 weeks
    #     n_splits=30,
    #     model_name="HCNN", 
    #     output_dir="outputs",
    #     time = None
    # )


    # Experiment 3
    # config = ModelConfig(
    #     forecast_horizon=24,
    #     training_horizon=8*7*24, # 8 weeks
    #     n_splits=30,
    #     model_name="HCNN", 
    #     output_dir="outputs",
    #     time = None
    # )

    # Experiment 4
    # config = ModelConfig(
    #     forecast_horizon=24,
    #     training_horizon=16*7*24, # 16 weeks
    #     n_splits=30,
    #     model_name="HCNN", 
    #     output_dir="outputs",
    #     time = None
    # )


    # Experiment 5
    # config = ModelConfig(
    #     forecast_horizon=24,
    #     training_horizon=32*7*24, # 32 weeks
    #     n_splits=30,
    #     model_name="HCNN", 
    #     output_dir="outputs",
    #     time = None
    # )

    # Experiment 6
    # config = ModelConfig(
    #     forecast_horizon=24,
    #     training_horizon=48*7*24, # 48 weeks
    #     n_splits=30,
    #     model_name="HCNN", 
    #     output_dir="outputs",
    #     time = None
    # )

    # Execute hcnn Model
    # train_and_forecast(dataset=dataset, config=config)

    
    
    #### Just to automate the task ####
    # Define a function to run multiple experiments
    def run_experiments(base_config, training_horizons):
        for i, training_horizon in enumerate(training_horizons, start=1):
            # Update the configuration for each experiment
            config = ModelConfig(
                forecast_horizon=base_config["forecast_horizon"],
                training_horizon=training_horizon,
                n_splits=base_config["n_splits"],
                model_name=base_config["model_name"],
                output_dir=f"{base_config['output_dir']}",
                time=base_config["time"]
            )

            print(f"Running Experiment {i} with training horizon: {training_horizon} hours...")
            # Execute the model
            dataset = pd.read_csv("Enriched_data.csv")
            train_and_forecast(dataset=dataset, config=config)

    # Define the base configuration
    base_config = {
        "forecast_horizon": 24,
        "n_splits": 30,
        "model_name": "HCNN",
        "output_dir": "outputs",
        "time": None
    }

    # Define the training horizons for the experiments
    training_horizons = [
        # 2 * 7 * 24,  # 2 weeks
        # 4 * 7 * 24,  # 4 weeks
        # 8 * 7 * 24,  # 8 weeks
        # 16 * 7 * 24, # 16 weeks
        32 * 7 * 24, # 32 weeks
        48 * 7 * 24  # 48 weeks
    ]

    # Run the experiments
    run_experiments(base_config, training_horizons)