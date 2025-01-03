import plotly.graph_objects as go
import pandas as pd

if __name__ == "__main__":

    # Load datasets
    # Read Actual data
    df_actual = pd.read_csv(r"Enriched_data.csv")
    df_actual = df_actual[["timestamp", 'Actual Load [MW]']]
    df_actual = df_actual.iloc[-720:]
    df_actual.set_index("timestamp", inplace=True)


    # Read Autoarima forecasts
    df_forecast_arima_2w = pd.read_csv(r"outputs\AutoARIMA_forecasts_TH_336_FH_24_2024-12-28_15-39.csv")
    df_forecast_arima_2w.set_index("timestamp", inplace=True)
    df_forecast_arima_4w = pd.read_csv(r"outputs\AutoARIMA_forecasts_TH_672_FH_24_2024-12-28_15-47.csv")
    df_forecast_arima_4w.set_index("timestamp", inplace=True)
    df_forecast_arima_8w = pd.read_csv(r"outputs\AutoARIMA_forecasts_TH_1344_FH_24_2024-12-28_18-08.csv")
    df_forecast_arima_8w.set_index("timestamp", inplace=True)
    df_forecast_arima_16w = pd.read_csv(r"outputs\AutoARIMA_forecasts_TH_2688_FH_24_2024-12-28_18-13.csv")
    df_forecast_arima_16w.set_index("timestamp", inplace=True)
    df_forecast_arima_32w = pd.read_csv(r"outputs\AutoARIMA_forecasts_TH_5376_FH_24_2024-12-28_18-50.csv")
    df_forecast_arima_32w.set_index("timestamp", inplace=True)
    df_forecast_arima_48w = pd.read_csv(r"outputs\AutoARIMA_forecasts_TH_8064_FH_24_2024-12-28_19-53.csv")
    df_forecast_arima_48w.set_index("timestamp", inplace=True)


    # Read Autoarima with future covariates forecasts
    df_forecast_arima_wFC_2w = pd.read_csv(r"outputs\AutoARIMA_wFC_forecasts_TH_336_FH_24_2024-12-29_13-01.csv")
    df_forecast_arima_wFC_2w.set_index("timestamp", inplace=True)
    df_forecast_arima_wFC_4w = pd.read_csv(r"outputs\AutoARIMA_wFC_forecasts_TH_672_FH_24_2024-12-29_12-31.csv")
    df_forecast_arima_wFC_4w.set_index("timestamp", inplace=True)
    df_forecast_arima_wFC_8w = pd.read_csv(r"outputs\AutoARIMA_wFC_forecasts_TH_1344_FH_24_2024-12-29_13-10.csv")
    df_forecast_arima_wFC_8w.set_index("timestamp", inplace=True)
    df_forecast_arima_wFC_16w = pd.read_csv(r"outputs\AutoARIMA_wFC_forecasts_TH_2688_FH_24_2024-12-29_13-26.csv")
    df_forecast_arima_wFC_16w.set_index("timestamp", inplace=True)
    df_forecast_arima_wFC_32w = pd.read_csv(r"outputs\AutoARIMA_wFC_forecasts_TH_5376_FH_24_2024-12-29_14-39.csv")
    df_forecast_arima_wFC_32w.set_index("timestamp", inplace=True)
    df_forecast_arima_wFC_48w = pd.read_csv(r"outputs\AutoARIMA_wFC_forecasts_TH_8064_FH_24_2024-12-29_17-06.csv")
    df_forecast_arima_wFC_48w.set_index("timestamp", inplace=True)


    # Read LSTM forecasts
    df_forecast_LSTM_2w = pd.read_csv(r"outputs\LSTM_forecasts_TH_336_FH_24_2024-12-29_21-07.csv")
    df_forecast_LSTM_2w.set_index("timestamp", inplace=True)
    df_forecast_LSTM_4w = pd.read_csv(r"outputs\LSTM_forecasts_TH_672_FH_24_2024-12-29_21-08.csv")
    df_forecast_LSTM_4w.set_index("timestamp", inplace=True)
    df_forecast_LSTM_8w = pd.read_csv(r"outputs\LSTM_forecasts_TH_1344_FH_24_2024-12-29_21-09.csv")
    df_forecast_LSTM_8w.set_index("timestamp", inplace=True)
    df_forecast_LSTM_16w = pd.read_csv(r"outputs\LSTM_forecasts_TH_2688_FH_24_2024-12-29_21-11.csv")
    df_forecast_LSTM_16w.set_index("timestamp", inplace=True)
    df_forecast_LSTM_32w = pd.read_csv(r"outputs\LSTM_forecasts_TH_5376_FH_24_2024-12-29_21-15.csv")
    df_forecast_LSTM_32w.set_index("timestamp", inplace=True)
    df_forecast_LSTM_48w = pd.read_csv(r"outputs\LSTM_forecasts_TH_8064_FH_24_2024-12-29_21-25.csv")
    df_forecast_LSTM_48w.set_index("timestamp", inplace=True)


    # Read LSTM with future covariates forecasts
    df_forecast_LSTM_wFC_2w = pd.read_csv(r"outputs\LSTM_wFC_forecasts_TH_336_FH_24_2024-12-29_23-10.csv")
    df_forecast_LSTM_wFC_2w.set_index("timestamp", inplace=True)
    df_forecast_LSTM_wFC_4w = pd.read_csv(r"outputs\LSTM_wFC_forecasts_TH_672_FH_24_2024-12-29_23-11.csv")
    df_forecast_LSTM_wFC_4w.set_index("timestamp", inplace=True)
    df_forecast_LSTM_wFC_8w = pd.read_csv(r"outputs\LSTM_wFC_forecasts_TH_1344_FH_24_2024-12-29_23-12.csv")
    df_forecast_LSTM_wFC_8w.set_index("timestamp", inplace=True)
    df_forecast_LSTM_wFC_16w = pd.read_csv(r"outputs\LSTM_wFC_forecasts_TH_2688_FH_24_2024-12-29_23-14.csv")
    df_forecast_LSTM_wFC_16w.set_index("timestamp", inplace=True)
    df_forecast_LSTM_wFC_32w = pd.read_csv(r"outputs\LSTM_wFC_forecasts_TH_5376_FH_24_2024-12-29_23-19.csv")
    df_forecast_LSTM_wFC_32w.set_index("timestamp", inplace=True)
    df_forecast_LSTM_wFC_48w = pd.read_csv(r"outputs\LSTM_wFC_forecasts_TH_8064_FH_24_2024-12-29_23-31.csv")
    df_forecast_LSTM_wFC_48w.set_index("timestamp", inplace=True)

    # Read BiLSTM forecasts
    df_forecast_BiLSTM_2w = pd.read_csv(r"outputs\BiLSTM_forecasts_TH_336_FH_24_2024-12-29_21-43.csv")
    df_forecast_BiLSTM_2w.set_index("timestamp", inplace=True)
    df_forecast_BiLSTM_4w = pd.read_csv(r"outputs\BiLSTM_forecasts_TH_672_FH_24_2024-12-29_21-44.csv")
    df_forecast_BiLSTM_4w.set_index("timestamp", inplace=True)
    df_forecast_BiLSTM_8w = pd.read_csv(r"outputs\BiLSTM_forecasts_TH_1344_FH_24_2024-12-29_21-45.csv")
    df_forecast_BiLSTM_8w.set_index("timestamp", inplace=True)
    df_forecast_BiLSTM_16w = pd.read_csv(r"outputs\BiLSTM_forecasts_TH_2688_FH_24_2024-12-29_21-47.csv")
    df_forecast_BiLSTM_16w.set_index("timestamp", inplace=True)
    df_forecast_BiLSTM_32w = pd.read_csv(r"outputs\BiLSTM_forecasts_TH_5376_FH_24_2024-12-29_21-53.csv")
    df_forecast_BiLSTM_32w.set_index("timestamp", inplace=True)
    df_forecast_BiLSTM_48w = pd.read_csv(r"outputs\BiLSTM_forecasts_TH_8064_FH_24_2024-12-29_22-18.csv")
    df_forecast_BiLSTM_48w.set_index("timestamp", inplace=True)


    # Read BiLSTM with future covariates forecasts
    df_forecast_BiLSTM_wFC_2w = pd.read_csv(r"outputs\BiLSTMwFC_forecasts_TH_336_FH_24_2024-12-30_20-16.csv")
    df_forecast_BiLSTM_wFC_2w.set_index("timestamp", inplace=True)
    df_forecast_BiLSTM_wFC_4w = pd.read_csv(r"outputs\BiLSTMwFC_forecasts_TH_672_FH_24_2024-12-30_20-17.csv")
    df_forecast_BiLSTM_wFC_4w.set_index("timestamp", inplace=True)
    df_forecast_BiLSTM_wFC_8w = pd.read_csv(r"outputs\BiLSTMwFC_forecasts_TH_1344_FH_24_2024-12-30_20-18.csv")
    df_forecast_BiLSTM_wFC_8w.set_index("timestamp", inplace=True)
    df_forecast_BiLSTM_wFC_16w = pd.read_csv(r"outputs\BiLSTMwFC_forecasts_TH_2688_FH_24_2024-12-30_20-21.csv")
    df_forecast_BiLSTM_wFC_16w.set_index("timestamp", inplace=True)
    df_forecast_BiLSTM_wFC_32w = pd.read_csv(r"outputs\BiLSTMwFC_forecasts_TH_5376_FH_24_2024-12-30_20-28.csv")
    df_forecast_BiLSTM_wFC_32w.set_index("timestamp", inplace=True)
    df_forecast_BiLSTM_wFC_48w = pd.read_csv(r"outputs\BiLSTMwFC_forecasts_TH_8064_FH_24_2024-12-30_20-51.csv")
    df_forecast_BiLSTM_wFC_48w.set_index("timestamp", inplace=True)


    # Read HCNN forecasts
    df_forecast_HCNN_2w = pd.read_csv(r"outputs\HCNN_forecasts_TH_336_FH_24_2025-01-02_15-25.csv")
    df_forecast_HCNN_2w.set_index("timestamp", inplace=True)
    df_forecast_HCNN_4w = pd.read_csv(r"outputs\HCNN_forecasts_TH_672_FH_24_2024-12-30_02-49.csv")
    df_forecast_HCNN_4w.set_index("timestamp", inplace=True)
    df_forecast_HCNN_8w = pd.read_csv(r"outputs\HCNN_forecasts_TH_1344_FH_24_2024-12-30_04-47.csv")
    df_forecast_HCNN_8w.set_index("timestamp", inplace=True)
    df_forecast_HCNN_16w = pd.read_csv(r"outputs\HCNN_forecasts_TH_2688_FH_24_2024-12-29_04-55.csv")
    df_forecast_HCNN_16w.set_index("timestamp", inplace=True)
    df_forecast_HCNN_32w = pd.read_csv(r"outputs\HCNN_forecasts_TH_5376_FH_24_2025-01-02_16-35.csv")
    df_forecast_HCNN_32w.set_index("timestamp", inplace=True)
    df_forecast_HCNN_48w = pd.read_csv(r"outputs\HCNN_forecasts_TH_8064_FH_24_2025-01-03_01-14.csv")
    df_forecast_HCNN_48w.set_index("timestamp", inplace=True)


    # Initialize figure
    fig = go.Figure()


    # Add Traces
    # Add ground truth
    fig.add_trace(go.Scatter(x=df_actual.index, y=df_actual["Actual Load [MW]"], mode='lines', name='Ground Truth'))


    # Add week 2
    fig.add_trace(go.Scatter(x=df_forecast_arima_2w.index, y=df_forecast_arima_2w["forecast"], mode='lines', name='Arima'))
    fig.add_trace(go.Scatter(x=df_forecast_arima_wFC_2w.index, y=df_forecast_arima_wFC_2w["forecast"], mode='lines', name='Arima_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_2w.index, y=df_forecast_LSTM_2w["forecast"], mode='lines', name='LSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_wFC_2w.index, y=df_forecast_LSTM_wFC_2w["forecast"], mode='lines', name='LSTM_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_2w.index, y=df_forecast_BiLSTM_2w["forecast"], mode='lines', name='BiLSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_wFC_2w.index, y=df_forecast_BiLSTM_wFC_2w["forecast"], mode='lines', name='BiLSTMwFC'))
    fig.add_trace(go.Scatter(x=df_forecast_HCNN_2w.index, y=df_forecast_HCNN_2w["forecast"], mode='lines', name='HCNN'))


    # Add week 4
    fig.add_trace(go.Scatter(x=df_forecast_arima_4w.index, y=df_forecast_arima_4w["forecast"], mode='lines', name='Arima'))
    fig.add_trace(go.Scatter(x=df_forecast_arima_wFC_4w.index, y=df_forecast_arima_wFC_4w["forecast"], mode='lines', name='Arima_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_4w.index, y=df_forecast_LSTM_4w["forecast"], mode='lines', name='LSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_wFC_4w.index, y=df_forecast_LSTM_wFC_4w["forecast"], mode='lines', name='LSTM_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_4w.index, y=df_forecast_BiLSTM_4w["forecast"], mode='lines', name='BiLSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_wFC_4w.index, y=df_forecast_BiLSTM_wFC_4w["forecast"], mode='lines', name='BiLSTMwFC'))
    fig.add_trace(go.Scatter(x=df_forecast_HCNN_4w.index, y=df_forecast_HCNN_4w["forecast"], mode='lines', name='HCNN'))


    # Add week 8
    fig.add_trace(go.Scatter(x=df_forecast_arima_8w.index, y=df_forecast_arima_8w["forecast"], mode='lines', name='Arima'))
    fig.add_trace(go.Scatter(x=df_forecast_arima_wFC_8w.index, y=df_forecast_arima_wFC_8w["forecast"], mode='lines', name='Arima_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_8w.index, y=df_forecast_LSTM_8w["forecast"], mode='lines', name='LSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_wFC_8w.index, y=df_forecast_LSTM_wFC_8w["forecast"], mode='lines', name='LSTM_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_8w.index, y=df_forecast_BiLSTM_8w["forecast"], mode='lines', name='BiLSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_wFC_8w.index, y=df_forecast_BiLSTM_wFC_8w["forecast"], mode='lines', name='BiLSTMwFC'))
    fig.add_trace(go.Scatter(x=df_forecast_HCNN_8w.index, y=df_forecast_HCNN_8w["forecast"], mode='lines', name='HCNN'))


    # Add week 16
    fig.add_trace(go.Scatter(x=df_forecast_arima_16w.index, y=df_forecast_arima_16w["forecast"], mode='lines', name='Arima'))
    fig.add_trace(go.Scatter(x=df_forecast_arima_wFC_16w.index, y=df_forecast_arima_wFC_16w["forecast"], mode='lines', name='Arima_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_16w.index, y=df_forecast_LSTM_16w["forecast"], mode='lines', name='LSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_wFC_16w.index, y=df_forecast_LSTM_wFC_16w["forecast"], mode='lines', name='LSTM_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_16w.index, y=df_forecast_BiLSTM_16w["forecast"], mode='lines', name='BiLSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_wFC_16w.index, y=df_forecast_BiLSTM_wFC_16w["forecast"], mode='lines', name='BiLSTMwFC'))
    fig.add_trace(go.Scatter(x=df_forecast_HCNN_16w.index, y=df_forecast_HCNN_16w["forecast"], mode='lines', name='HCNN'))


    # Add week 32
    fig.add_trace(go.Scatter(x=df_forecast_arima_32w.index, y=df_forecast_arima_32w["forecast"], mode='lines', name='Arima'))
    fig.add_trace(go.Scatter(x=df_forecast_arima_wFC_32w.index, y=df_forecast_arima_wFC_32w["forecast"], mode='lines', name='Arima_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_32w.index, y=df_forecast_LSTM_32w["forecast"], mode='lines', name='LSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_wFC_32w.index, y=df_forecast_LSTM_wFC_32w["forecast"], mode='lines', name='LSTM_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_32w.index, y=df_forecast_BiLSTM_32w["forecast"], mode='lines', name='BiLSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_wFC_32w.index, y=df_forecast_BiLSTM_wFC_32w["forecast"], mode='lines', name='BiLSTMwFC'))
    fig.add_trace(go.Scatter(x=df_forecast_HCNN_32w.index, y=df_forecast_HCNN_32w["forecast"], mode='lines', name='HCNN'))


    # Add week 48
    fig.add_trace(go.Scatter(x=df_forecast_arima_48w.index, y=df_forecast_arima_48w["forecast"], mode='lines', name='Arima'))
    fig.add_trace(go.Scatter(x=df_forecast_arima_wFC_48w.index, y=df_forecast_arima_wFC_48w["forecast"], mode='lines', name='Arima_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_48w.index, y=df_forecast_LSTM_48w["forecast"], mode='lines', name='LSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_LSTM_wFC_48w.index, y=df_forecast_LSTM_wFC_48w["forecast"], mode='lines', name='LSTM_wFC'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_48w.index, y=df_forecast_BiLSTM_48w["forecast"], mode='lines', name='BiLSTM'))
    fig.add_trace(go.Scatter(x=df_forecast_BiLSTM_wFC_48w.index, y=df_forecast_BiLSTM_wFC_48w["forecast"], mode='lines', name='BiLSTMwFC'))
    fig.add_trace(go.Scatter(x=df_forecast_HCNN_48w.index, y=df_forecast_HCNN_48w["forecast"], mode='lines', name='HCNN'))


    # Update layout
    fig.update_layout(
        template='presentation',  # Dark background template
        title='Comparative Forecast',
        xaxis_title='Time',
        yaxis_title='Load [MW]',
        xaxis=dict(showgrid=True, gridcolor='gray'),
        yaxis=dict(showgrid=True, gridcolor='gray'),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        legend=dict(
            x=0,  # X-position (0 is left, 1 is right)
            y=1.08,  # Y-position (1 is top, 0 is bottom)
            traceorder='normal',
            orientation='h',  # Horizontal orientation
            bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for legend
        )
    )


    # Add menu
    fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="TH_2w",
                     method="update",
                     args=[{"visible": [True, 
                                        True, True, True, True, True, True, True,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,]},
                           {"title": "Comparative Forecast - Training Horizon 2w - Forecast Horizon 24h",
                            "annotations": []}]),
                dict(label="TH_4w",
                     method="update",
                     args=[{"visible": [True, 
                                        False, False, False, False, False, False, False,
                                        True, True, True, True, True, True, True,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,]},
                           {"title": "Comparative Forecast - Training Horizon 4w - Forecast Horizon 24h",
                            "annotations": []}]),
                dict(label="TH_8w",
                     method="update",
                     args=[{"visible": [True, 
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        True, True, True, True, True, True, True,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,]},
                           {"title": "Comparative Forecast - Training Horizon 8w - Forecast Horizon 24h",
                            "annotations": []}]),
                dict(label="TH_16w",
                     method="update",
                     args=[{"visible": [True, 
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        True, True, True, True, True, True, True,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,]},
                           {"title": "Comparative Forecast - Training Horizon 16w - Forecast Horizon 24h",
                            "annotations": []}]),
                dict(label="TH_32w",
                     method="update",
                     args=[{"visible": [True, 
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        True, True, True, True, True, True, True,
                                        False, False, False, False, False, False, False,]},
                           {"title": "Comparative Forecast - Training Horizon 32w - Forecast Horizon 24h",
                            "annotations": []}]),
                dict(label="TH_48w",
                     method="update",
                     args=[{"visible": [True, 
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False,
                                        True, True, True, True, True, True, True,]},
                           {"title": "Comparative Forecast - Training Horizon 48w - Forecast Horizon 24h",
                            "annotations": []}]),
            ]),
            direction="down",
            pad={"r": 5, "t": 5},
            showactive=True,
            x=0.9,
            xanchor="left",
            y=1.08,
            yanchor="top"
        )
    ])

    # Save the plot as an HTML file
    fig.write_html(f"results/lineplot_model_comparison.html")
    # Show Fig
    fig.show()