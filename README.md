# comparative-analysis-energy-timeseries-forecasting-semml

## Overview

This research focuses on a comparative study of machine learning and deep learning models for energy time series forecasting. The study evaluates and contrasts the performance of the following models:

- **ARIMA** (Auto-Regressive Integrated Moving Average)
- **LSTM** (Long Short-Term Memory)
- **HCNN** (Historically Consistent Neural Networks)
- **BiLSTM** (Bidirectional Long Short-Term Memory)

The aim is to identify the strengths and weaknesses of each model in predicting energy requirements based on historical data.

---

## Dataset

The dataset used in this study consists of electrical load data for Germany, sourced from **ENTSO-E**, spanning from 2020 to 19th December 2024. To enhance the dataset, additional features were incorporated:

- **Weather-related features:** Average Temperature data of 20 different states of Germany is retrieved using the Python library **Meteostat**.
- **Time-related features:** Covariates such as hour, day of the year, month and day of the week were created to capture temporal patterns and trends in energy consumption.

This enriched dataset provides a robust foundation for building models that can learn from both historical load patterns and external influencing factors like weather.

---

## Results

Below are the visual summaries of the results:

### Line Graph


- **Description:** A comparison of actual vs. predicted energy values for all models over the evaluation period.

### Box Plot

- **Description:** A statistical distribution of the prediction errors (e.g., MAE, RMSE) for each model.




---

## Conclusion

- **ARIMA:** Performed well for linear trends but struggled with capturing non-linear patterns.
- **LSTM:** Demonstrated strong performance on capturing long-term dependencies but required significant computational resources.
- **HCNN:** Ensured temporal consistency but had longer training times for larger datasets.
- **BiLSTM:** Achieved high accuracy by leveraging past and future context but had increased complexity.

Overall, **BiLSTM** showed the best performance in terms of accuracy and adaptability to energy time series data.

---

## How to Use

1. Clone the repository: `git clone https://github.com/your-repo/comparative-analysis-energy-timeseries-forecasting-semml.git`
2. Clone the repository inside the comparative-analysis-energy-timeseries-forecasting-semml folder: `git clone https://github.com/uselessskills/hcnn?tab=readme-ov-file`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the models: `python main.py`
4. View results in the `results/` folder.

