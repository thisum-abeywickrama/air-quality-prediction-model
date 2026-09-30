Developed machine learning and deep learning pipelines to forecast next-hour PM2.5 particulate concentrations using multi-station hourly meteorological and pollutant observations from the Beijing Air Quality Dataset.
Implemented end-to-end data preprocessing: forward/backward imputation, StandardScaler continuous normalization, cyclic temporal feature encoding, wind vector decomposition, exponential moving averages, and lag/rolling statistics.
Trained a Deep Gated Recurrent Unit (GRU, LayerNorm, GELU, Dropout) and a LightGBM Regressor using 5-fold Time-Series Cross-Validation with a 24-hour purge gap and Cosine Annealing learning rate schedule.
Formulated a weighted averaging ensemble model achieving a Test RMSE of 16.63 µg/m³ and Test MAE of 8.80 µg/m³.
