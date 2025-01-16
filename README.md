
# Spatial Sample Convolution and Interaction Network (Spatial SCINet) Modeling for Handling Spatial Heterogeneity and Temporal Variation (Case Study of PM10 Concentration in DKI Jakarta, 2023)  

## Introduction  
One unique characteristic of time series data is that most of the information remains preserved even after downsampling. SCINet leverages this unique property of time series data by employing a **downsample-convolve-interact** approach to achieve higher predictive accuracy.  

SCINet is a deep learning method discovered in 2021 by Liu et al. In the research conducted by Liu et al. (2021), SCINet was applied to both time series and spatial modeling tasks. For spatial modeling, SCINet's performance was compared with deep learning models based on Graph Neural Networks (GNN) such as GraphWaveNet, DCRNN, STGCN, ASTGCN(r), STSGCN, STFGNN, AGCRN, and LSGCN. The study concluded that SCINet achieved superior performance compared to the other models, even without explicitly modeling spatial relationships.  

This result highlights the need for a method capable of extracting spatial information from data for spatial modeling tasks. To address this gap, the **Spatial SCINet** method was introduced.  

## What is Spatial SCINet?  
The **Spatial Sample Convolution and Interaction Network (Spatial SCINet)** combines the power of **Conv2D** and **SCINet** to handle both spatial and temporal aspects of the data effectively:  
- **Conv2D**: Used to extract spatial information from the data.  
- **SCINet**: Designed to capture temporal patterns in the data and make predictions.  

By integrating these two components, Spatial SCINet can address spatial and temporal modeling tasks more comprehensively, delivering better predictive performance.  

## Features of Spatial SCINet  
- **Temporal Modeling**: Utilizes SCINet's downsampling mechanism to learn temporal dependencies efficiently.  
- **Spatial Extraction**: Applies two layers of Conv2D to extract spatial relationships within the data.  

## Dataset
The dataset used in this project includes spatial and temporal observations for DKI Jakarta from January 1, 2023, to December 31, 2023. The primary variables include:
- **Dependent variable:** PM10 concentration (Particulate Matter 10).
- **Independent variables:**
  - **Rainfall:** Measured in inches.
  - **Wind speed:** Measured in miles per hour.
  - **Humidity:** Measured as a percentage.
  - **Temperature:** Measured in degrees Fahrenheit.
- **Spatial variables:** Latitude and Longitude of observation locations.
- **Temporal variable:** Time (Date).


## Methodology
1. **Data Preparation:**
   - **Data Cleaning:** Imputation of missing values.
   - **Data Scaling:** Zero mean normalization.
   - **Data Splitting:**
     - 60% of the data is used for training.
     - 20% of the data is used for validation.
     - 20% of the data is used for testing.

2. **Model Development:**
   - Parameter initiation of the model using ParameterGrid function from sklearn package dan model selection package in Python. 

3. **Model Evaluation:**
   - Evaluation of model fit and performance metrics using Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Root Mean Squared Error (RMSE).


## Results
- The best model chosen has activation function Leaky ReLU in the Conv2D layers, a batch size of 16, a hid size of 8, and a learning rate of 0.0001.
- On the test data, the best model achieved MAPE: 12.87%, MAE: 7.40, and RMSE: 9.45.


## References  
- This repository is adapted from https://github.com/HiddeKanger/SCINet.
- Liu, M., Zeng, A., Xu, Z., Lai, Q., & Xu, Q. (2021). Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction. arXiv preprint arXiv:2106.09305, 1(9).
