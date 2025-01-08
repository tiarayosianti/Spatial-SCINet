
# Spatial SCINet: Spatial Sample Convolution and Interaction Network  

## Introduction  
One unique characteristic of time series data is that most of the information remains preserved even after downsampling. SCINet leverages this unique property of time series data by employing a **downsample-convolve-interact** approach to achieve higher predictive accuracy.  

In the research conducted by Liu et al. (2021), SCINet was applied to both time series and spatial modeling tasks. For spatial modeling, SCINet's performance was compared with deep learning models based on Graph Neural Networks (GNN) such as GraphWaveNet, DCRNN, STGCN, ASTGCN(r), STSGCN, STFGNN, AGCRN, and LSGCN. The study concluded that SCINet achieved superior performance compared to the other models, even without explicitly modeling spatial relationships.  

This result highlights the need for a method capable of extracting spatial information from data for spatial modeling tasks. To address this gap, the **Spatial SCINet** method was introduced.  

## What is Spatial SCINet?  
The **Spatial Sample Convolution and Interaction Network (Spatial SCINet)** combines the power of **Conv2D** and **SCINet** to handle both spatial and temporal aspects of the data effectively:  
- **Conv2D**: Used to extract spatial information from the data.  
- **SCINet**: Designed to capture temporal patterns in the data and make predictions.  

By integrating these two components, Spatial SCINet can address spatial and temporal modeling tasks more comprehensively, delivering better predictive performance.  

## Features of Spatial SCINet  
- **Temporal Modeling**: Utilizes SCINet's downsampling mechanism to learn temporal dependencies efficiently.  
- **Spatial Extraction**: Applies Conv2D to extract spatial relationships within the data.  

## References  
- This repository is adapted from https://github.com/HiddeKanger/SCINet.
- Liu, M., Zeng, A., Xu, Z., Lai, Q., & Xu, Q. (2021). Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction. arXiv preprint arXiv:2106.09305, 1(9).
