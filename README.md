---

# **Time Series Forecasting with N-BEATS Model**

This project demonstrates time-series forecasting using the N-BEATS (Neural Basis Expansion Analysis for Time Series) model with Python libraries such as Darts and Statsmodels.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Setup and Installation](#setup-and-installation)
3. [Data Preparation](#data-preparation)
4. [Model Building](#model-building)
5. [Evaluation](#evaluation)
6. [Results and Insights](#results-and-insights)
7. [Conclusion](#conclusion)
8. [References](#references)
9. [Author and Acknowledgments](#author-and-acknowledgments)

---

## **1. Introduction**
The project focuses on time-series forecasting using the N-BEATS model, leveraging the Darts library for modeling and evaluation. The primary objective is to build a robust model for accurate forecasting.

---

## **2. Setup and Installation**

### **Mount Google Drive (if using Colab)**
```python
from google.colab import drive
drive.mount('/content/drive')
```

### **Navigate to Working Directory**
```python
%cd /content/drive/MyDrive/Python - Time Series Forecasting/Deep Learning for Time Series Forecasting/N-BEATS
```

### **Install Required Libraries**
```python
!pip install -q darts
```

---

## **3. Data Preparation**

### **Import Required Libraries**
```python
# Standard Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error)

# Darts Functions
from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
```

---

### **Load Data**
```python
# Load the dataset
data = pd.read_csv("your_dataset.csv")
```

### **Data Preprocessing**
```python
# Preprocess the data (example)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
```

---

## **4. Model Building**

### **Model Initialization**
```python
model = NBEATSModel(
    input_chunk_length=30,
    output_chunk_length=7,
    n_epochs=100,
    num_stacks=2,
    num_blocks=3,
    num_layers=4,
    layer_widths=256,
)
```

### **Model Training**
```python
train, val = TimeSeries.split(data, 0.8)
model.fit(train)
```

---

## **5. Evaluation**

### **Model Predictions**
```python
forecast = model.predict(n=30, series=train)
```

### **Performance Metrics**
```python
mae = mean_absolute_error(val, forecast)
mse = mean_squared_error(val, forecast)
mape = mean_absolute_percentage_error(val, forecast)

print(f"MAE: {mae}, MSE: {mse}, MAPE: {mape}%")
```

---

## **6. Results and Insights**
- Visualizations of forecasted results
- Performance evaluation graphs
- Key insights and model accuracy

---

## **7. Conclusion**
- Summary of the process
- Challenges encountered and lessons learned

---

## **8. References**
- [Darts Documentation](https://github.com/unit8co/darts)
- [N-BEATS Paper](https://arxiv.org/abs/1905.10437)

---

## **9. Author and Acknowledgments**
- Project Author: [Your Name]
- Acknowledgments: Contributors, Tutorials, and Online Communities

---
