#!/usr/bin/env python
# coding: utf-8

# #### import pandas_datareader.data as web
# 
# df = web.get_data_yahoo('6569.TWO','01/01/2015',interval='m')

# In[58]:


## importing necessary packages
import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf


# In[14]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# In[22]:


## importing data from yahoo finance
import pandas_datareader.data as web
df = web.get_data_yahoo('4142.TW','01/01/2018',interval='m')
df


# In[23]:


# Import the plotting library
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot the close price of the AAPL
df['Close'].plot()
plt.show()


# In[113]:


## Spliting data for modeling
train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
plt.figure(figsize=(12,7))
plt.title('Forecasted')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.plot(df['Open'], 'blue', label='Training Data')
plt.plot(test_data['Open'], 'green', label='Testing Data')
plt.xticks(np.arange(0,7982, 2000), df['Date'][0:7982:1300])
plt.legend()


# In[114]:


## Modeling the data using ARIMA method
train_ar = train_data['Open'].values
test_ar = test_data['Open'].values
history = [x for x in train_ar]
print(type(history))
predictions = list()
for t in range(len(test_ar)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
error = mean_squared_error(test_ar, predictions)
print('Testing Mean Squared Error: %.3f' % error)
error2 = smape_kun(test_ar, predictions)
print('Symmetric mean absolute percentage error: %.3f' % error2)


# In[115]:


## using ARIMA method
plt.figure(figsize=(12,7))
plt.plot(df['Open'], 'green', color='blue', label='Training Data')
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', 
         label='Predicted Price')
plt.plot(test_data.index, test_data['Open'], color='red', label='Actual Price')
plt.title(' Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(0,7982, 1300), df['Date'][0:7982:1300])
plt.legend()


# In[116]:


plt.figure(figsize=(12,7))
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed',label='Predicted Price')
plt.plot(test_data.index, test_data['Open'], color='red', label='Actual Price')
plt.legend()
plt.title('Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(6386,7982, 300), df['Date'][6386:7982:300])
plt.legend()


# In[24]:


## Second Try to Forecast
timeseries = df["Close"]


# In[25]:


from statsmodels.tsa.stattools import adfuller
print("p-value:", adfuller(timeseries.dropna())[1])


# In[26]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(311)
fig = plot_acf(timeseries, ax=ax1,
               title="Autocorrelation on Original Series") 
ax2 = fig.add_subplot(312)
fig = plot_acf(timeseries.diff().dropna(), ax=ax2, 
               title="1st Order Differencing")
ax3 = fig.add_subplot(313)
fig = plot_acf(timeseries.diff().diff().dropna(), ax=ax3, 
               title="2nd Order Differencing")


# In[27]:


plot_pacf(timeseries.diff().dropna(), lags=12)


# In[28]:


plot_acf(timeseries.diff().dropna())


# In[29]:


from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(timeseries, order=(2, 1, 1))
results = model.fit()
results.plot_predict(1, 100)

