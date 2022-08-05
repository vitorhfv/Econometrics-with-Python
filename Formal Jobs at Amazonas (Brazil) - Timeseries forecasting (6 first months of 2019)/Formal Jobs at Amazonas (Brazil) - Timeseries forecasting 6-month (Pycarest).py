#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import plotly
import plotly.io as pio


# In[2]:


import pandas as pd
from bcb import sgs
import datetime as dt


# In[3]:


# Formal jobs at Amazonas state, Brazil, between 2010-2018

emprego_am=sgs.get({'empregos':12572}, start='2010-01-01', end='2018-12-01')


# In[4]:


emprego_am.index = pd.to_datetime(emprego_am.index) # transforms the index type into datatime


# In[5]:


emprego_am.plot()


# In[6]:


from pycaret.time_series import *


# In[7]:


setup(emprego_am, fh=12, fold_strategy='expanding', seasonal_period='M') 

# setup the data


# In[9]:


best_model=compare_models ()

# compare the models and choose the best one that fits.


# In[10]:


arima=create_model('arima')


# In[11]:


import matplotlib.pyplot as plt


# In[31]:


import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "kaggle"


# In[32]:


plot_model(arima)


# In[33]:


plot_model(arima, plot='train_test_split')


# In[34]:


final=finalize_model(arima)


# In[35]:


plot_model(final, plot='forecast')


# In[19]:

# Plotting a forecast of 6 (six months)
plot_model(final, plot='forecast', data_kwargs={'fh':6})


# In[ ]:


# Plotting a timeseries of what happenned during the interval of prediction above


# In[20]:


empregoam2=sgs.get({'empregos':12572}, start='2017-01-01', end='2019-06-30')


# In[93]:


empregoam2.plot()

