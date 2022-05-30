#!/usr/bin/env python
# coding: utf-8

# In[3]:


from bcb import sgs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sidetable as sbn
import datetime
import sklearn as sk
import statsmodels.api
import statsmodels.api as sm


# In[4]:


renda=sgs.get({'renda':29023}, start = '2005-01-01', end = '2010-12-01')
consumo=sgs.get({'consumo familias':22110}, start = '2005-01-01', end = '2010-12-01')


# In[3]:


consumo2=consumo.reset_index()
consumo2.head()


# In[5]:


renda2=renda


# In[8]:


renda3=renda2.reset_index()
renda3.head()


# In[52]:


renda4=renda3[renda3['Date'].dt.month.isin([1,4,7,10])]


# In[58]:


consumo2.reset_index(drop=True, inplace=True)


# In[62]:


renda4.reset_index(drop=True, inplace=True)


# In[63]:


renda_consumo=pd.concat([renda4,consumo2], axis=1)
renda_consumo.head(25)


# In[67]:


rendaconsumo=renda_consumo.iloc[0:24]


# In[68]:


sns.regplot(x='renda', y='consumo familias', data=rendaconsumo)


# In[69]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
model = LinearRegression()
X=rendaconsumo['renda']
y=rendaconsumo['consumo familias']


# In[70]:


y1=y.values.reshape(-1,1)
X1=X.values.reshape(-1,1)


# In[71]:


variables = list(rendaconsumo.columns)
y1 = rendaconsumo['consumo familias']
X1 = rendaconsumo['renda']
x = [var for var in variables if var not in y ]

# Ordinary least squares regression
model_Simple = y1, X1

# Add a constant term like so:
model = sm.OLS(y1, sm.add_constant(X1)).fit()

model.summary()


# In[72]:


#aumento na renda de R$1 leva a aumento no consumo de 0,43 centavos
# y=73,4786+0,4328x


# In[73]:


# o modelo explica em até 97% a variação do consumo em relação à renda

