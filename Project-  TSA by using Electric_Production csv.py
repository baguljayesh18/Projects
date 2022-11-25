#!/usr/bin/env python
# coding: utf-8

# # Problem statement
# 
# 
# 
# Dataset contains 2 columns only, one column is Date and the other column relates to the Production percentage.It shows the Production of electricity from 1985 till 2018.               
# The goal is to predict electricity Production for the next 4 years i.e. till 2022.

# In[51]:


from warnings import filterwarnings
filterwarnings("ignore")


# # Reading dataset

# In[52]:


import pandas as pd
A = pd.read_csv("C:/Users/Admin/Desktop/Self study\ML/Time series/Dataset/Electric_Production.csv")


pd.set_option('display.max_columns', None) # maximize the display value of column
pd.set_option('display.max_rows', None)


# In[53]:


A.shape


# # Preview

# In[54]:


A.head()


# In[55]:


A.DATE = pd.to_datetime(A.DATE)


# In[56]:


A.columns = ["DATE","Production"]


# In[57]:


A.head(3)


# # Date conversion

# In[58]:


A.index = A.DATE                       #Month made index
A = A.drop(labels=["DATE"],axis=1)   #column Month deleted.


# # Plot TS

# In[59]:


import matplotlib.pyplot as plt


# In[60]:


plt.plot(A)


# In[61]:


A.plot()


# # Decomposition

# In[62]:


from statsmodels.tsa.seasonal import seasonal_decompose
Q = seasonal_decompose(A,model='multiplicative')


# In[63]:


original = Q.observed        #here we got original time series and store in  "original" variable.


# In[64]:


trend = Q.trend              #here we got trend


# In[65]:


season = Q.seasonal          #here we got season


# In[66]:


Error = Q.resid             #here we got Error


# In[67]:


plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
original.plot()
plt.subplot(2,2,2)
trend.plot()

plt.subplot(2,2,3)
season.plot()

plt.subplot(2,2,4)
Error.plot()


# # Check stationarity of TS using rolling mean

# In[68]:


RM = A.rolling(window=12).mean()


# In[69]:


plt.plot(A,c="blue")
plt.plot(RM,c="red")


# # Attempt 1 for stationary conversion

# In[70]:


from numpy import log
LG = log(A)
LGRM = LG.rolling(window=12).mean()


# In[71]:


plt.plot(LG,c="blue")
plt.plot(LGRM,c="red")


# # Attempt 2(square root)

# In[72]:


from numpy import sqrt
SR = sqrt(A)
SRRM = SR.rolling(window=12).mean()
plt.plot(SR,c="blue")
plt.plot(SRRM,c="red")


# # Attempt 3(log-RM(log))

# In[73]:


LG = log(A)
LGRM = LG.rolling(window=12).mean()

diff_LG_RM = LG-LGRM
RM_diff_LG_RM = diff_LG_RM.rolling(window=12).mean()


# In[74]:


plt.plot(diff_LG_RM,c="brown")
plt.plot(RM_diff_LG_RM,c="blue")


# # Divide Data in training and testing set

# In[75]:


T = diff_LG_RM[diff_LG_RM.Production.isna()==False]


# In[76]:


T.shape[0]*0.8   #spliting 


# In[77]:


T = diff_LG_RM[diff_LG_RM.Production.isna()==False]


# # Divide data training and testing set

# In[78]:


trd = T[0:308]
tsd = T[308:]


# In[79]:


trd.shape


# In[80]:


tsd.shape


# In[81]:


trd


# In[82]:


tsd


# # Create an autoregression forecasting model

# In[83]:


from statsmodels.tsa.ar_model import AR
ar = AR(trd).fit()
pred = ar.predict(start="1994-04-01	",end="2018-01-01")


# In[84]:


pred              #it is not properly predicted values because is in the form of log-RM
                  #we will convert into its original format.


# # Conversion

# In[85]:


LGRM.shape


# In[86]:


pred.index


# In[87]:


LGRM_new = LGRM[LGRM.Production.isna()==False][308:]
Q1 = pd.DataFrame(pred)
Q1.columns =["Production"]


# In[88]:


from numpy import exp
pred_final = exp(LGRM_new + Q1)


# In[89]:


plt.plot(A)
plt.plot(pred_final)


# In[90]:


A.tail()


# In[91]:


from statsmodels.tsa.ar_model import AR
ar = AR(trd).fit()
pred1 = ar.predict(end="01/01/2022")


# In[ ]:





# # Prediction for 2019-2022

# In[92]:


T = LGRM[LGRM.Production.isna()==False]


# In[93]:


ar = AR(T).fit()
pred2 = ar.predict(end="01/01/2022")


# In[94]:


pred1.head(2)


# In[95]:


pred2.head(2)


# In[ ]:





# In[96]:


pred2.shape


# In[97]:


pred1 = pred1[1:]


# In[98]:


pred1.shape


# In[99]:


pred2.shape


# In[100]:


exp(pred1+pred2)


# In[ ]:





# In[ ]:




