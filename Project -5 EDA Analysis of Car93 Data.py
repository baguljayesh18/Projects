#!/usr/bin/env python
# coding: utf-8

# # Problem statement
# 
# 
#  EDA Analysis of Car93 Data

# In[1]:


from warnings import filterwarnings
filterwarnings("ignore")


# # Reading dataset

# In[2]:


import pandas as pd
A=pd.read_csv("C:/Users/Admin/Desktop/ETL class/machine learning/13-08-2022(Logistic regression)/Cars93.csv")



pd.set_option('display.max_columns', None) # maximize the display value of column
pd.set_option('display.max_rows', None)


# In[3]:


A.head()


# # Dropped Un-necessary columns

# In[4]:


len(A.id.unique())


# In[5]:


len(A.Model.unique())


# In[6]:


len(A.Make.unique())


# In[7]:


X= A.drop(labels=["id","Make","Model"],axis=1)


# In[8]:


X.head()


# # Missing Data Treatment

# In[9]:


X.isna().sum()


# In[10]:


from PM8wd import replacer
replacer(X)


# # Divide data in categorical and continuous

# In[11]:


cat=[]
con=[]
for i in X.columns:
    if(X[i].dtypes == "object"):
        cat.append(i)
    else:
        con.append(i)    


# In[12]:


cat


# In[13]:


con


# # EDA

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# # 1. Univariate Analysis

# # 1.1 Univariate Analysis of continous columns

# # Histogram

# In[15]:


plt.hist(X.Weight)
plt.title("Weight Vs No_of_Weight")
plt.xlabel("Weight")
plt.ylabel("No_of_Weight")
plt.show()


# # Distribution plot

# In[16]:


sb.distplot(X.Weight)


# In[17]:


X.Weight.mean()


# In[18]:


X.Weight.median()


# In[19]:


X.Weight.skew()    #Right skew


# # 1.2 Univariate of categorical columns

# # pie chart

# In[20]:


X.Type.value_counts()


# In[21]:


X.Type.value_counts().plot(kind="pie")                   #piechart


# # Bar plot

# In[22]:


X.Type.value_counts()


# In[23]:


X.Type.value_counts().plot(kind="bar")
plt.title("Car_Type_Bar",fontsize=15)
plt.xlabel("Car type",fontsize=12)
plt.ylabel("No of Cars",fontsize=12)
plt.show()


# In[24]:


X.Type.value_counts().plot(kind="barh")
plt.title("Car_Type_Bar",fontsize=15)
plt.ylabel("Car type",fontsize=12)
plt.xlabel("No of Cars",fontsize=12)
plt.show()


# # Multiple plots in one area

# In[25]:


plt.figure(figsize=(15,15))

plt.subplot(3,3,1)
sb.countplot(X.Manufacturer)

plt.subplot(3,3,2)
sb.countplot(X.Type)

plt.subplot(3,3,3)
sb.countplot(X.AirBags)

plt.subplot(3,3,4)
sb.countplot(X.DriveTrain)

plt.subplot(3,3,5)
sb.countplot(X.Cylinders)

plt.subplot(3,3,6)
sb.countplot(X["Man.trans.avail"])

plt.subplot(3,3,7)
sb.countplot(X.Origin)


# # 2.Bivariate Analysis

# # 2.1 (Y-con,X-cat)

# In[26]:


X.head()


# In[27]:


#Y=A[["Weight"]]


# In[28]:


import seaborn as sb
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))

plt.subplot(3,3,1)
sb.boxplot(X.Manufacturer,X.Weight)

plt.subplot(3,3,2)
sb.boxplot(X.AirBags,X.Weight)

plt.subplot(3,3,3)
sb.boxplot(X.DriveTrain,X.Weight)

plt.subplot(3,3,4)
sb.boxplot(X.Type,X.Weight)

plt.subplot(3,3,5)
sb.boxplot(X.Origin,X.Weight)


# # 2.2 (X-con,Y-con)

# In[29]:


X.corr()


# In[30]:


import seaborn as sb
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))

plt.subplot(3,3,1)
sb.scatterplot(X.Weight,X.Price)

plt.subplot(3,3,2)
sb.scatterplot(X.Weight,X.EngineSize)

plt.subplot(3,3,3)
sb.scatterplot(X.Weight,X.Horsepower)

plt.subplot(3,3,4)
sb.scatterplot(X.Weight,X.RPM)

plt.subplot(3,3,5)
sb.scatterplot(X.Weight,X.Revpermile)

plt.subplot(3,3,6)
sb.scatterplot(X.Weight,X.Width)

plt.subplot(3,3,7)
sb.scatterplot(X.Weight,X.Length)

plt.subplot(3,3,8)
sb.scatterplot(X.Weight,X.Wheelbase)

plt.subplot(3,3,9)
sb.scatterplot(X.Weight,X["Turn.circle"])


# In[ ]:




