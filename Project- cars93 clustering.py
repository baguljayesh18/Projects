#!/usr/bin/env python
# coding: utf-8

# # Problem statement
# Search pattern of similar car recommendation

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


# # Which columns should be used for clustering

# This will be based on your domain knowledge and customer perception. Example: In India general buying is based on Price | MPG.city

# In[4]:


B = A[["Price","MPG.city"]]


# # Standardize

# In[5]:


from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler() # std *(max-min) + min
Q = pd.DataFrame(mm.fit_transform(B))
Q.columns = ["Price","MPG.city"]


# In[6]:


Q.head(3)


# # deciding the requried no. clusters

# In[7]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters=4)
model = km.fit(B)


# In[8]:


model.labels_


# In[9]:


B['Cluster_no']=model.labels_


# In[10]:


B['Make']=A.Make


# In[11]:


B.sort_values(by="Cluster_no").head(3)


# # Find similar cars for a given car

# In[12]:


w = input("Enter the car Make: ")
cluster_no_on_input_car = B[B.Make==w].Cluster_no.values[0]
similar_cars = list(B[(B['Cluster_no'] == cluster_no_on_input_car) & (B.Make != w)].Make.values)


# In[17]:


similar_cars


# In[18]:


k = range(1,15,1)
WCSS = []
for i in k:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=i)
    model = km.fit(Q)
    WCSS.append(model.inertia_)


# In[19]:


WCSS


# In[20]:


import matplotlib.pyplot as plt
plt.scatter(k,WCSS,c="red")
plt.plot(k,WCSS,c="blue")
plt.xticks(range(1,15,1))
plt.xlabel("no of clusters")
plt.ylabel("WCSS")
plt.title("Elbow curve")

