#!/usr/bin/env python
# coding: utf-8

# # Problem statement
# 
# To predict Breast Cancer whether the cancer is benign or malignant Using Machine Learning Algorithms.

# In[1]:


from warnings import filterwarnings
filterwarnings("ignore")


# # Reading dataset

# In[4]:


import pandas as pd
A=pd.read_csv("C:/Users/Admin/Desktop/ETL class/Project/Breast cancer/breast cancer.csv")


pd.set_option('display.max_columns', None) # maximize the display value of column
pd.set_option('display.max_rows', None)


# In[5]:


A.head()


# # Missing Data Treatment

# In[6]:


from PM8wd import replacer
replacer(A)


# # Define X and Y

# In[7]:


Y=A[["diagnosis"]]     
X=A.drop(labels=["diagnosis","id"],axis=1)


# # Divide data in categorical and continuous

# In[8]:


cat=[]
con=[]
for i in X.columns:
    if (X[i].dtypes=="object"):
        cat.append(i)
    else:
        con.append(i)


# In[9]:


cat   #In our dataset cat column not present so it is empty.


# In[10]:


con


# # EDA

# In[11]:


A.corr()


# In[12]:


import seaborn as sb
import matplotlib.pyplot as plt                              #  here Y(cat) and X(con),so we appied boxplot.
plt.figure(figsize=(20,20))

plt.subplot(3,3,1)
sb.boxplot(A.diagnosis,A.radius_mean)

plt.subplot(3,3,2)
sb.boxplot(A.diagnosis,A.texture_mean)

plt.subplot(3,3,3)
sb.boxplot(A.diagnosis,A.perimeter_mean)

plt.subplot(3,3,4)
sb.boxplot(A.diagnosis,A.area_mean)

plt.subplot(3,3,5)
sb.boxplot(A.diagnosis,A.smoothness_mean)

plt.subplot(3,3,6)
sb.boxplot(A.diagnosis,A.compactness_mean)

plt.subplot(3,3,7)
sb.boxplot(A.diagnosis,A.concavity_mean)


# # Standardize the data

# In[13]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X1 = pd.DataFrame(ss.fit_transform(X[con]),columns=con)
#X2 = pd.get_dummies(X[cat])
Xnew = X1


# # Finding outliers

# In[14]:


outliers = []
for i in con:
    outliers.extend(Xnew[(Xnew[i] > 3) | (Xnew[i] < -3)].index)    #  we find outliers grea than +3 and less than -3

Xnew = Xnew.drop(labels=outliers,axis=0)                           #here we dropped rows containing outliers.
Y = Y.drop(labels=outliers,axis=0)


# In[15]:


outliers


# In[16]:


Xnew.shape


# In[17]:


Y.shape


# # Divide data training and testing set

# In[18]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)


# # Model

# In[19]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
model=dtc.fit(xtrain,ytrain)
pred=model.predict(xtest)


# In[20]:


pred   #predicted values


# In[21]:


ytest


# In[22]:


Q=ytest[["diagnosis"]]
Q["Predicted_diagnosis"]=pred    #here we adding one column--->"Predicted_diagnosis" and name will be give from pred.  


# In[23]:


Q


# # Here we are finding accuracy score and confusion matrix.

# In[24]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest,pred)


# In[25]:


import seaborn as sb
import matplotlib.pyplot as plt


# In[26]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,pred) 
sb.heatmap(cm,annot=True)
plt.show()


#   60 and 30 is true values                         
#   6 and 3 is false values  
#   
#   30---->B(benign) predicted as B(benign)                    
#   60---->M(malignant) predicted as M(malignant)                  
#   
#   3---->B(benign) predicted as M(malignant)               
#   6---->M(malignant) predicted as B(benign)                        

# # Fining accuracy score using RandomForestClassifier

# In[27]:


from sklearn.ensemble  import RandomForestClassifier
rfr = RandomForestClassifier(n_estimators=30,criterion="entropy",random_state=21,max_depth=3)
model = rfr.fit(xtrain,ytrain)
pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = round(accuracy_score(ytrain,pred_tr),2)
ts_acc = round(accuracy_score(ytest,pred_ts),2)


# In[28]:


tr_acc


# In[29]:


ts_acc


# In[30]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,pred_ts) 
sb.heatmap(cm,annot=True)
plt.show()


# # Fining accuracy score using KNN

# In[31]:


from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=5)
model = knc.fit(xtrain,ytrain)
pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = round(accuracy_score(ytrain,pred_tr),2)
ts_acc = round(accuracy_score(ytest,pred_ts),2)


# In[32]:


tr_acc


# In[33]:


ts_acc


# In[34]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,pred_ts) 
sb.heatmap(cm,annot=True)
plt.show()


# After training all the algorithms , we found that KNN, Decision Tree Classification , Random Forest Classification Model have high accuracy. From them we choose the KNN Model as it gives the highest accuracy.

# In[ ]:




