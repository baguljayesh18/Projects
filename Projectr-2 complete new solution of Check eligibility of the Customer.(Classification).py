#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# A Company wants to automate the loan eligibility process based on customer details provided while filling online application form. The details filled by the customer are Gender, Marital Status, Education, Number of Dependents, Income of self and co applicant, Required Loan Amount, Required Loan Term, Credit History and others. The requirements are as follows:
# 
# 1)Check eligibility of the Customer given the inputs described above.(Classification)

# In[1]:


from warnings import filterwarnings     # import warnings to avoid warnings
filterwarnings("ignore")


# # Reading Dataset

# In[2]:


import pandas as pd
A=pd.read_csv("C:/Users/Admin/Desktop/ETL class/machine learning/Projects/Loan project -2/training_set.csv")




pd.set_option('display.max_columns', None) # maximize the display value of column
pd.set_option('display.max_rows', None)


# In[3]:


A.head(3)


# # Missing Data Treatment

# In[4]:


A.isna().sum()  #checking null values present in dataset.


# In[5]:


from PM8wd import replacer
replacer(A)


# In[6]:


A.isna().sum()


# # Declaration of X and Y variables

# In[7]:


Y = A[["Loan_Status"]]
X = A.drop(labels=["Loan_ID","Loan_Status"],axis=1)    


# # Divide data in categorical and continuous

# In[8]:


cat = []
con = []
for i in X.columns:
    if(X[i].dtypes == "object"):
        cat.append(i)
    else:
        con.append(i)


# In[9]:


cat


# In[10]:


con


# # EDA

# In[11]:


imp_cols = []
from PM8wd import chisq
for i in cat:
    j = chisq(A,"Loan_Status",i)   # here Y(cat) and X(cat),so will apply CHISQ/Countplot with Hue.
    print("Loan_Status vs",i)
    print("Pval: ",j) 


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sb

plt.figure(figsize=(20,15))

plt.subplot(2,3,1)
sb.countplot(A.Gender,hue=A.Loan_Status)

plt.subplot(2,3,2)
sb.countplot(A.Married,hue=A.Loan_Status)

plt.subplot(2,3,3)
sb.countplot(A.Dependents,hue=A.Loan_Status)

plt.subplot(2,3,4)
sb.countplot(A.Education,hue=A.Loan_Status)

plt.subplot(2,3,5)
sb.countplot(A.Self_Employed,hue=A.Loan_Status)

plt.subplot(2,3,6)
sb.countplot(A.Property_Area,hue=A.Loan_Status)


# In[13]:


imp_cols = []
from PM8wd import ANOVA
for i in con:                                  # here Y(cat) and X(con),so will apply ANOVA/boxplot
    b = ANOVA(A,"Loan_Status",i)
    print("Loan_Status vs",i)
    print("Pval: ",b)


# In[14]:


import seaborn as sb
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))

plt.subplot(3,3,1)
sb.boxplot(A.Loan_Status,A.ApplicantIncome)

plt.subplot(3,3,2)
sb.boxplot(A.Loan_Status,A.CoapplicantIncome)

plt.subplot(3,3,3)
sb.boxplot(A.Loan_Status,A.LoanAmount)

plt.subplot(3,3,4)
sb.boxplot(A.Loan_Status,A.Loan_Amount_Term)

plt.subplot(3,3,5)
sb.boxplot(A.Loan_Status,A.Credit_History)


# # Skew

# In[15]:


X.skew()


# # Standardize the data

# In[16]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X1 = pd.DataFrame(ss.fit_transform(X[con]),columns=con)
X2 = pd.get_dummies(X[cat])
Xnew = X1.join(X2)


# In[17]:


Xnew.head()


# # Finding outliers

# In[18]:


outliers = []
for i in con:
    outliers.extend(Xnew[(Xnew[i] > 3) | (Xnew[i] < -3)].index)    #  we find outliers grea than +3 and less than -3

Xnew = Xnew.drop(labels=outliers,axis=0)                           #here we dropped rows containing outliers.
Y = Y.drop(labels=outliers,axis=0)


# In[19]:


outliers


# In[20]:


Xnew.shape    # Rows and columns not containing outliers.


# In[21]:


Y.shape


# # Divide Data into training and testing split

# In[22]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)


# # Model

# In[23]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(xtrain,ytrain)


# In[24]:


pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)


# In[25]:


from sklearn.metrics import accuracy_score
print(round(accuracy_score(ytrain,pred_tr),3))
print(round(accuracy_score(ytest,pred_ts),3))


# Here model subjected overfitting so we will try tree model

# # DTC

# In[26]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy",random_state=21)
model = dtc.fit(xtrain,ytrain)
pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,pred_tr)
ts_acc = accuracy_score(ytest,pred_ts)


# In[27]:


tr_acc


# In[28]:


ts_acc


# Model subjected overfitting,so will make pruning

# 1.max_depth

# In[29]:


tr = []
ts = []
for i in range(2,30,1):
    
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,max_depth=i)
    model = dtc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)

    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    
    tr.append(tr_acc)
    ts.append(ts_acc)


# In[30]:


import matplotlib.pyplot as plt
plt.plot(tr)
plt.plot(ts)


# 2.min_samples_leaf

# In[31]:


tr = []
ts = []
for i in range(2,25,1):
    
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,min_samples_leaf=i)
    model = dtc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)

    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    
    tr.append(tr_acc)
    ts.append(ts_acc)


# In[32]:


import matplotlib.pyplot as plt
plt.plot(tr)
plt.plot(ts)


# 3.min_samples_split

# In[33]:


tr = []
ts = []
for i in range(2,105,1):
    
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,min_samples_split=i)
    model = dtc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)

    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    
    tr.append(tr_acc)
    ts.append(ts_acc)


# In[34]:


import matplotlib.pyplot as plt
plt.plot(tr)
plt.plot(ts)


# In[ ]:





# # Adaboost classifier

# In[35]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,max_depth=2)
abc = AdaBoostClassifier(dtc,n_estimators=50)
model = abc.fit(xtrain,ytrain)
pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,pred_tr)
ts_acc = accuracy_score(ytest,pred_ts)


# In[36]:


tr_acc


# In[37]:


ts_acc


# so model subjected overfitting

# 1.max_depth

# In[38]:


tr = []
ts = []
for i in range(2,25,1):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,max_depth=i)
    abc = AdaBoostClassifier(dtc,n_estimators=50)
    model = abc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)


# In[39]:


import matplotlib.pyplot as plt
plt.plot(range(2,25,1),tr,c="blue")
plt.plot(range(2,25,1),ts,c="red")


# 2.min_samples_leaf

# In[40]:


tr = []
ts = []
for i in range(2,25,1):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,min_samples_leaf=i)
    abc = AdaBoostClassifier(dtc,n_estimators=50)
    model = abc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)


# In[41]:


import matplotlib.pyplot as plt
plt.plot(range(2,25,1),tr,c="blue")
plt.plot(range(2,25,1),ts,c="red")


# 3.min_samples_split

# In[42]:


tr = []
ts = []
for i in range(2,25,1):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion="entropy",random_state=21,min_samples_split=i)
    abc = AdaBoostClassifier(dtc,n_estimators=50)
    model = abc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)


# In[43]:


import matplotlib.pyplot as plt
plt.plot(range(2,25,1),tr,c="blue")
plt.plot(range(2,25,1),ts,c="red")


# # Random Forest 

# In[44]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=21,max_depth=2)
model = rfc.fit(xtrain,ytrain)
pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = round(accuracy_score(ytrain,pred_tr),3)
ts_acc = round(accuracy_score(ytest,pred_ts),3)


# In[45]:


tr_acc


# In[46]:


ts_acc


# 1.max_depth

# In[47]:


tr = []
ts = []
for i in range(2,20,1):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=30,criterion="entropy",random_state=21,max_depth=i)
    model = rfc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)


# In[48]:


import matplotlib.pyplot as plt
plt.plot(range(2,20,1),tr,c="blue")
plt.plot(range(2,20,1),ts,c="red")


# 2.min_samples_leaf

# In[49]:


tr = []
ts = []
for i in range(2,10,1):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=30,criterion="entropy",random_state=21,min_samples_leaf=i)
    model = rfc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)


# In[50]:


import matplotlib.pyplot as plt
plt.plot(range(2,10,1),tr,c="blue")
plt.plot(range(2,10,1),ts,c="red")


# 3.min_samples_split
# 

# In[51]:


tr = []
ts = []
for i in range(2,30,1):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=30,criterion="entropy",random_state=21,min_samples_split=i)
    model = rfc.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,pred_tr)
    ts_acc = accuracy_score(ytest,pred_ts)
    tr.append(tr_acc)
    ts.append(ts_acc)


# In[52]:


import matplotlib.pyplot as plt
plt.plot(range(2,30,1),tr,c="blue")
plt.plot(range(2,30,1),ts,c="red")


# In[ ]:





# # Prediction on test data

# In[53]:


import pandas as pd

B=pd.read_csv("C:/Users/Admin/Desktop/ETL class/machine learning/Projects/Loan project -2/testing_set.csv")


# In[54]:


B


# # missing data treatment

# In[55]:


B.isna().sum()


# In[56]:


from PM8wd import replacer
replacer(B)


# In[57]:


X = B.drop(labels=["Loan_ID"],axis=1)


# # Divide data in categorical and continuous

# In[58]:


cat = []
con = []
for i in X.columns:
    if(X[i].dtypes == "object"):
        cat.append(i)
    else:
        con.append(i)


# # Standardize the data

# In[59]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X1 = pd.DataFrame(ss.fit_transform(X[con]),columns=con)
X2 = pd.get_dummies(X[cat])
Xnew = X1.join(X2)


# In[60]:


Xnew.head(5)


# # Finding outliers

# In[61]:


outliers = []
for i in con:
    outliers.extend(Xnew[(Xnew[i] > 3) | (Xnew[i] < -3)].index)    #  we find outliers grea than +3 and less than -3

Xnew = Xnew.drop(labels=outliers,axis=0)                           #here we dropped rows containing outliers.


# In[62]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(xtrain,ytrain)


# In[63]:


pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)


# In[64]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytrain,pred_tr))
print(confusion_matrix(ytest,pred_ts))


# In[65]:


pred_ts


# In[66]:


pred_tr


# In[67]:


W1 =pd.DataFrame(pred_tr)


# In[68]:


W1.columns = ["Predicted_Loan_eligibility_status"]


# In[69]:


W1


# In[70]:


W1.to_csv("C:/Users/Admin/Desktop/ETL class/machine learning/Projects/Loan project -2/Solution/Self solution/new load prediction.csv")


# In[ ]:




