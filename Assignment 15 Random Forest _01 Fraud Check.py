#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# In[2]:


data = pd.read_csv("Fraud_check.csv")
data.head()


# In[3]:


data.describe()


# In[4]:


data[data.duplicated()].shape


# In[5]:


data = data.rename({'Taxable.Income': 'TaxableIncome'}, axis=1)


# In[6]:


data['TaxableIncome'] = pd.cut(data.TaxableIncome,bins=(0,30000,199778),labels=['Risky','Good'])
data


# In[7]:


plt.figure(figsize=(20 , 20 ))
sns.pairplot(data)


# In[8]:


x = data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[9]:


label_encoder_x=LabelEncoder()


# In[10]:


x=x.apply(LabelEncoder().fit_transform)


# In[11]:


rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=100,criterion="entropy")
rf


# In[12]:


rf.fit(x,y)


# In[13]:


rf.estimators_


# In[14]:


data['rf_pred'] = rf.predict(x)


# In[15]:


cols = ['rf_pred','TaxableIncome']
cols


# In[16]:


data[cols].head()


# In[17]:


data['TaxableIncome']


# In[18]:


confusion_matrix(data['TaxableIncome'],data['rf_pred'])


# In[19]:


pd.crosstab(data['TaxableIncome'],data['rf_pred'])

