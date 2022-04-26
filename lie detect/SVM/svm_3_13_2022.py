#!/usr/bin/env python
# coding: utf-8

# In[218]:


from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
#dataset = datasets.fetch_olivetti_faces()
dataset = pd.read_csv('datald2.csv')
#dataset
#ds = dataset.to_numpy()
ds = pd.read_csv('a1dataset.csv')
ds


# In[219]:


dataset


# In[220]:


X_train, X_test, y_train, y_test = train_test_split(ds, dataset.is_Truth, test_size=0.3,random_state=512)


# In[221]:


clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[222]:


print("accuracy: ",metrics.accuracy_score(y_test, y_pred)*100,"%")
print("precision:",metrics.precision_score(y_test, y_pred)*100,"%")
print("recall:",metrics.recall_score(y_test, y_pred)*100,"%")
#accuracy:  99.86559139784946 %
#precision: 99.74619289340102 %
#recall: 100.0 %

