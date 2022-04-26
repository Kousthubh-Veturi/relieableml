#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd


# In[85]:


#Reading the training data
data = pd.read_csv('Downloads/Data_for_Lie_Detection.csv')
data


# In[86]:


#Getting input and output
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]


# In[87]:


#Splitting into 80-20 percent for checking the training / testing accuracy

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) # 80% training and 20% test


# In[88]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[89]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[90]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[91]:


#Testing on the recorded data for test
test_data = pd.read_csv('Downloads/liedetect/Test_Data_for_Lie_Detection.csv')
test_data


# In[92]:


#Predict the response for test dataset
y_pred_test = clf.predict(test_data)


# In[93]:


y_pred_test


# In[84]:


# where 1 represents truth
# where 0 represents lie


# In[ ]:





# In[ ]:




