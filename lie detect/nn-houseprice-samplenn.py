#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as py
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import keras
from keras.layers import LayerNormalization
import tensorflow as tf


# In[3]:


df = pd.read_csv('housepricedata.csv')
df


# In[4]:


dataset = df.values
dataset


# In[5]:


X = dataset[:,0:10]
Y = dataset[:,10]


# In[6]:


min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)


# In[7]:


X_scale


# In[8]:


X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.75)


# In[9]:


X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=1)


# In[10]:


print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


# In[27]:


from keras.models import Sequential
from keras.layers import Dense
model = Sequential([ Dense(32, activation='relu', input_shape=(10,)), Dense(32, activation='relu'), Dense(1, activation='sigmoid'), ])
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=32, epochs=2000, validation_data=(X_val, Y_val))


# In[28]:


print(model.evaluate(X_train,Y_train)[1]*100,"%")
#96.43835425376892% accuracy 10k epochs
#90.13698697090149% accuracy 2k epochs
#89.31506872177124% accuracy 1k epochs
#84.93150472640991% accuracy 100 epochs
#52.054792642593384% accuracy 10 epochs


# In[ ]:




