#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as py
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import keras
from keras.layers import LayerNormalization
import tensorflow as tf


# In[54]:


df = pd.read_csv('datald2.csv')

#dataset
#ds = dataset.to_numpy()
ds = pd.read_csv('a1dataset.csv')


# In[55]:


dataset = df.values
dataset


# In[56]:


X = dataset[:,0:135]
Y = dataset[:,136]


# In[57]:


min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)


# In[58]:


X_scale


# In[59]:


X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.75)


# In[60]:


X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=1)


# In[61]:


print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


# In[69]:


from keras.models import Sequential
from keras.layers import Dense
model = Sequential([ Dense(32, activation='relu', input_shape=(135,)), Dense(32, activation='relu'), Dense(1, activation='sigmoid'), ])
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=64, epochs=100, validation_data=(X_val, Y_val))


# In[70]:


print(model.evaluate(X_train,Y_train)[1]*100,"%")
#100.0% accuracy 20k epochs
#96.43835425376892% accuracy 10k epochs
#91.78082346916199% accuracy 2k epochs
#92.05479621887207% accuracy 1k epochs
#84.93150472640991% accuracy 100 epochs
#52.054792642593384% accuracy 10 epochs


# In[71]:


import matplotlib.pyplot as plt
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[72]:


from keras.layers import Dropout
from keras import regularizers
reg = Sequential([
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(135,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
])
reg.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
hreg = reg.fit(X_train, Y_train,
          batch_size=64, epochs=100,
          validation_data=(X_val, Y_val))


# In[73]:


print(reg.evaluate(X_train,Y_train)[1]*100,"%")


# In[74]:


plt.plot(hreg.history['loss'])
plt.plot(hreg.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.show()

