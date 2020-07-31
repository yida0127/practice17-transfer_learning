#!/usr/bin/env python
# coding: utf-8

# # 用CNN手寫辨識練習轉移學習

# In[1]:


get_ipython().run_line_magic('env', 'KERAS_BACKEND=tensorflow')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# KERAS function
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# mnist
from keras.datasets import mnist

# KERAS utils 
from keras.utils import np_utils


# In[4]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[5]:


print('There are %d train data of size %d x %d' %x_train.shape)
print('There are %d test data of size is %d x %d' %x_test.shape)


# In[6]:


# add one more dimension for CNN tunnel
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


# In[7]:


# checking
x_train.shape, x_test.shape


# In[8]:


# seperate x data which answer is 0 & 1
x_train_01 = x_train[y_train <= 1]
x_test_01 = x_test[y_test <= 1]


# In[9]:


# one-hot encoding
y_train_10 = np_utils.to_categorical(y_train, 10)
y_test_10 = np_utils.to_categorical(y_test, 10)

y_train_01 = y_train[y_train <= 1]
y_train_01 = np_utils.to_categorical(y_train_01, 2)

y_test_01 = y_test[y_test <= 1]
y_test_01 = np_utils.to_categorical(y_test_01, 2)


# In[10]:


x_train_01.shape, x_test_01.shape


# In[11]:


y_train_01.shape, y_test_01.shape


# In[12]:


# Convolutional layer
conv_layer = [Conv2D(32,(3,3), padding='same', input_shape=(28,28,1)),
             Activation('sigmoid'),
             MaxPooling2D(pool_size=(2,2)),
             
             Conv2D(64,(3,3), padding='same'),
             Activation('sigmoid'),
             MaxPooling2D(pool_size=(2,2)),
             
             Conv2D(128,(3,3), padding='same'),
             Activation('sigmoid'),
             MaxPooling2D(pool_size=(2,2))]

# Fully connected layer
fc_layer = [Flatten(),
           Dense(200),
           Activation('sigmoid'),
           Dense(10),
           Activation('softmax')]

model_0_to_9 = Sequential(conv_layer + fc_layer)
model_0_to_9.summary()


# In[13]:


# load weight to check if construction is correct
model_0_to_9.load_weights('handwriting_weights_cnn.h5')


# In[14]:


# Revise fully connected layer to distinguish 0 & 1 only
new_fc_layer = [Flatten(),
               Dense(200),
               Activation('sigmoid'),
               Dense(2),
               Activation('softmax')]

model_0_to_1 = Sequential(conv_layer + new_fc_layer)
model_0_to_1.summary()


# In[15]:


for layer in conv_layer:
    layer.trainable = False


# In[16]:


# check the non-trainable params again
model_0_to_1.summary()


# In[17]:


# compile the model
model_0_to_1.compile(loss='mse',
                    optimizer=SGD(lr=0.1),
                    metrics = ['accuracy'])


# In[18]:


# train the model
model_0_to_1.fit(x_train_01, y_train_01, batch_size=100, epochs=12)


# In[19]:


model_0_to_1.evaluate(x_test_01, y_test_01)


# In[20]:


from ipywidgets import interact_manual


# In[21]:


predict = model_0_to_1.predict_classes(x_test_01)


# In[22]:


def test(測試編號):
    plt.imshow(x_test_01[測試編號].reshape(28,28), cmap='Greys')
    print('神經網路判斷為:',predict[測試編號])


# In[23]:


interact_manual(test, 測試編號=(0,2115))


# In[24]:


# save model and weight
model_0_to_1.json = model_0_to_1.to_json()
open('CNN_handwriting_model_transferlearning.json','w').write(model_0_to_1.json)

model_0_to_1.save_weights('CNN_handwriting_weights_transferlearning.h5')

