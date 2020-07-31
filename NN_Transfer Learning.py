#!/usr/bin/env python
# coding: utf-8

# # 用轉移學習的方式做NN圖形辨識模型

# ### NN 標準神經網路
# 
# 先做一次一般的0-9手寫數字判讀模型，再用轉移學習的方式做一個只有0或1的手寫數字判讀模型。

# In[1]:


# 初始準備
get_ipython().run_line_magic('env', 'KERAS_BACKEND=tensorflow')


# In[2]:


# KERAS function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# KERAS dataset - modified NIST
from keras.datasets import mnist

# KERAS utils function
from keras.utils import np_utils


# In[3]:


# read in train, test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# check the shape
print('x_train:',x_train.shape)
print('x_test:',x_test.shape)
print('y_train:',y_train.shape)
print('y_test:',y_test.shape)


# In[4]:


# reshape x_train, x_test
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

# seperate 0 and 1 data
x_train_01 = x_train[y_train <= 1]
x_test_01 = x_test[y_test <= 1]


# transfer y_train, y_test to one-hot encoding
y_train_10 = np_utils.to_categorical(y_train, 10)
y_test_10 = np_utils.to_categorical(y_test, 10)

# transfer y_train_01, y_test_01 into one-hot encoding
y_train_01 = y_train[y_train <= 1]
y_train_01 = np_utils.to_categorical(y_train_01, 2)
y_test_01 = y_test[y_test <= 1]
y_test_01 = np_utils.to_categorical(y_test_01, 2)


# In[5]:


print('y_train_10:',x_train.shape)
print('y_test_10:',x_test.shape)
print('y_train_01:',y_train_01.shape)
print('y_test_01:',y_test_01.shape)


# In[6]:


# build layers 
all_except_last = [Dense(500, input_dim=(784)), 
                   Activation('sigmoid'),
                   Dense(500, Activation('sigmoid'))]

output_layer = [Dense(10, Activation('softmax'))]

# assemble all layers
model_0_to_9 = Sequential(all_except_last + output_layer)
model_0_to_9.summary()


# In[7]:


# compile 
model_0_to_9.compile(loss='mse',
                     optimizer = SGD(lr=0.1),
                     metrics=['accuracy'])


# In[8]:


# fit
model_0_to_9.fit(x_train, y_train_10, batch_size=100, epochs=10)


# In[9]:


# check accuracy
model_0_to_9.evaluate(x_test,y_test_10)


# In[10]:


# new_output_layer for 0,1 
new_output_layer = [Dense(2), Activation('softmax')]

# assemble new model
model_0_to_1 = Sequential(all_except_last + new_output_layer)
model_0_to_1.summary()


# In[11]:


# training models except first & second layer

for layer in all_except_last:
    layer.trainable = False
    
# check the amount of trainable and non-trainable
model_0_to_1.summary()


# In[12]:


# compile
model_0_to_1.compile(loss='mse',
                    optimizer=SGD(lr=0.1),
                    metrics=['accuracy'])


# In[13]:


# as data amount decrease
# epochs should decrease as well to avoid over-fitting
model_0_to_1.fit(x_train_01, y_train_01, batch_size=100, epochs=5)


# In[14]:


# check score
model_0_to_1.evaluate(x_test_01, y_test_01)


# 首先利用MNIST資料庫與NN標準神經網路，利用list的方式寫出各層layer，再利用Sequential函式建構0-9的手寫數字辨識模型，再來訓練與評分可以發現準確度有91%，若增加訓練次數可以再近一步改善準確率。
# 
# 再來，我們想將這個神經網路的概念移植到0、1兩種數字的手寫判讀模型，首先抽取出0、1的資料，更改產出層(new_output_layer)將Dense層改成只有2種output，再來訓練與組裝都與之前相同，但要注意因為資料量減少所以訓練次數epochs不要太多，避免over-fitting。

# In[15]:


# save model
model_0_to_1.json = model_0_to_1.to_json()
open('handwriting_model_nn_transferlearning.json','w').write(model_0_to_1.json)

# save weights
model_0_to_1.save_weights('handwriting_weights_nn_transferlearning.h5')

