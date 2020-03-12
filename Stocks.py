#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


company_info = pd.read_csv('../input/securities.csv')
company_info.head()
company_info["Ticker symbol"].nunique()
company_info.loc[company_info.Security.str.startswith('Face') , :]
df =  pd.read_csv('../input/prices.csv', header=0)
df


# In[6]:


company_data = company_info.loc[(company_info["Security"] == 'Xerox Corp.') | (company_info["Security"] == 'Yahoo Inc.') | (company_info["Security"] == 'Adobe Systems Inc')
               | (company_info["Security"] == 'Adobe Systems Inc') 
              | (company_info["Security"] == 'Facebook') | (company_info["Security"] == 'Goldman Sachs Group') , ["Ticker symbol"] ]["Ticker symbol"] 


# In[11]:


def closing_data(code):
    global closing_stock
    plt.subplot(211)
    company_close = df[df['symbol']==code]
    company_close = company_close.close.values.astype('float32')
    company_close = company_close.reshape(-1, 1)
    closing_stock = company_close
    plt.xlabel('Time')
    plt.ylabel(code + " close stock prices")
    plt.plot(company_close , 'r')
    plt.show()


# In[12]:


for i in company_data:
    closing_data(i)


# In[13]:

stocks = closing_stock[: , 0]
stocks = stocks.reshape(len(stocks) , 1)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
stocks = scaler.fit_transform(stocks)


# In[14]:


import numpy as np 
train, test= np.split(stocks, [int(.8 *len(stocks))])
train = stocks[0:train]
test = stocks[len(train) : ]
train = train.reshape(len(train) , 1)
test = test.reshape(len(test) , 1)
print(train.shape , test.shape)


# In[15]:


def preprocessing(data , days=2):
    X, Y = [], []
    for i in range(len(data)-days-1):
        a = data[i:(i+days), 0]
        X.append(a)
        Y.append(data[i + days, 0])
    return np.array(X), np.array(Y)


# In[16]:


X_train, y_train = preprocessing(train, 2)
X_test, y_test = preprocessing(test, 2)


# In[17]:


X_train = X_train.reshape(X_train.shape[0] , 1 ,X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0] , 1 ,X_test.shape[1])


# In[19]:


import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense , BatchNormalization , Dropout , Activation
from keras.layers import LSTM , GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
days =2
model = Sequential()
model.add(GRU(256 , input_shape = (1 , days) , return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(256))
model.add(Dropout(0.4))
model.add(Dense(64 ,  activation = 'relu'))
model.add(Dense(1))
print(model.summary())


# In[20]:


optimizer = optimizers.Adam(lr=0.001)
from keras.callbacks import ReduceLROnPlateau
model.compile(loss='mean_squared_error', optimizer=optimizer , metrics = ['mean_squared_error'])
history = model.fit(X_train, y_train, epochs=100 , batch_size = 128 , 
           validation_data = (X_test,y_test))


# In[22]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[23]:


import math
def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]

model_score(model, X_train, y_train , X_test, y_test)


# In[24]:


pred = model.predict(X_test)
pred = scaler.inverse_transform(pred)
pred[:10]
y_test = y_test.reshape(y_test.shape[0] , 1)
y_test = scaler.inverse_transform(y_test)
y_test[:10]
print("Red - Predicted,  Blue - Actual")
plt.rcParams["figure.figsize"] = (15,7)
plt.plot(y_test , 'b')
plt.plot(pred , 'r')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.grid(True)
plt.show()


# In[ ]:


stocks.shape

from keras.models import load_model
 
model.save('lstm_model.h5')


# In[25]:


from sklearn.preprocessing import MinMaxScaler
def preprocess(closing_stocks):
    stocks_1 = closing_stocks[: , 0]
    stocks_1 = stocks_1.reshape(len(stocks) , 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    stocks_1 = scaler.fit_transform(stocks_1)
    x, y = arrange(stocks_1)
    x = x.reshape(x.shape[0] , 1 ,x.shape[1])
    return x,y


# In[26]:


def prediction(x,y):
    from matplotlib import pyplot as plt
    y_pred = model.predict(x)
    plt.rcParams["figure.figsize"] = (15,7)
    plt.figure()
    plt.plot(y , 'b')
    plt.plot(y_pred , 'r')
    plt.title('Prediction vs Real Stock Price')
    plt.ylabel('Price')
    plt.xlabel('Days')
    plt.legend(['Prediction', 'Real'], loc='upper left')
    plt.show()

    


# In[36]:


def stock_prediction(code):
    company_close = df[df['symbol']==code]
    company_close = company_close.close.values.astype('float32')
    company_close = company_close.reshape(-1, 1)
    closing_stock = company_close
    stocks = closing_stock[: , 0]
    stocks = stocks.reshape(len(stocks) , 1)
    stocks = scaler.fit_transform(stocks)
    X,Y = preprocessing(stocks,2)
    X = X.reshape(X.shape[0] , 1 ,X.shape[1])
    prediction(X,Y)


# In[37]:


for i in company_data:
    stock_prediction(i)


# In[ ]:




