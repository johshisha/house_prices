
# coding: utf-8

# In[1]:

#import liblaries
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib

#import matplotlib.pyplot as plt

#get_ipython().magic("config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook")
#get_ipython().magic('matplotlib inline')


# ## データの読み込み

# In[41]:

train = pd.read_csv('train.csv')
print(train.shape)
train.head()


# In[44]:

test = pd.read_csv('test.csv')
print(test.shape)
test.head()


# ## テキストを数値に変換

# In[80]:

all_data =  pd.concat((train.drop('SalePrice', axis=1), test), axis=0)
print(all_data.shape)
all_data = pd.get_dummies(all_data)    # カテゴリカルデータを数値に変換する
all_data = all_data.fillna(all_data.mean())    # NaNは行の平均値で代替
print(all_data.shape)
all_data.head()


# ## 線形回帰で学習＆予測

# In[61]:

train_target = train['SalePrice']
train_data = all_data[:train.shape[0]]
test_data = all_data[train.shape[0]:]


# In[62]:

from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(train_data, train_target)


# In[81]:

predicted = lr.predict(test_data)
predicted = pd.DataFrame({"Id":test_data.Id, "SalePrice":predicted})
predicted.head()


# In[79]:

predicted.to_csv("satoshi-sanjo_solution.csv", index=False)   # write to csv


# ## おまけ

# ### 教師データの分布

# In[83]:

#matplotlib.rcParams['figure.figsize'] = (6.0, 4.0)

#prices = pd.DataFrame({"price":train["SalePrice"]})
#ax= prices.plot(bins=50, alpha=0.5, figsize=(10,6), kind='hist')


# In[ ]:



