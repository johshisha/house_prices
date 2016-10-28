
# coding: utf-8

# In[1]:

#import liblaries
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib

import matplotlib.pyplot as plt

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


# In[89]:

from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(train_data, train_target)


# In[90]:

lr_predicted = lr.predict(test_data)
lr_predicted = pd.DataFrame({"Id":test_data.Id, "SalePrice":lr_predicted})
lr_predicted.head()


# In[95]:

lr_predicted.to_csv("LR_solution.csv", index=False)   # write to csv


# ## おまけ

# ### RandomForestRegressor

# In[92]:

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1)
rfr.fit(train_data, train_target)


# In[93]:

rfr_predicted = rfr.predict(test_data)
rfr_predicted = pd.DataFrame({"Id":test_data.Id, "SalePrice":rfr_predicted})
rfr_predicted.head()


# In[94]:

rfr_predicted.to_csv("RFR_solution.csv", index=False)   # write to csv


# ### 教師データの分布

# In[83]:

#matplotlib.rcParams['figure.figsize'] = (6.0, 4.0)

prices = pd.DataFrame({"price":train["SalePrice"]})
ax= prices.plot(bins=50, alpha=0.5, figsize=(10,6), kind='hist')

# ### 予測したデータの分布

# In[100]:

result = pd.DataFrame({"LR estimated":lr_predicted["SalePrice"], "RFR estimated":rfr_predicted["SalePrice"]})
ax= result.plot(bins=50, alpha=0.3, figsize=(10,6), kind='hist')

# In[107]:

# Output feature importance coefficients, map them to their feature name, and sort values
coef = pd.Series(rfr.feature_importances_, index = train_data.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
coef.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()

# ### 変数2個

# In[113]:

lr_removed = linear_model.LinearRegression()
lr_removed.fit(train_data[['OverallQual', 'GrLivArea']], train_target)


# In[115]:

lr_removed_predicted = lr_removed.predict(test_data[['OverallQual', 'GrLivArea']])
lr_removed_predicted = pd.DataFrame({"Id":test_data.Id, "SalePrice":lr_removed_predicted})
lr_removed_predicted.head()


# In[123]:

lr_removed_predicted.to_csv("LR_R_solution.csv", index=False)   # write to csv


# In[122]:

result = pd.DataFrame({"LR_R estimated":lr_removed_predicted["SalePrice"]})
ax= result.plot(bins=50, alpha=0.3, figsize=(10,6), kind='hist')


# In[ ]:
#plt.show()


