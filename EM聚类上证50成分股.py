#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data=pd.read_csv("data/train_data_for_50.csv",encoding="utf-8")

print(data.head())
print(data.shape)
#print(data.describe())
#print(data.info())

train_data=data.drop(columns=['代码','名称'],inplace=False)
# 设置plt正确显示中文
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# 用热力图呈现特征字段之间的相关性
corr = train_data.corr()
plt.figure(figsize=(10,10))
# annot=True显示每个方格的数据
a=sns.heatmap(corr,annot=True)
a.set_ylim([14,0])#为了让第一行和最后一行都显示出来
# 规范化
train_data=StandardScaler().fit_transform(train_data)
gm = GaussianMixture(n_components=10)
gm.fit(train_data)
labels=gm.predict(train_data)
result=data[['名称']]
result['target']=labels
print(result)
from sklearn.metrics import calinski_harabaz_score
print(calinski_harabaz_score(train_data, labels))


# In[ ]:




