#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings("ignore")

# 导入文件依据运行的环境和平台进行必要的更改
data=pd.read_csv("TikTok-data.csv",index_col=0)

# Rename
data.rename(columns={'1':'Length', '2':'Move_E', '3':'Move_D', '4':'Frame_E', '5':'Frame_D', '6':'Energe_E', '7':'Energe_D', '8':'ZCR_E', '9':'ZCR_D', '10':'Centroid_E', '11':'Centroid_D', '12':'Rolloff_E', '13':'Rolloff_D', '14':'Flux_E', '15':'Flux_D', '16':'BasFreq_E', '17':'BasFreq_D', '4124':'Edge_E', '4125':'Edge_D', 'labels':'Label'}, inplace=True)

# 特征名和标签名
col_name = data.columns[:-2]
label_name = data.columns[-1]

print ('训练集的标签:{}\n'.format(label_name))
print ('训练集的特征:{}\n'.format(col_name))
print ('训练集的形状:{}\n'.format(data.shape))


# In[2]:


# 打印data的前五行数据
data.head()


# In[3]:


# Label分布的直方图
sns.distplot(data['Label'], kde=False)


# In[4]:


# 描述数据中特征的分布
data.describe()


# In[5]:


# 时长Length分布和统计
data.drop(data[data['Length'] > 10000].index.tolist(), inplace=True)

fig, axes = plt.subplots(1, 2)
sns.barplot(x='Label', y='Length', data=data, ax=axes[0])
sns.stripplot(x='Label', y='Length', data=data, ax=axes[1], jitter=True)
plt.show()

facet = sns.FacetGrid(data[['Length', 'Label']], hue='Label', aspect=2)
facet.map(sns.kdeplot, "Length", shade=True)
facet.set(xlim=(0, 500))
facet.add_legend()
facet.set_axis_labels("Length", "Density")


# In[6]:


# 缺失值
data.isnull().any()


# In[7]:


# 填充缺失值
data = data.fillna(data.mean())


# In[8]:


# 重复值
data.drop_duplicates(inplace=True)
data.shape


# In[9]:


# Label -1 -> 0
data['Label'] = data['Label'].apply(lambda x:0 if x == -1 else x)
data['Label'].hist()


# # 特征工程
# - 特征过滤
# - 特征生成

# In[10]:


# 分离特征和标签
X = data.drop(['Label'], axis=1)
Y = data['Label']

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.75)

xtrain.shape


# ## 特征选择By随机森林
# - 使用随机森林进行特征选择
# - 训练集拟合随机森林模型
# - 用于获得feature_importances_

# In[11]:


from sklearn.ensemble import RandomForestClassifier
rfcModel = RandomForestClassifier()
rfcModel.fit(xtrain, ytrain)


# ## 特征重要性排序
# - 通过重要性值进行排序画出柱状图
# - 通过计算前缀和画出阶梯图

# In[12]:


# 将特征的重要性程度进行排序
N_most_important = 25

imp = np.argsort(rfcModel.feature_importances_)[::-1]
imp_slct = imp[:N_most_important]

FeaturesImportances = zip(col_name, map(lambda x:round(x,5), rfcModel.feature_importances_))
FeatureRank = pd.DataFrame(columns=['Feature', 'Imp'], data=sorted(FeaturesImportances, key=lambda x:x[1], reverse=True)[:N_most_important])


# In[13]:


# 重新选择X
xtrain_slct = xtrain.iloc[:,imp_slct]
xtest_slct  = xtest.iloc[:,imp_slct]


# In[14]:


# 特征排序图
ax1 = fig.add_subplot(111)
ax1 = sns.barplot(x='Feature', y='Imp', data=FeatureRank)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

SumImp = FeatureRank
for i in SumImp.index:
    if (i==0):
        SumImp['Imp'][i] = FeatureRank['Imp'][i]
    else:
        SumImp['Imp'][i] = SumImp['Imp'][i-1] + FeatureRank['Imp'][i]
ax2 = ax1.twinx()
plt.step(x=SumImp['Feature'], y=SumImp['Imp'])


# ## PCA
# 
# > 使用PCA进行特征生成，即与选择出的主成分与原数据合并，能够一定程度上提高预测精准度

# In[15]:


from sklearn.decomposition import PCA
pca = PCA(n_components=N_most_important)
pca.fit(xtrain)
pca.explained_variance_ratio_


# In[16]:


pca1 = PCA(6)
pc = pd.DataFrame(pca1.fit_transform(xtrain))
pc.index = xtrain.index
xtrain_pca = xtrain_slct.join(pc)


# In[17]:


pc = pd.DataFrame(pca1.fit_transform(xtest))
pd.index = xtrain.index
xtest_pca = xtes_slctt.join(pc)


# # 模型评估
# - 评价训练集表现
# - 评价测试集表现
# - 随机猜测函数对比

# In[21]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score

# AUC和混淆矩阵评估
ytrain_pred_clf = rf.predict_proba(xtrain)
ytrain_pred = rf.predict(xtrain)
ytest_pred_clf = rf.predict_proba(xtest)
ytest_pred = rf.predict(xtest)

# 评估训练集效果，直观判断是否过拟合
print ('分类模型训练集表现：')
print ('ml train model auc score {:.6f}'.format(roc_auc_score(ytrain, ytrain_pred_clf[:,1])))
print ('------------------------------')
print ('ml train model accuracy score {:.6f}'.format(accuracy_score(ytrain, ytrain_pred)))
print ('------------------------------')
threshold = 0.5
print (confusion_matrix(ytrain, (ytrain_pred_clf>threshold)[:,1]))

# 评估测试集效果
print ('分类模型测试集表现：')
print ('ml model auc score {:.6f}'.format(roc_auc_score(ytest, ytest_pred_clf[:,1])))
print ('------------------------------')
print ('ml model accuracy score {:.6f}'.format(accuracy_score(ytest, ytest_pred)))
print ('------------------------------')
threshold = 0.5
print (confusion_matrix(ytest, (ytest_pred_clf>threshold)[:,1]))

# 随机猜测函数对比
ytest_random_clf = np.random.uniform(low=0.0, high=1.0, size=len(ytest))
print ('random model auc score {:.6f}'.format(roc_auc_score(ytest, ytest_random_clf)))
print ('------------------------------')
print (confusion_matrix(ytest, (ytest_random_clf<=threshold).astype('int')))


# > roc_auc  score 0.957044
# > accuracy score 0.897004

# In[22]:


from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(ytest,ytest_pred_clf[:,1])
roc_auc = auc(fpr,tpr)


# In[23]:


## TODO:假阳性率为横坐标，真阳性率为纵坐标做曲线
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, 
         label='ROC curve (area = %0.2f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

