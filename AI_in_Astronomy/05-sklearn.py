#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:02:52 2025

@author: wang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy import stats

from sklearn import datasets 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV               #调参
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

#字体-宋体显示
import matplotlib as mpl 
from matplotlib.font_manager import fontManager
fontManager.addfont("/Users/wang/code/AI-astro/homework/SIMSUN.TTC")
mpl.rc('font', family="SimSun")

#%% Step1 data_iris

iris = datasets.load_iris() 
data = iris['data']
feature_names = ["sepal_length","sepal_width","petal_length","petal_width"]
  #iris.feature_names = ['sepal length (cm)','sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

X = pd.DataFrame(data,columns=feature_names)
y = iris['target']

#%%  data_wine
wine = datasets.load_wine()
data = wine.data

X = pd.DataFrame(data,columns=wine.feature_names)
y = wine['target']

#%% data_读取csv文件
filename = "/Users/wang/university/homework/AI-astronomy/work/machine-learning/datasets/credit/credit.csv"
df = pd.read_csv(filename)

X = df.drop('good_bad',axis=1)
df['target'] = 0
df.loc[(df.good_bad == 'bad'), 'target'] = 1
y = df['target'] 

#%%  Step2 数据预处理_独热——离散不用做

sepal_length = pd.get_dummies(X.sepal_length,prefix='sepal_length')
sepal_width = pd.get_dummies(X.sepal_width,prefix='sepal_width')
petal_length = pd.get_dummies(X.petal_length,prefix='petal_length')
petal_width = pd.get_dummies(X.petal_width,prefix='petal_width')

X1 = pd.concat([sepal_length,sepal_width,petal_length,petal_width],axis=1)

#%% 数据预处理_标准化

standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
X = pd.DataFrame(X_standard, columns=X.columns)

#%% Step3 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123456)

#%% Step4  逻辑回归lr 需要独热
from sklearn.linear_model import LogisticRegression

filename = "/Users/wang/university/homework/AI-astronomy/work/machine-learning/datasets/credit/credit.csv"
df = pd.read_csv(filename)

checking = pd.get_dummies(df.checking,prefix='checking')
history = pd.get_dummies(df.history,prefix='history')
purpose = pd.get_dummies(df.purpose,prefix='purpose')
savings = pd.get_dummies(df.savings,prefix='savings')
employed = pd.get_dummies(df.employed,prefix='employed')
installp = pd.get_dummies(df.installp,prefix='installp')
marital = pd.get_dummies(df.marital,prefix='marital')
coapp = pd.get_dummies(df.coapp,prefix='coapp')
installp = pd.get_dummies(df.installp,prefix='installp')
resident = pd.get_dummies(df.resident,prefix='resident')
property = pd.get_dummies(df.property,prefix='property')
housing = pd.get_dummies(df.housing,prefix='housing')
existcr = pd.get_dummies(df.existcr,prefix='existcr')
job = pd.get_dummies(df.job,prefix='job')
depends = pd.get_dummies(df.depends,prefix='depends')
telephon = pd.get_dummies(df.telephon,prefix='telephon')
foreign = pd.get_dummies(df.foreign,prefix='foreign')

# pd.concat(列表，axis=1)
X = pd.concat([df.duration, df.amount, df.age, checking, history, purpose, savings, employed, installp, marital, coapp, installp, resident, property, housing, existcr, job, depends, telephon, foreign], axis=1)
standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
X = pd.DataFrame(X_standard, columns=X.columns)

df['target'] = 0
df.loc[(df.good_bad == 'bad'), 'target'] = 1
y = df['target'] 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123456)

lr = LogisticRegression(solver='liblinear')
lr.fit(X_train, y_train)     # y = lr.coef_* X + lr.intercept_

#调参
parameters = {
    'penalty': ('l1', 'l2'),
    'C': (0.01, 0.1, 1, 10) }

lr = LogisticRegression(solver='liblinear')
lr_search = GridSearchCV(lr, parameters, scoring='accuracy', cv=5)
lr_search.fit(X_train, y_train)

lr = LogisticRegression(C=lr_search.best_params_["C"],penalty=lr_search.best_params_["penalty"],solver='liblinear')
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

#%% KNN 
from sklearn.neighbors import KNeighborsClassifier

wine = datasets.load_wine()
data = wine.data

X = pd.DataFrame(data,columns=wine.feature_names)
y = wine['target']

standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
X = pd.DataFrame(X_standard, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123456)

knn = KNeighborsClassifier(n_neighbors=5) #创建实例
knn.fit(X_train,y_train)

knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

#KNN模型的决策边界
from matplotlib.colors import ListedColormap  
x=X.iloc[:,0:2]   #修改2个维度
y=y      

x_min,x_max=x.iloc[:,0].min() -.5, x.iloc[:,0].max()+.5
y_min, y_max=x.iloc[:,1].min()-.5, x.iloc[:,1].max()+.5

cmap_light=ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
h=.02
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))  
knn=KNeighborsClassifier()  
knn.fit(x,y)
Z=knn.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape) 
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
plt.scatter(x.iloc[:,0],x.iloc[:,1],c=y,s=10)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(),yy.max())

#%%决策树dt
from sklearn.tree import DecisionTreeClassifier, plot_tree

filename = "/Users/wang/university/homework/AI-astronomy/work/machine-learning/datasets/credit/credit.csv"
df = pd.read_csv(filename)
#X = df.drop('good_bad',axis=1)
#独热
checking = pd.get_dummies(df.checking,prefix='checking')
history = pd.get_dummies(df.history,prefix='history')
purpose = pd.get_dummies(df.purpose,prefix='purpose')
savings = pd.get_dummies(df.savings,prefix='savings')
employed = pd.get_dummies(df.employed,prefix='employed')
installp = pd.get_dummies(df.installp,prefix='installp')
marital = pd.get_dummies(df.marital,prefix='marital')
coapp = pd.get_dummies(df.coapp,prefix='coapp')
installp = pd.get_dummies(df.installp,prefix='installp')
resident = pd.get_dummies(df.resident,prefix='resident')
property = pd.get_dummies(df.property,prefix='property')
housing = pd.get_dummies(df.housing,prefix='housing')
existcr = pd.get_dummies(df.existcr,prefix='existcr')
job = pd.get_dummies(df.job,prefix='job')
depends = pd.get_dummies(df.depends,prefix='depends')
telephon = pd.get_dummies(df.telephon,prefix='telephon')
foreign = pd.get_dummies(df.foreign,prefix='foreign')
X = pd.concat([df.duration, df.amount, df.age, checking, history, purpose, savings, employed, installp, marital, coapp, installp, resident, property, housing, existcr, job, depends, telephon, foreign], axis=1)

df['target'] = 0
df.loc[(df.good_bad == 'bad'), 'target'] = 1
y = df['target']

standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
X = pd.DataFrame(X_standard, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123456)

dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X_train,y_train)

# 查看dt特征的重要性排序
features = pd.DataFrame({'feature':X_train.columns,'importance':dt.feature_importances_})
features = features.sort_values(by =['importance'], ascending=False)
print(features.head(20))

#画dt树
#plt.figure(figsize=(10,8))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=["good","bad"])
plt.title("Decision Tree Structure")
plt.show()

#调参
parameters = {
    'criterion':['gini','entropy'],
    'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12]}
dtree = DecisionTreeClassifier()
dtree_search = GridSearchCV(dtree, parameters, scoring='accuracy', cv=5)
dtree_search.fit(X_train, y_train)
#再拟合
dt = DecisionTreeClassifier(criterion=dtree_search.best_params_['criterion'],max_depth=dtree_search.best_params_['max_depth'])
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

#%% 随机森林 RT
from sklearn.ensemble import RandomForestClassifier

filename = "/Users/wang/university/homework/AI-astronomy/work/machine-learning/datasets/credit/credit.csv"
df = pd.read_csv(filename)
#X = df.drop('good_bad',axis=1)
#独热
checking = pd.get_dummies(df.checking,prefix='checking')
history = pd.get_dummies(df.history,prefix='history')
purpose = pd.get_dummies(df.purpose,prefix='purpose')
savings = pd.get_dummies(df.savings,prefix='savings')
employed = pd.get_dummies(df.employed,prefix='employed')
installp = pd.get_dummies(df.installp,prefix='installp')
marital = pd.get_dummies(df.marital,prefix='marital')
coapp = pd.get_dummies(df.coapp,prefix='coapp')
installp = pd.get_dummies(df.installp,prefix='installp')
resident = pd.get_dummies(df.resident,prefix='resident')
property = pd.get_dummies(df.property,prefix='property')
housing = pd.get_dummies(df.housing,prefix='housing')
existcr = pd.get_dummies(df.existcr,prefix='existcr')
job = pd.get_dummies(df.job,prefix='job')
depends = pd.get_dummies(df.depends,prefix='depends')
telephon = pd.get_dummies(df.telephon,prefix='telephon')
foreign = pd.get_dummies(df.foreign,prefix='foreign')
X = pd.concat([df.duration, df.amount, df.age, checking, history, purpose, savings, employed, installp, marital, coapp, installp, resident, property, housing, existcr, job, depends, telephon, foreign], axis=1)

df['target'] = 0
df.loc[(df.good_bad == 'bad'), 'target'] = 1
y = df['target']

standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
X = pd.DataFrame(X_standard, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123456)

rf=RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 调参
param_grid = {
    'n_estimators':[5,10,20,30,40,50],
    'max_features':[1,2,3,4,5,6,7],
    'criterion':['gini','entropy',"log_loss"]}
rf=RandomForestClassifier(random_state=42)
CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)

rf = RandomForestClassifier(criterion=CV_rfc.best_params_['criterion'],max_features=CV_rfc.best_params_['max_features'],n_estimators=CV_rfc.best_params_[ 'n_estimators'])
rf.fit(X_train, y_train)

# 查看dt特征的重要性排序
features = pd.DataFrame({'feature':X_train.columns,'importance':rf.feature_importances_})
features = features.sort_values(by =['importance'], ascending=False)
print(features.head(20))

rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

#%% 支持向量基 svc
from sklearn.svm import SVC # "Support vector classifier"

iris = datasets.load_iris() 
data = iris['data']
feature_names = ["sepal_length","sepal_width","petal_length","petal_width"]
  #iris.feature_names = ['sepal length (cm)','sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

X = pd.DataFrame(data,columns=feature_names)
y = iris['target']

standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
X = pd.DataFrame(X_standard, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123456)

svc = SVC(gamma=0.001, C=100)
svc.fit(X_train, y_train)
#调参
param_grid = {
    'C':[40,50,60,90,100],
    'gamma':[0.003,0.001,0.005]}
classifier = SVC(gamma=0.001, C=100)
CV_classifier = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= 5)
CV_classifier.fit(X_train, y_train)

svc = SVC(gamma=CV_classifier.best_params_["gamma"], C=CV_classifier.best_params_["C"])
svc.fit(X_train, y_train)

svc_pred = svc.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_pred)

#%% 神经网络 mlp
from sklearn.neural_network import MLPClassifier
iris = datasets.load_iris() 
data = iris['data']
feature_names = ["sepal_length","sepal_width","petal_length","petal_width"]
  #iris.feature_names = ['sepal length (cm)','sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X = pd.DataFrame(data,columns=feature_names)
y = iris['target']

std = StandardScaler()
X_standard = std.fit_transform(X)
X = pd.DataFrame(X_standard, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123456)

mlp = MLPClassifier(hidden_layer_sizes=(4),max_iter=500)    #4层
mlp.fit(X_train,y_train)

mlp_pred = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)

#%% 聚类 Kmeans 无需拆分
from sklearn.cluster import KMeans

iris = datasets.load_iris() 
data = iris['data']
feature_names = ["sepal_length","sepal_width","petal_length","petal_width"]
  #iris.feature_names = ['sepal length (cm)','sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X = pd.DataFrame(data,columns=feature_names)
y = iris['target']

std = StandardScaler()
X_standard = std.fit_transform(X)
X = pd.DataFrame(X_standard, columns=X.columns)

# 寻找最优的聚类个数——图
import scipy
from scipy.cluster import hierarchy
dendro=hierarchy.dendrogram(hierarchy.linkage(X,method='ward'))
plt.show()

kmeans = KMeans(n_clusters=3, random_state=123456)
kmeans.fit(X)

cluster_pred = kmeans.predict(X)  #预测的分类的标签0、1、2
cluster_centers = kmeans.cluster_centers_  #聚类中心

cluster = pd.DataFrame(cluster_pred,columns=['cluster'])

plt.figure(figsize=(8, 6))
plt.scatter(X.iloc[cluster_pred == 0, 0], X.iloc[cluster_pred == 0, 1], label='Cluster 0', s=50, color='blue')
plt.scatter(X.iloc[cluster_pred == 1, 0], X.iloc[cluster_pred == 1, 1], label='Cluster 1', s=50, color='green')
plt.scatter(X.iloc[cluster_pred == 2, 0], X.iloc[cluster_pred == 2, 1], label='Cluster 2', s=50, color='red')
plt.scatter(cluster_centers[:, 0],cluster_centers[:, 1], c='yellow', s=200, alpha=0.75)  # 簇中心

#plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=cluster_pred, s=50, cmap='viridis')
#plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, alpha=0.75)  # 簇中心

plt.title("KMeans Cluster Centers")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.show()

# 0,1,2 不一定一一对应
cluster_ = cluster_ = cluster.replace({0: 2, 2: 0})
print(accuracy_score(y,cluster))
print(classification_report(y,cluster))
print(confusion_matrix(y,cluster))


#%% 降维 PCA
from sklearn.decomposition import PCA 
wine = datasets.load_wine()
data = wine.data

X = pd.DataFrame(data,columns=wine.feature_names)
y = wine['target']

std = StandardScaler()
X_standard = std.fit_transform(X)
X = pd.DataFrame(X_standard, columns=X.columns)

pca = PCA(n_components=6)    #降低到6维
X_pca = pca.fit_transform(X)

var_ratio = pca.explained_variance_ratio_#计算比例
print(var_ratio)

X_train, X_test, y_train, y_test = train_test_split(X_pca,y, test_size=0.3, random_state=123456)

#...

#%%  Step5 模型评估

def show_confusion_matrix(cnf_matrix, class_labels):
    plt.matshow(cnf_matrix, cmap=plt.cm.YlGn, alpha=0.7)
    ax = plt.gca()
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks(range(0,len(class_labels)))
    ax.set_xticklabels(class_labels,rotation=45)
    ax.set_ylabel('Actual Label', fontsize=16, rotation=90)
    ax.set_yticks(range(0,len(class_labels)))
    ax.set_yticklabels(class_labels)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    for row in range(len(cnf_matrix)):
        for col in range(len(cnf_matrix[row])):
            ax.text(col, row, cnf_matrix[row][col], va='center', ha='center', fontsize=16)

def show_result(pred,accuracy):
    print(f"accuracy:{accuracy}")
    print(classification_report(y_test,pred))
    print(confusion_matrix(y_test,pred))
    
    class_labels = [0,1]
    cnf_matrix = confusion_matrix(y_test,pred) 
    show_confusion_matrix(cnf_matrix, class_labels)
    plt.show()

show_result(lr_pred, lr_accuracy)
show_result(knn_pred, knn_accuracy)
show_result(dt_pred, dt_accuracy)
show_result(rf_pred, rf_accuracy)
show_result(svc_pred, svc_accuracy)
show_result(mlp_pred, mlp_accuracy)
show_result(mlp_pred, mlp_accuracy)
show_result(mlp_pred, mlp_accuracy)

