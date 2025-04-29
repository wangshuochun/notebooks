#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:33:30 2025

Panda 使用说明  

>> Series  比 ndarray多 索引
>> Dataframe  比 Series可 多列

@author: wang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%% Series  一维数组 & 对应索引
data = [3,2,5]
a = pd.Series(data)   #data 数据，支持类数组、字典、标量对象；

from datetime import datetime
times = [datetime(2018, 10, 10, 6), datetime(2018, 10, 10, 7), 
         datetime(2018, 10, 10, 8), datetime(2018, 10, 10, 11), 
         datetime(2018, 10, 10, 12), datetime(2018, 10, 10, 13), 
         datetime(2018, 10, 10, 18), datetime(2018, 10, 10, 19), 
         datetime(2018, 10, 10, 20), datetime(2018, 10, 10, 21)]
b = pd.Series([60, 70, 65, 100, 230, 150, 100, 300, 250, 150], index=times)
b.index
b.values

# 索引 Series对象 —— 字典
b['2018-10-10 06:00:00']
b.iloc[0]

b['2018-10-10 06:00:00'] = 100    #可变的

#%% Dataframe 多了列索引
times = np.array([6, 7, 8, 11, 12, 13, 18, 19, 20, 21])
customers = np.array([60, 70, 65, 100, 230, 150, 100, 300, 250, 150])
costs = np.array([6, 7, 8, 24, 23, 26, 45, 55, 45, 40])
data = {'customers': customers, 'costs': costs}

df = pd.DataFrame(data, index=times)  #可列表、字典、数组

df.index  #行索引，不指定则为1-n
df.values
df.columns  #列索引

df=pd.Dataframe(data,index=times,columns=["customers","costs"])

#%% 写csv文件

%%writefile demo01.csv
restaurant_name,times,meals,customers,costs,notes
Xuhui,2018-10-10 06:00:00,breakfast,60,6.6,xx
Xuhui,2018-10-10 07:00:00,breakfast,70,7.6,xx
Xuhui,2018-10-10 11:00:00,lunch,100,24.5,xx
Xuhui,2018-10-10 12:00:00,lunch,230,23.4,xx
Xuhui,2018-10-10 20:00:00,dinner,250,35.5,xx
Xuhui,2018-10-10 21:00:00,dinner,150,40.6,xx

#%% 读取文件

pd.read_csv('data.csv', sep=',', header=0, index_col=None, usecols=['column1', 'column2'],skiprows=2)     
#读取CSV文件,header：指定哪一行作为列名，默认为 0（第一行); index_col：指定哪一列作为行索引;skiprows：跳过文件开始的行数;nrows：读取的行数.
pd.read_table('data.tsv', sep='\t', header=0, index_col=None)   #读取分隔符文本文件
pd.read_excel('data.xlsx', sheet_name='Sheet1', header=0, index_col=None, usecols=['column1', 'column2'])   #读取Excel文件
pd.read_sql(filename)     #读取SQL表/库
pd.read_json(filename)    #读取JSON字符串
pd.read_html(filename)    #解析URL或者HTML文件，抽取表格数据
pd.DataFrame(dict)        #从字典对象创建`DataFrame`

#%% 输入文件

df.to_csv('output.csv', sep=",",index=True,header=True, encoding='utf-8',columns=["A","B"])
df.to_excel('output.xlsx', sheet_name='Data', index=False,header=True)
df.to_json(filename)
df.to_sql(filename)
#%% 数据检索与描述性统计

#数据选取
df.shape
df.head(n)
df.tail(n)
df.info
df.dtypes           # 数据列及其类型

df["列名1","列名2"]
df.loc["行索引"]
df.iloc[0,:]        #按位置索引
df.iloc[1:5]        #索引2-5行
df[df["flag"]==1]
df_sigma = df.std() #数据计算标准差，返回dataframe
df.loc[abs(df['it']) > 8 * df_sigma['it'], 'it']
df.loc[abs(df['finance']) > 8 * df_sigma['finance'], 'finance']

#数据统计
df.isna()                   # 是否有Nan数据
df.isnull().sum()           # null值统计
df.duplicated().sum()       # 是否有重复行

df.describe()               # 平均值等多个统计结果
df.value_counts()           # 频数统计
df['列名'].unique()          # 将每一列的重复的值清空

#分组计算-groupby
df.groupby('hotal_name')['total_costs'].count() #按name来分组计算cost的值，sum、mean

#%%清洗数据

#无用数据
df.drop(["A","B"],axis=1,inplace=True)      #删除多列，axis=1 表示操作的是列,inplace=True 表示在原DataFrame修改 

#重复数据
df.duplicated()             #查看重复的行
df.drop_duplicates(['列名1', '列名2'], inplace=True)   #按照列名查看有无重复的行，保留第一次出现的行，并在原Dataframe上修改（inplace=True）

#缺失数据
df.isnull()                 #返回缺失数据的布尔结果
df.notnull()                #返回有数据的布尔结果
df2=df.dropna(inplace=False,axis=0,how="any")  #删除有nan值的行（默认axis=0，列为1），只要有一个nan值便删除（how=“any”，为“all”需整行为nan），返回新dataframe
df.fillna(value=0,inplace=True)                #填用0充缺失数据

#错误数据
df.replace(to_replace='error_value', value='correct_value', inplace=True)  #修改错误数据

#更改数据类型
s.astype(float)             #更改数据类型
#%% 关联和合并

pd.concat([data1,data2], axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True)
# axis默认为 0（行方向）,1为列方向；join连接方式，'outer'表示并集，'inner'表示交集；ignore_index 如果为 True，则不保留原始索引，而是生成一个新的连续索引；keys：用于创建层次化索引的键列表.

df.append(data2, ignore_index=False, verify_integrity=False, sort=False)
#ignore_index：如果为 True，则不保留原始索引，而是生成一个新的连续索引.

pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
#how：合并方式，可以是 'inner'（内连接）、'outer'（外连接）、'left'（左连接）、'right'（右连接);
#on：合并的键名，如果在两个 DataFrame 中键名相同，可以使用 on 参数;
#left_on 和 right_on：分别指定左 DataFrame 和右 DataFrame 中的键名;
#left_index 和 right_index：如果为 True，则使用索引作为合并键.;
#suffixes：用于处理重名列的后缀.



