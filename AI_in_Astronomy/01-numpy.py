#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 13:06:25 2025

Numpy >> 多维数组 ndarray

@author: wang
"""

import numpy as np 
import scipy
import matplotlib.pyplot as plt

#%%创建数组
arr = np.arange(9).reshape((3,3))
np.transpose(arr) #逆矩阵

"""
使用如下函数可以创建数组：
- `np.zeros()`，创建数组，填充为0。
- `np.ones()`，创建数组，填充为1。
- `np.empty()`，创建数组，元素均未初始化。
- `np.full()`，创建数组，用指定值来填充。
- `np.eye()`，创建二维数组，其中对角线为1，零点为零。
- `np.identity()`，创建标识数组，维度由一个正整数指定，除了对角线值为1外，其他均为0值。
扩展：
- `np.zeros_like()`，仿照给定数组，返回新数组，填充为0。
- `np.ones_like`， 仿照给定数组，返回新数组，填充为1。
- `np.empty_like`，仿照给定数组，返回新数组。
- `np.full_like`，仿照给定数组，返回新数组，用指定值来填充。

用来创建类似网格形式的数组:
- `np.arange()`，在给定的间隔内返回均匀间隔的值。
- `np.linspace()`，在指定的间隔内返回均匀间隔的数字。
- `np.logspace()`，返回数在对数刻度上均匀分布。
- `np.geomspace()`，返回数在对数尺度上均匀分布(几何级数)。

"""

a0 = np.zeros(5)
a1 = np.ones(5)
a2 = np.full((2,2),np.inf)

x = np.arange(6, dtype=int)
np.full_like(x, 1)

a3 = np.logspace(0,100,10)
dat = np.linspace(0 , 2*np.pi , 360)

#%%`Numpy`包提供由常用的数学常数：

np.pi
np.e
np.euler_gamma  #欧拉常数
np.PZERO, np.NZERO #正数零，负数零
np.nan #非数字
np.inf #无穷大

#%%统计数据
x = np.arange(100)

np.sum(x) 
np.max()
np.min()
np.mean(x)  

np.median(x)
np.average()  #加权平均值
scipy.stats.mode(x)
np.var(x)     #方差
np.std(x)     #标准差

np.cumsum()   #累计求和
np.cumprod()  #累计乘积

np.corrcoef(x,y)   #计算pearson相关系数
np.cov(x,y)        #计算协方差阵
np.correlate(a, v) #计算相关系数
#%% 数学运算
arr1 = np.array([1.0, 2.0, 3.0, 4.0])
arr2 = np.array([-1.0, -2.0, -3.0, -4.0])

np.add(arr1,arr2)
np.divide(arr1,arr2)
np.multiply(arr1,arr2)

#%% random
n=10

np.random.rand(n,n)                     #多维，[0,1]
np.random.randn(n,n)                    #多维,标准正态分布
np.random.randint(1,100,n,dtpye=float)  #随机整数 [1,100]
np.random.random(n)                     #随机浮点数 [0,1]

np.random.normal(loc=0.0,scale=1.0,size=n) #正态高斯分布
np.random.uniform(low=0.0,high=1.0,size=n) #均匀分布

#%%数据拟合

xdata = (0.21, 0.26, 0.27, 0.27, 0.45, 0.5, 0.8, 1.1, 1.4, 2.0)
ydata = (130, 70, 185, 220, 200, 270, 300, 450, 500, 800)
plt.plot(xdata, ydata, '.')
         
#多项式拟合 polyfit
deg = 4 #1为线性拟合，2为二次拟合
z1 = np.polyfit(xdata, ydata, deg)

x = np.arange(0, 2.2, 0.01)
y = np.polyval(z1, x) #计算

plt.plot(xdata, ydata, '.')
plt.plot(x, y)
plt.show()

##例子——sin拟合
xstart, xstop, ndata = -2*np.pi, 2*np.pi, 128
xdata = np.linspace(xstart, xstop, ndata)
ydata = np.sin(xdata) + np.random.normal(scale=0.5, size=ndata)
plt.plot(xdata, ydata, '.')

z1 = np.polyfit(xdata, ydata, 3)
z2 = np.polyfit(xdata, ydata, 5)
z3 = np.polyfit(xdata, ydata, 7)

x = np.linspace(xstart, xstop, 1024)
y1 = np.polyval(z1, x)
y2 = np.polyval(z2, x)
y3 = np.polyval(z3, x)

plt.plot(xdata, ydata, '.')
plt.plot(x, y1, label='3')
plt.plot(x, y2, label='5')
plt.plot(x, y3, label='7')
plt.legend()
plt.show()

#%% 在进行数组运算时，会忽略其中的`np.nan`值

np.nansum()                 #加
np.nancumsum()              #累加
np.nanprod()                #*
np.nancumprod()             #累乘

np.nanmax()
np.nanmin()
np.nanargmax()              #返回数组最大值的索引
np.nanargmin()

np.nanmean()
np.nanmedian()
np.nanstd() 
np.nansum() 
np.nanvar()

np.nanpercentile()          #百分位           

#%% 随机漫步实验

nsteps = 1024

nums = np.random.randint(0, 2, size=nsteps)
steps = np.where(nums > 0, 1, -1)
  #numpy.where(condition, [x, y])  true=>x, flase=> y
  
steps.cumsum()
plt.plot(steps.cumsum())



