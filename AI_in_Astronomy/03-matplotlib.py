#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 21:06:46 2025

@author: wang
"""

import numpy as np
import matplotlib.pyplot as plt

plt.gcf()/plt.gca()  #获得当前图/子图
plt.scf()/plt.sca()  #设定当前图/子图
plt.clf()/plt.cla()  #清空当前图/子图

#%%面对对象式 [距离左边多远,距离底边多远,宽长，高长]

fig = plt.figure()  # 设置图形大小

ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                   xticklabels=[],ylim=(-1.2, 1.2), title='A')
x = np.linspace(0, 10)
ax1.plot(np.sin(x), label='sin(x)')
ax1.legend()
ax1.grid(True,color="grey",alpha=0.4,linestyle="--")
ax1.set_xlabel('x')  
ax1.set_ylabel('sin(x)') 

ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                   ylim=(-1.2, 1.2), title='B')
ax2.plot( np.cos(x), label='cos(x)')
ax2.legend()  
ax2.grid(True,color="grey",alpha=0.4,linestyle="--")
ax2.set_xlabel('x')
ax2.set_ylabel('cos(x)')  
plt.show()

#%% 移动轴到图中央

import numpy as np
import matplotlib.pyplot as plt

# build data
x1 = np.linspace(-0.5*np.pi, 1.5*np.pi, 128)
y1 = 2 * np.sin(x1)
x2 = np.linspace(-0.5*np.pi, 1.5*np.pi, 256)
y2 = 1.7 * np.sin(x2)
x3 = np.linspace(-0.5*np.pi, 1.5*np.pi, 512)
y3 = 1.4 * np.sin(x3)
x4 = np.linspace(-0.5*np.pi, 1.5*np.pi, 1024)
y4 = 1.2 * np.sin(x4)
x5 = np.linspace(-0.5*np.pi, 1.5*np.pi, 2048)
y5 = 1.0 * np.sin(x5)

# Plot...
plt.plot(x1, y1, '-', linewidth=1, color='b', label='G-FeCow at open circuit')
plt.plot(x2, y2, '--', linewidth=1, color='g', label='G-FeCow at $+1.4$ V')
plt.plot(x3, y3, '-.', linewidth=1, color='r', label='A-FeCow at open circuit')
plt.plot(x4, y4, ':', linewidth=1, color='c', label='A-FeCow at $+1.4$ V')
plt.plot(x5, y5, '-', linewidth=1, color='m', label='WO$_3$ ref')
plt.xlim(-0.5*np.pi, 1.5*np.pi)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
plt.xticks([-np.pi/2, 0, np.pi/2, np.pi, np.pi*3/2],
           [r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
plt.yticks([-2, -1, 0, 1, 2])
plt.xlabel('Energy (eV)')
plt.ylabel('WL$_3$-edge XANES Intensity (A.U.)')
plt.title('SINAP Demo A')

# plt.legend()
plt.legend(fontsize='x-small')
# export a pdf file
#plt.savefig('demo01h.png')
plt.show()

#%% 画子图

plt.subplot(2, 2, 1)
plt.plot(x1, y1, '.')

plt.subplot(2, 2, 2)
plt.plot(x1, y1, '-')

plt.subplot(2, 2, 3)
plt.plot(x1, y1, '+')

plt.subplot(2, 2, 4)
plt.plot(x1, y1, 'o')

#%%不规则子图

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))
