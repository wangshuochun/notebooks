#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:09:06 2025

with open("xx.txt","r") as f 

@author: wang
"""

#%%常见的读写操作：

with open('filename.txt', 'r') as f:
   content = f.read(f)  #文件的读操作

with open('data.txt', 'w') as f:
   f.write('hello world')  #文件的写操作

#参数
r:	 # 以只读方式打开文件。文件的指针将会放在文件的开头。这是**默认模式**。
rb:  # 以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。
r+:  # 打开一个文件用于读写。文件指针将会放在文件的开头。
rb+: # 以二进制格式打开一个文件用于读写。文件指针将会放在文件的开头。
w:	 # 打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
wb:	 # 以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
w+:	 # 打开一个文件用于读写。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
wb+: # 以二进制格式打开一个文件用于读写。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
a:	 # 打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
ab:	 # 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
a+:	 # 打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写。
ab+: # 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。如果该文件不存在，创建新文件用于读写。

#file对象的属性
file.read([size])        #从文件读取指定的字节数，如果未给定或为负则读取所有。将文件数据作为字符串返回，可选参数size控制读取的字节数
file.readlines([size])   #读取整行，包括 "\n" 字符。返回文件中行内容的列表，size参数可选
file.write(str)          #将字符串序列写入文件，如果需要换行则要自己加入每行的换行符。
file.close()             #关闭文件
file.closed	             #表示文件已经被关闭，否则为False

file.mode	             #Access文件打开时使用的访问模式
file.encoding	         #文件所使用的编码
file.name	             #文件名
file.newlines	         #未读取到行分隔符时为None，只有一种行分隔符时为一个字符串，当文件有多种类型的行结束符时，
                         #则为一个包含所有当前所遇到的行结束的列表
file.softspace	         #为0表示在输出一数据后，要加上一个空格符，1表示不加。这个属性一般程序员用不着，由程序内部使用

#%% open 和 with （open 需要关闭文件操作）
try:
    f = open('/path/to/file', 'r')
    print(f.read())
finally:
    if f:
        f.close()


with open('/path/to/file', 'r',encoding='UTF-8') as f:
    print(f.read())
    
    for line in f.readlines(): 
        print(line.strip()) # 把末尾的'\n'删掉

#%%实例
#%%writefile sumscore.py
# Open the input and output file
inpfile = '/Users/wang/university/homework/AI-astronomy/work/python-learning/Python学习/python-basics/9-文件处理/assets/scores.csv'
outfile = '/Users/wang/university/homework/AI-astronomy/work/python-learning/Python学习/python-basics/9-文件处理/assets/scores.done.csv'

with open(inpfile, 'r') as inpfh, open(outfile, 'w') as outfh:
    # the first line
    headline = inpfh.readline()         #读取第一行
    s = '{0}   total\n'.format(headline.strip())      # 去除行尾的换行符，并添加total列
    outfh.write(s)

    s = '{0:>4}   {1:<16}  {2:4.1f}  {3:4.1f}  {4:4.1f}  {5:4.1f}\n'
    for line in inpfh:      #读取剩余行
        words = line.split()  #将每行按空格分割成列表
        studentNumber = int(words[0])
        name = words[1]
        chinese = float(words[2])
        math = float(words[3])
        python = float(words[4])
        total = chinese + math + python
        outline = s.format(studentNumber, name, chinese, math, python, total)
        outfh.write(outline)

#%%
import pickle  # 实际上在这个任务中不需要使用pickle

lst = [{'no':'1', 'name':'tom'}, {'no':'2', 'name':'rose'}, {'no':'3', 'name':'peter'}]

s = ""  # 空字符串，用于保存每位同学的信息
for item in lst:
    for k, v in item.items():
        s += str(k) + ":" + str(v) + " "
    s += "\n"

with open("aa.txt", "wb") as fw:  # 以二进制写的方式打开文件
    fw.write(s.encode())  # 将字符串对象s写入aa.txt文件

with open("aa.txt", "rb") as fr:  # 以二进制读的方式打开文件
    data = fr.read()  # 读取文件内容
    print(data.decode())  # 解码并打印内容
    
#%%

import shelve

lst = [{"no": "1", "name": "tom"}, {"no": "2", "name": "rose"}, {"no": "3", "name": "peter"}]

i = 1
with shelve.open("aa", flag="c") as fw:
    for item in lst:
        fw[i] = item  # 给每个列表元素加一个键 i
        i += 1

with shelve.open("aa", flag="r") as fr:  # 打开文件进行读取
    for item in fr:  # 在当前shelve对象中遍历
        print(item)  # 打印每个项

#shelve.open 的 flag 参数设置为 "c" 表示创建一个新的数据库文件，如果文件不存在的话。在读取时，您应该使用 "r" 标志来打开文件进行读取。此外，shelve 对象在遍历时会返回键，如果您需要值，可以使用 fr[item] 来访问。

