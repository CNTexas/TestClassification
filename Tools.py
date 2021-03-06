#!/usr/bin/env python
# encoding: utf-8
'''
@author: CNTexas
@contact: dzhx0621@gmail.com
@file: py.py
@time: 2018/4/15 21:35
@desc:
'''

import pickle
import matplotlib.pyplot as plt
import numpy as np

# 保存至文件
def savefile(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)


# 读取文件
def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


def writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


# 读取bunch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch
#
#输出结果
'''
def drawret(total,acurations):
    x = [0, 10]
    plt.figure()
    plt.ylim(0, 1.0, 0.1)
    print(acurations)
    for acuration in acurations:
        y=[acuration,acuration]
        #print(acuration)
        plt.plot(x,y)
    y=[total,total]
    plt.plot(x,y,'r')
    plt.show()
'''
def drawret(acurations):
    name_list = ['Business', 'IT', 'Health', 'Education','Military','Tourism','Sports','Recruitment']
    plt.bar(range(len(acurations)), acurations,color='rgbyckmr',tick_label=name_list)
    plt.show()

