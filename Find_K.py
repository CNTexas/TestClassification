#!/usr/bin/env python

# encoding: utf-8

'''
@author: CNTexas
@contact: dzhx0621@gmail.com
@file: Find_K.py
@time: 2018/4/16 22:49
@desc:
'''
from sklearn import metrics
from Tools import *
from sklearn.neighbors import KNeighborsClassifier

import math
def findK():
    # 导入训练集
    trainpath = "valid_train_word_bag/tfdifspace.dat"
    train_set = readbunchobj(trainpath)

    # 导入测试集
    testpath = "valid_test_word_bag/testspace.dat"
    test_set = readbunchobj(testpath)

    ###
    list_k = {}
    count=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    for i in range(19):
        knn = KNeighborsClassifier(n_neighbors=count[i],weights='distance',metric='euclidean')
        knn.fit(train_set.tdm,train_set.label)
        predicted = knn.predict(test_set.tdm)
        #list_k.append(metrics.precision_score(test_set.label, predicted, average='weighted'))
        list_k[i]=metrics.precision_score(test_set.label, predicted, average='weighted')
        #print("{}:{}".format(i,list_k[i]))
    return list(list_k.keys())[list(list_k.values()).index(max(list_k.values()))]


