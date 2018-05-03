#!/usr/bin/env python

# encoding: utf-8

'''
@author: CNTexas
@contact: dzhx0621@gmail.com
@file: crossTest.py
@time: 2018/5/3 9:47
@desc:
'''
from Tools import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def cross_validated():
    # 导入测试集
    testpath = "train_word_bag/tfdifspace.dat"
    test_set = readbunchobj(testpath)


    count=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    list_k = {}
    for i in range(19):
        knn = KNeighborsClassifier(n_neighbors=count[i],weights='distance',metric='euclidean')
        scores = cross_val_score(knn, test_set.tdm, test_set.label, cv=10)
        list_k[i] =scores.mean()
    print(list_k.values())
    return list(list_k.keys())[list(list_k.values()).index(max(list_k.values()))]
if __name__ == '__main__':
    print(cross_validated())
