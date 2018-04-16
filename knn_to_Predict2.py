#!/usr/bin/env python
# encoding: utf-8
'''
@author: CNTexas
@contact: dzhx0621@gmail.com
@file: py.py
@time: 2018/3/15 21:35
@desc:
'''
from sklearn import metrics
from Tools import *
from sklearn.neighbors import KNeighborsClassifier
from Find_K import findK
# 导入训练集
trainpath = "train_word_bag/tfdifspace.dat"
train_set = readbunchobj(trainpath)

# 导入测试集
testpath = "test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)
print(1)
##
K = findK()
knn = KNeighborsClassifier(n_neighbors=K,weights='distance',metric='euclidean')
knn.fit(train_set.tdm,train_set.label)
predicted = knn.predict(test_set.tdm)

count=0
total=0
tem=0
test_set_len = 300
acurations=list()
for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    tem += 1
    if flabel != expct_cate:
        print(file_name, ": 实际类别:", flabel, " --->预测类别:", expct_cate)
    else:
        count = count + 1
        total += 1
    if tem == 300:
        acurations.append(count / tem)#最后一次没有传入，或许数据少了几个，导致没到300就结束了。
        count = 0
        tem=0
acurations.append(count / tem)


print("正确的判断:"+str(count))
print("预测完毕!!!")

# 计算分类精度：

def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))


metrics_result(test_set.label, predicted)
drawret(total/(test_set_len*9),acurations=acurations)
####