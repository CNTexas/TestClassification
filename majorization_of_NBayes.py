from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
from sklearn import metrics
from Tools import *

import numpy as np
import matplotlib.pyplot as plt

# 导入训练集
trainpath = "train_word_bag_2/tfdifspace.dat"
train_set = readbunchobj(trainpath)

# 导入测试集
testpath = "test_word_bag_2/testspace.dat"
test_set = readbunchobj(testpath)

# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)

# 预测分类结果
predicted = clf.predict(test_set.tdm)

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
print("正确的判断:"+str(total))
print("预测完毕!!!")




# 计算分类精度：

def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))


metrics_result(test_set.label, predicted)
drawret(total/(test_set_len*9),acurations=acurations)