#encoding:utf-8
import os
import sys
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import metrics
from Tools import readbunchobj
from sklearn.neighbors import KNeighborsClassifier

# 导入训练集
trainpath = "train_word_bag/tfdifspace.dat"
train_set = readbunchobj(trainpath)

# 导入测试集
testpath = "test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)


SVM = LinearSVC(penalty="l1",dual=False,tol=1e-4)
SVM.fit(train_set.tdm,train_set.label)

#预测分类结果
predicted = SVM.predict(test_set.tdm)

count=0
for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        print(file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate)
    else:
        count=count+1

print("正确的判断:"+str(count))
print("预测完毕!!!")

# 计算分类精度：

def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))


metrics_result(test_set.label, predicted)
####