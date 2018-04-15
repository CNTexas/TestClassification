#!/usr/bin/env python
# encoding: utf-8
'''
@author: CNTexas
@contact: dzhx0621@gmail.com
@file: py.py
@time: 2018/4/15 21:35
@desc:
'''

import os
import pickle

from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from Tools import readfile, readbunchobj, writebunchobj
from sklearn.datasets.base import Bunch
from Tools import savefile, readfile


def corpus2Bunch(wordbag_path, seg_path):
    catelist = os.listdir(seg_path)  # 获取seg_path下的所有子目录，也就是分类信息
    # 创建一个Bunch实例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)
    '''
    extend(addlist)是python list中的函数，意思是用新的list（addlist）去扩充
    原来的list
    '''
    # 获取每个目录下所有的文件
    for mydir in catelist:
        class_path = seg_path + mydir + "/"  # 拼出分类子目录的路径
        file_list = os.listdir(class_path)  # 获取class_path下的所有文件
        for file_path in file_list:  # 遍历类别目录下文件
            fullname = class_path + file_path  # 拼出文件名全路径
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(readfile(fullname))  # 读取文件内容
            '''append(element)是python list中的函数，意思是向原来的list中添加element，注意与extend()函数的区别'''
    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("构建文本对象结束！！！")


def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path=None):
    stpwrdlst = readfile(stopword_path).splitlines()
    bunch = readbunchobj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})

    if train_tfidf_path is not None:
        trainbunch = readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_

    writebunchobj(space_path, tfidfspace)
    print("if-idf词向量空间实例创建成功！！！")





if __name__ == "__main__":

    # 对训练集进行Bunch化操作：
    wordbag_path = "train_word_bag_2/train_set.dat"  # Bunch存储路径
    seg_path = "train_corpus_seg_2/"  # 分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)

    # 对测试集进行Bunch化操作：
    wordbag_path = "test_word_bag_2/test_set.dat"  # Bunch存储路径
    seg_path = "test_corpus_seg_2/"  # 分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)

    stopword_path = "train_word_bag_2/stopwords.txt"
    bunch_path = "train_word_bag_2/train_set.dat"
    space_path = "train_word_bag_2/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path)

    bunch_path = "test_word_bag_2/test_set.dat"
    space_path = "test_word_bag_2/testspace.dat"
    train_tfidf_path = "train_word_bag_2/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)



