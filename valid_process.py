#!/usr/bin/env python

# encoding: utf-8

'''
@author: CNTexas
@contact: dzhx0621@gmail.com
@file: valid_process.py
@time: 2018/4/16 22:38
@desc:
'''
import os
import jieba

from Tools import savefile, readfile
from corpus_segment import *
from corpus_Bunch import *
from TFIDF_space import *

if __name__ == "__main__":
    # 对训练集进行分词
    corpus_path = "./valid_train_corpus/"  # 未分词分类语料库路径
    seg_path = "./valid_train_corpus_seg/"  # 分词后分类语料库路径
    corpus_segment(corpus_path, seg_path)

    # 对测试集进行分词
    corpus_path = "./valid_test_corpus/"  # 未分词分类语料库路径
    seg_path = "./valid_test_corpus_seg/"  # 分词后分类语料库路径
    corpus_segment(corpus_path, seg_path)

    # 对训练集进行Bunch化操作：
    wordbag_path = "valid_train_word_bag/train_set.dat"  # Bunch存储路径
    seg_path = "valid_train_corpus_seg/"  # 分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)

    # 对测试集进行Bunch化操作：
    wordbag_path = "valid_test_word_bag/test_set.dat"  # Bunch存储路径
    seg_path = "valid_test_corpus_seg/"  # 分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)

    stopword_path = "valid_train_word_bag/chinese_stopword.txt"
    bunch_path = "valid_train_word_bag/train_set.dat"
    space_path = "valid_train_word_bag/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path)

    bunch_path = "valid_test_word_bag/test_set.dat"
    space_path = "valid_test_word_bag/testspace.dat"
    train_tfidf_path = "valid_train_word_bag/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)