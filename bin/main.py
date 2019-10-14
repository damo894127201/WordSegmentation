# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 22:23
# @Author  : Weiyang
# @File    : main.py

#===================================================================================
# 分词器展示
#===================================================================================

from src.Segmentation import Segmentation
import jieba as jb

model = Segmentation(model='S')

while True:
    print('请输入一个句子:')
    sentence = input()
    result = model.cut(sentence)
    result2 = jb.cut(sentence)
    print('我的模型分词结果:')
    print('\t'*2,result)
    print()
    #print('结巴分词结果:')
    #print('\t'*2,list(result2))
    #print()