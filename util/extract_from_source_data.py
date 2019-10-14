# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 8:12
# @Author  : Weiyang
# @File    : extract_from_source_data.py

#===================================================================================================================
# 从source_data目录下199801.txt中抽取数据，本模块返回以下数据：
# 1. 分词数据，eg：迈向  充满  希望  的  新  世纪  ——  一九九八年  新年  讲话  （  附  图片  １  张  ）
#    由例子可知，我们将标点符号也作为一个分词
# 2. 分词与词性对数据
#    eg：X = [迈向,充满,希望,的,新,世纪,——,一九九八年,新年,讲话,（,附,图片,１,张,）]
#        Y = [/v,/v,/n,/u,/a,/n,/w,/t,/t,/n,/w,/v,/n,/m,/q,/w]
# 3. 分词词典，即包含语料中出现的各种词及其其它字符
# 4. 词性词典，即包含语料中出现的各种词性
# 5. 字词典，即包含语料中出现的各种字及其其它字符
#===================================================================================================================

f1 = open('../data/word.txt','w',encoding='utf-8') # 存储分词数据
f2 = open('../data/pos.txt','w',encoding='utf-8')   # 存储分词对应的词性
f3 = open('../data/word_dict.txt','w',encoding='utf-8') # 存储分词词典
f4 = open('../data/pos_dict.txt','w',encoding='utf-8')  # 存储词性词典
f5 = open('../data/char_dict.txt','w',encoding='utf-8') # 存储字词典

words = []
poses = []
chars = []

with open('../source_data/199801.txt','r',encoding='utf-8') as fi:
    for line in fi:
        line = line.strip().split()
        if len(line) == 0:
            continue
        # 写入数据
        for item in line[1:]:
            word,pos = item.split('/')
            if '[' in word:
                word = word.strip('[')
                words.append(word)
                chars += list(word)
                f1.write(word + ' ')
            else:
                chars += list(word)
                words.append(word)
                f1.write(word + ' ')
            # 这里不处理语料中标注的短语
            if ']' in pos:
                pos,_ = pos.split(']')
                poses.append(pos)
                f2.write(pos + ' ')
            else:
                poses.append(pos)
                f2.write(pos + ' ')
        f1.write('\n')
        f2.write('\n')
# 增加一些特殊符号
for ch in list('0123456789'):
    f1.write(ch + '\n')
    f2.write('m' + '\n') # 添加对应的词性
for ch in list('/*-@#$!;:{}[]()（）》《、：；-——&…^'):
    f1.write(ch + '\n')
    f2.write('w' + '\n') # 添加对应的词性
words += list('0123456789/*-@#$!;:{}[]()（）》《、：；-——&…^')
chars += list('0123456789/*-@#$!;:{}[]()（）》《、：；-——&…^')
words = list(set(words))
poses = list(set(poses))
chars = set(chars)
f3.write(' '.join(words))
f4.write(' '.join(poses))
f5.write(' '.join(chars))
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
print(len(words),len(poses),len(chars))