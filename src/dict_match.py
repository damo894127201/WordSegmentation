# -*- coding: utf-8 -*-
# @Time    : 2019/7/12 10:40
# @Author  : Weiyang
# @File    : dict_match.py

########################################################################################################################
# 分词算法：最大正向匹配算法
# 算法思想：最大匹配算法是指以字典为依据，
########################################################################################################################

def dict_match(line,mydict):
    '''line: 输入的文本行，mydict: 自定义的分词字典,格式为:[word1,word2,...]'''
    # 遍历自定义分词字典，找出字典中最大单词的长度
    max_len = 0
    for word in mydict:
        if len(word) > max_len:
            max_len = len(word)
    # 将不同单词的长度分配到不同的字典中
    wordDict = dict()
    for key in range(1,max_len+1):
        wordDict[key] = []
    for word in mydict:
        wordDict[len(word)].append(word)
    # 开始分词
    result = [] # 存储分词结果
    # 移动遍历文本行，直到文本行分词完毕
    while len(line) > 0:
        # 用pointer表示当前移动指针所在的位置，然后取该位置内的文本，循环遍历不同长度的文本，同时匹配字典
        pointer = max_len  # 当前指针，用以确定当前用于匹配的字段
        while len(line[:pointer]) > 0:
            # 如果当前字段在相应长度的词典中，则加入分词结果中
            if line[:pointer] in wordDict[pointer]:
                result.append(line[:pointer])
                # 删除原始文本行line中已分词的字段，并跳出循环
                line = line[pointer:]
                break
            else:
                # 如果当前字段只剩一个字，则将其加入分词结果中，并跳出当前扫描
                if pointer == 1:
                    result.append(line[:pointer])
                    # 删除原始文本行line中已分词的字段
                    line = line[pointer:]
                    break
                pointer -= 1
        # 判断文本行是否只有一个字
        if len(line) == 1:
            result.append(line)
            break
    return result

if __name__ == '__main__':
    mydict = ['aa','b','dd','dsfds','dfd']
    line = 'adfdfdddsfdsdfdaab'
    result = dict_match(line,mydict)
    print(result)
