# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 12:25
# @Author  : Weiyang
# @File    : Segmentation.py

#==================================================================================================================
# 分词器：隐马尔可夫模型和字典匹配两种方式，其中，隐马尔可夫模型又分为 监督学习模型 和 无监督学习模型
#==================================================================================================================

from HMM import HMM
from dict_match import dict_match
import numpy as np

class Segmentation(object):
    '''分词器'''

    def __init__(self,model='S'):
        self.model = model # 选择分词的方式,有三种方式，无监督HMM模型，有监督HMM模型，字典匹配
                           # 取值分别为：'U','S','D'
        self.char2code = dict() # 字符到编码的映射
        self.code2char = dict()  # 编码到字符的映射
        self.code2latentState = dict() # 编码到隐状态的映射
        self.init_prob_dist = None # 初始概率矩阵
        self.state_trans_matrix = None # 状态转移概率矩阵
        self.emission_matrix = None # 观测概率矩阵
        self.words_dict = [] # 存储分词词典，用于字典匹配分词
        self._readParameter()

    def _readParameter(self):
        # 使用有监督学习的参数
        if self.model == 'S':
            with open('../matrix/S_char_code.txt','r',encoding='utf-8') as fi:
                for line in fi:
                    # 如果遇到:符号
                    if line.count(':') > 1:
                        _, _,code = line.strip().split(':')
                        ch = ':'
                        code = int(code)
                    else:
                        ch,code = line.strip().split(':')
                        code = int(code)
                    self.char2code[ch] = code
                    self.code2char[code] = ch
            with open('../matrix/S_latent_state_code.txt','r',encoding='utf-8') as fi:
                for line in fi:
                    # 如果遇到:符号
                    if line.count(':') > 1:
                        _, _,code = line.strip().split(':')
                        ch = ':'
                        code = int(code)
                    else:
                        ch,code = line.strip().split(':')
                        code = int(code)
                    self.code2latentState[code] = ch
            matrix = np.load('../matrix/S_init_prob_dist.npz')
            self.init_prob_dist = matrix['init_prob_dist']

            matrix = np.load('../matrix/S_state_trans_matrix.npz')
            self.state_trans_matrix = matrix['state_trans_matrix']

            matrix = np.load('../matrix/S_emission_matrix.npz')
            self.emission_matrix = matrix['emission_matrix']
        elif self.model == 'U':
            with open('../matrix/U_char_code.txt', 'r', encoding='utf-8') as fi:
                for line in fi:
                    # 如果遇到:符号
                    if line.count(':') > 1:
                        _, _,code = line.strip().split(':')
                        ch = ':'
                        code = int(code)
                    else:
                        ch, code = line.strip().split(':')
                        code = int(code)
                    self.char2code[ch] = code
                    self.code2char[code] = ch
            with open('../matrix/U_latent_state_code.txt', 'r', encoding='utf-8') as fi:
                for line in fi:
                    # 如果遇到:符号
                    if line.count(':') > 1:
                        _, _,code = line.strip().split(':')
                        ch = ':'
                        code = int(code)
                    else:
                        ch, code = line.strip().split(':')
                        code = int(code)
                    self.code2latentState[code] = ch
            matrix = np.load('../matrix/U_init_prob_dist.npz')
            self.init_prob_dist = matrix['init_prob_dist']

            matrix = np.load('../matrix/U_state_trans_matrix.npz')
            self.state_trans_matrix = matrix['state_trans_matrix']

            matrix = np.load('../matrix/U_emission_matrix.npz')
            self.emission_matrix = matrix['emission_matrix']
        elif self.model == 'D':
            with open('../data/word_dict.txt','r',encoding='utf-8') as fi:
                for line in fi:
                    self.words_dict += line.strip().split()

    def cut(self,inputs):
        '''inputs是输入数据，形式为中文字符序列,eg: 我今天很高兴！哈哈。。。。'''
        if self.model == 'D':
            result = dict_match(inputs,self.words_dict)
        else:
            model = HMM(num_latent_states=len(self.code2latentState),
                        num_observation_states=len(self.char2code),
                        init_prob_dist=self.init_prob_dist,
                        state_trans_matrix=self.state_trans_matrix,
                        emission_matrix=self.emission_matrix)
            # 将输入数据，转为数字编码列表
            new_input = [self.char2code[ch] for ch in inputs]
            result = model.viterbi([new_input])
            result = [self.code2latentState[code] for code in result[0]]
            result = self.formatResult(inputs,result)
        return result

    def formatResult(self,raw_input,result):
        '''
        raw_input: 是一个字符序列，即字符串；
        result是一个隐状态序列，eg: ['B','M',...]；
        本函数的目的是输出分词后的结果'''
        words = []
        word = '' # 临时存储字符
        for i,(ch,flag) in enumerate(zip(raw_input,result)):
            if flag == 'B':
                # 如果遇到'B'，则将先前的字符组成的词语加入结果中，并重置为空
                if word != '':
                    words.append(word)
                    word = ''
                word += ch
            elif flag == 'M':
                word += ch
            elif flag == 'E':
                word += ch
                if word != '':
                    words.append(word)
                    word = ''
            elif flag == 'S':
                if word != '':
                    words.append(word)
                    words.append(ch)
                    word = ''
                else:
                    if ch != '':
                        words.append(ch)
        return words

if __name__ == '__main__':
    print('有监督学习HMM模型的分词结果：')
    model = Segmentation(model='S')
    result = model.cut('１９９７年，是中国发展历史上非常重要的很不平凡的一年。')
    print('\t',result)
    print()
    result = model.cut('据环球时报了解，特朗普总统星期五会见刘鹤副总理时说，对美墨加达成贸易协定，'
                       '市场没什么反应，但是美中谈判一有积极进展，股市立刻上涨，市场反响强烈，这一次又是这样。')
    print('\t', result)
    print()
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print('基于字典最大正向匹配的分词结果：')
    model = Segmentation(model='D')
    result = model.cut('１９９７年，是中国发展历史上非常重要的很不平凡的一年。')
    print('\t',result)
    print()
    result = model.cut('据环球时报了解，特朗普总统星期五会见刘鹤副总理时说，对美墨加达成贸易协定，'
                       '市场没什么反应，但是美中谈判一有积极进展，股市立刻上涨，市场反响强烈，这一次又是这样。')
    print('\t', result)