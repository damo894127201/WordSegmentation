# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 15:47
# @Author  : Weiyang
# @File    : discriminate_latent_state.py

#==============================================================================================================
# 基于测试数据，来确定无监督学习各个隐状态对应的具体状态，即将[0,1,2,3]与['B','M','E','S']对应起来
#==============================================================================================================

import numpy as np
from src.HMM import HMM

char2code = dict()  # 字符到编码的映射
code2char = dict()  # 编码到字符的映射
init_prob_dist = None  # 初始概率矩阵
state_trans_matrix = None  # 状态转移概率矩阵
emission_matrix = None  # 观测概率矩阵
latent_states = ['B','M','E','S'] # 隐状态集合

with open('../matrix/U_char_code.txt', 'r', encoding='utf-8') as fi:
    for line in fi:
        # 如果遇到:符号
        if line.count(':') > 1:
            _, _, code = line.strip().split(':')
            ch = ':'
            code = int(code)
        else:
            ch, code = line.strip().split(':')
            code = int(code)
        char2code[ch] = code
        code2char[code] = ch

matrix = np.load('../matrix/U_init_prob_dist.npz')
init_prob_dist = matrix['init_prob_dist']

matrix = np.load('../matrix/U_state_trans_matrix.npz')
state_trans_matrix = matrix['state_trans_matrix']

matrix = np.load('../matrix/U_emission_matrix.npz')
emission_matrix = matrix['emission_matrix']

model = HMM(num_latent_states=len(latent_states),
                        num_observation_states=len(char2code),
                        init_prob_dist=init_prob_dist,
                        state_trans_matrix=state_trans_matrix,
                        emission_matrix=emission_matrix)

inputs = ['１９９７年，是中国发展历史上非常重要的很不平凡的一年。',
          '据环球时报了解，特朗普总统星期五会见刘鹤副总理时说，对美墨加达成贸易协定，'
          '市场没什么反应，但是美中谈判一有积极进展，股市立刻上涨，市场反响强烈，这一次又是这样。']
for content in inputs:
    # 将输入数据，转为数字编码列表
    new_input = [char2code[ch] for ch in content]
    result = model.viterbi([new_input])
    print('文本：\t',list(content))
    print('结果：\t',result)
print(model.init_prob_dist)
print(model.state_trans_matrix)