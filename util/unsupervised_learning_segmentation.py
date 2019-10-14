# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 14:55
# @Author  : Weiyang
# @File    : unsupervised_learning_segmentation.py

#======================================================================================================================
# 无监督学习学习HMM模型参数：鲍姆-韦尔奇算法

# 对于分词来说，我们采用BMES标注体系，即B表示词的第一个字，M表示词的中间字，E表示词的最后一个字，S表示单个字的词

# 本模块输入../data中的数据，输出以下内容：

# 对于分词来说
# 1. 初始概率矩阵
# 2. 状态转移矩阵
# 3. 观测概率矩阵

# 但是对于通过无监督学习来获取HMM模型的三个参数矩阵而言，事先我们是无法知道，隐状态中哪个flag对应相应的['B','M','E','S']之一
# 因此，我们需要事后，即训练完毕HMM模型后，通过在测试集上预测结果，来最终确定隐状态具体对应的真实的flag
#======================================================================================================================

from HMM import HMM
import numpy as np
import time

start = time.time()

observation_states = [] # 存储字符,作为观测状态集合
observation_encode = dict() # 对观测状态进行编码，是按顺序进行数字编码
with open('../data/char_dict.txt','r',encoding='utf-8') as fi:
    temp = []
    for line in fi:
        temp += line.strip().split()
    observation_states = temp
    for code,obs in enumerate(observation_states):
        observation_encode[obs] = code

# 对观测序列进行编码
observation_state_sequences = []
with open('../data/word.txt','r',encoding='utf-8') as fi:
    # 遍历每行分词数据
    count = 1
    for line in fi:
        line = line.strip().split()
        # 对每个字符进行编码,生成编码序列
        state = []
        for ch in ''.join(line):
            # 对观测值进行编码
            state.append(observation_encode[ch])
        observation_state_sequences.append(state)

latent_states = ['B','M','E','S'] # 隐状态集合

# 创建模型
model = HMM(num_latent_states=len(latent_states), num_observation_states=len(observation_states))
# 训练初始概率矩阵、观测概率矩阵、转移概率矩阵
model.baum_welch(observation_state_sequences)

# 输出观测状态对应的编码，U表示这是由无监督学习学习到的参数
with open('../matrix/U_char_code.txt','w',encoding='utf-8') as fi:
    for code,obs in enumerate(observation_states):
        fi.write(obs + ':' + str(code) + '\n')

latent_states = ['B','M','E','S'] # 隐状态集合
# 输出隐状态对应的编码，U表示这是由无监督学习学习到的参数
with open('../matrix/U_latent_state_code.txt','w',encoding='utf-8') as fi:
    for code,state in enumerate(latent_states):
        fi.write(state + ':' + str(code) + '\n')

# 输出初始状态概率矩阵，S表示这是由无监督学习学习到的参数
np.savez('../matrix/U_init_prob_dist.npz',init_prob_dist=model.init_prob_dist)
# 输出状态转移概率矩阵，S表示这是由无监督学习学习到的参数
np.savez('../matrix/U_state_trans_matrix.npz',state_trans_matrix=model.state_trans_matrix)
# 输出观测概率矩阵，S表示这是由无监督学习学习到的参数
np.savez('../matrix/U_emission_matrix.npz',emission_matrix=model.emission_matrix)

end = time.time()
print('耗时: {} 分钟'.format((end - start)/60))