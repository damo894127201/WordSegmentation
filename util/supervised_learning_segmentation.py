# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 9:04
# @Author  : Weiyang
# @File    : supervised_learning_segmentation.py

#==============================================================================================================
# 对于分词来说，我们采用BMES标注体系，即B表示词的第一个字，M表示词的中间字，E表示词的最后一个字，S表示单个字的词
# 本模块输入../data中的数据，输出以下内容：

# 对于分词来说
# 1. 初始概率矩阵
# 2. 状态转移矩阵
# 3. 观测概率矩阵
#==============================================================================================================

import numpy as np

observation_states = [] # 存储字符,作为观测状态集合
observation_encode = dict() # 对观测状态进行编码，是按顺序进行数字编码
with open('../data/char_dict.txt','r',encoding='utf-8') as fi:
    temp = []
    for line in fi:
        temp += line.strip().split()
    observation_states = temp
    for code,obs in enumerate(observation_states):
        observation_encode[obs] = code

latent_states = ['B','M','E','S'] # 隐状态集合
latent_state_encode = {'B':0,'M':1,'E':2,'S':3} # 对隐状态进行数字编码为[0,1,2,3]

# 初始化：统计初始概率矩阵、状态转移概率矩阵，观测概率矩阵
init_prob_dist = np.zeros((len(latent_states),)) # 初始概率矩阵
state_trans_matrix = np.zeros((len(latent_states),len(latent_states))) # 状态转移概率矩阵
emission_matrix = np.zeros((len(latent_states),len(observation_states))) # 观测概率矩阵

# 初始状态词典：{隐状态：次数}，表示隐状态处于序列开头的次数
init_state = {}

# 状态i 转移到 状态j 词典：{隐状态i->隐状态j:次数}
state_trans_state = {}
# 状态i 转移到其它状态 词典：{隐状态：次数}，表示由该状态转移到其它状态的次数
state_trans = {}

# 观测状态词典：{隐状态i->观测状态j:次数}
state_emission_observation = {}
# 隐状态生成观测状态的次数：{隐状态i:次数}，表示由隐状态i生成观测状态的次数
state_emission = {}

with open('../data/word.txt','r',encoding='utf-8') as fi:
    # 遍历每行分词数据
    count = 1
    for line in fi:
        line = line.strip().split()
        # 对每个字符进行标注,生成标注序列
        state = []
        for item in line:
            # 如果是单字符词
            if len(item) == 1:
                state.append('S')
            # 如果是双字符词
            elif len(item) == 2:
                state.append('B')
                state.append('E')
            # 如果是多字符词
            else:
                state.append('B')
                state += ['M'] * (len(item)-2)
                state.append('E')
        # 将分词数据转为字符序列
        chars = ''.join(line)
        # 统计数据
        prior_pos = '' # 存储前一个隐状态
        for i,(ch,pos) in enumerate(zip(chars,state)):
            # 如果是序列开头
            if i == 0:
                init_state[pos] = init_state.get(pos,0)
                init_state[pos] += 1
                state_emission_observation[pos + '->' + ch] = state_emission_observation.get(pos + '->' + ch,0)
                state_emission_observation[pos + '->' + ch] += 1
                state_emission[pos] = state_emission.get(pos,0)
                state_emission[pos] += 1
                prior_pos = pos
                continue
            # 如果不是序列开头
            state_trans_state[prior_pos + '->' + pos] = state_trans_state.get(prior_pos + '->' + pos,0)
            state_trans_state[prior_pos + '->' + pos] += 1
            state_trans[prior_pos] = state_trans.get(prior_pos,0)
            state_trans[prior_pos] += 1
            state_emission_observation[pos + '->' + ch] = state_emission_observation.get(pos + '->' + ch, 0)
            state_emission_observation[pos + '->' + ch] += 1
            state_emission[pos] = state_emission.get(pos, 0)
            state_emission[pos] += 1
            prior_pos = pos
        print(count)
        count += 1

# 计算：统计初始概率矩阵、状态转移概率矩阵，观测概率矩阵
count = sum(init_state.values())
for pos in init_state.keys():
    init_prob_dist[latent_state_encode[pos]] = init_state[pos] / float(count)

for pos1_pos2 in state_trans_state.keys():
    pos1,pos2 = pos1_pos2.split('->')
    state_trans_matrix[latent_state_encode[pos1],latent_state_encode[pos2]] = state_trans_state[pos1_pos2] / float(state_trans[pos1])

for pos_obs in state_emission_observation.keys():
    pos,obs = pos_obs.split('->')
    emission_matrix[latent_state_encode[pos],observation_encode[obs]] = state_emission_observation[pos_obs] / float(state_emission[pos])

# 输出观测状态对应的编码，S表示这是由监督学习学习到的参数
with open('../matrix/S_char_code.txt','w',encoding='utf-8') as fi:
    for code,obs in enumerate(observation_states):
        fi.write(obs + ':' + str(code) + '\n')

# 输出隐状态对应的编码，S表示这是由监督学习学习到的参数
with open('../matrix/S_latent_state_code.txt','w',encoding='utf-8') as fi:
    for code,state in enumerate(latent_states):
        fi.write(state + ':' + str(code) + '\n')

# 输出初始状态概率矩阵，S表示这是由监督学习学习到的参数
np.savez('../matrix/S_init_prob_dist.npz',init_prob_dist=init_prob_dist)
# 输出状态转移概率矩阵，S表示这是由监督学习学习到的参数
np.savez('../matrix/S_state_trans_matrix.npz',state_trans_matrix=state_trans_matrix)
# 输出观测概率矩阵，S表示这是由监督学习学习到的参数
np.savez('../matrix/S_emission_matrix.npz',emission_matrix=emission_matrix)