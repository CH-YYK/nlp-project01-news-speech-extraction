# -*- coding: utf-8 -*-

#@author:yangsong
#此处为预处理文件，包括语言模型的生成、相关近似词语提取

import os
import jieba
from functools import lru_cache
from gensim.models import Word2Vec,KeyedVectors
from  collections import defaultdict
import multiprocessing
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
path='/home/yangsong/nlp-project/01/mini_model/mini.model'
source_path='/home/yangsong/nlp-project/01/mini_source'
#获取相似词语，
def similars_bfs(word):
    unseen=[word]
    seen=defaultdict(int)
    max_size = 1000
    while unseen and len(seen)<max_size:
        word=unseen.pop(0)
        similars=get_similar_word(word)
        words=[k  for k,v in similars]
        unseen+=words
        seen[word]+=1
    sorted(seen.items(), key=lambda x: x[1], reverse=True)
    return seen

@lru_cache(None)
def  get_similar_word(word):
    return model.most_similar([word])
class MySentences(object):
    # 文件迭代
    def __init__(self, dirname):
        self.dirname = dirname

    # 迭代器
    def __iter__(self):
        words=[]
        for fname in os.listdir(self.dirname,):
            for line in open(os.path.join(self.dirname, fname),encoding='UTF-8'):
                for l in line.split('。'):
                    if l.split():
                        yield l.split()

#生成语言模型
def get_word2vec(word,path,source_path):
    sentence=MySentences(source_path)
    model = Word2Vec(sentence,size=256,window=5,min_count=3,workers=multiprocessing.cpu_count())
    # model.wv.save_Word2Vec_format(path,binary=True)
    model.save(path) #保存模型
    return model.wv.most_similar([word])
a=get_word2vec('说',path,source_path)
print(a)
# sims=similars_bfs('说')
# tmp_path='/home/yangsong/sims'
# tmp_list=[]
# for k,v in sims.items():
#     min_frequency=10
#     if v>min_frequency:
#         tmp_list.append(k)
# print(tmp_list)

# with open(tmp_path,'a+') as f:
#     for k,v in sims.items():
#         min_frequency=5
#         if v>min_frequency:
#             insert = "insert into word_sim(origin_word, sim_word) values('说',"+k+');'+'\n'
#             f.write(insert)