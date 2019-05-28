# -*- coding: utf-8 -*-

#@author:yangsong
import redis
import pickle
from gensim.models import Word2Vec
class Redis:
    @staticmethod
    def connect(host='127.0.0.1', port=6379, db=0):
        r = redis.StrictRedis(host, port, db)
        return r

    # # 将内存数据二进制通过序列号转为文本流，再存入redis
    # @staticmethod
    # def set_data(r, key, data, ex=None):
    #     r.set(pickle.dumps(key), pickle.dumps(data), ex)
    #
    # # 将文本流从redis中读取并反序列化，返回
    # @staticmethod
    # def get_data(r, key):
    #     data = r.get(pickle.dumps(key))
    #     if data is None:
    #         return None
    #
    #     return pickle.loads(data)


#缓存语言模型至redis
# def cache_model():
#     print('begin to cache...')
#     path = '/home/yangsong/nlp-project/01/model/word.model'
#     model = Word2Vec.load(path)
#     r=redis.Redis.connect()
#     redis.Redis.set_data(r,'model',model)
#     print('cached success!')

# cache_model()
if __name__=='main':
    print('begin to cache...')
    path = '/home/yangsong/nlp-project/01/model/word.model'
    model = Word2Vec.load(path)
    r = Redis.connect()
    Redis.connect(r, 'model', model)
    print('cached success!')
