# -*- coding: utf-8 -*-
import redis
from news_spider.Mysql import MySql


class MyRedis:
    redis_db = redis.Redis(host='127.0.0.1', port=6379, db=0)
    redis_data_dict = "itemid"
    table='news'
    mySql = MySql()

    def __init__(self):
        pass


    def cache_redis(self):
        #缓存MySQL数据库中url至redis中
        dbnum = self.mySql.select_num_db(self.table)[0]
        renum = self.redis_db.scard('itemid')
        if dbnum != renum:
            self.redis_db.flushdb()
            if self.redis_db.hlen(self.redis_data_dict) == 0:  #
                datas = self.mySql.select(self.table)
                for id,url,content,source,cate in datas:
                    self.redis_db.sadd(self.redis_data_dict, url)
        print('mysql数据已存入redis缓存....')



    def is_exit(self, itemId):
        if self.redis_db.sismember(self.redis_data_dict, itemId):
            print(itemId, ':---exists')
            return False
        else:
            return True

    def insert_redis(self, item_id):
        self.redis_db.sadd(self.redis_data_dict, item_id)

    def __del__(self):
        pass
