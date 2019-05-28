# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from news_spider.Mysql import MySql
from news_spider.MyRedis import MyRedis


class ShopeePipeline(object):

    myRedis = MyRedis()
    mysql = MySql()
    tablename='news'

    def process_item(self,item,tablename):
        try:
            # print("*********************")
            # print(item)
            # print("#####################")
            self.mysql.insert_db(self.tablename,item)

            self.myRedis.insert_redis(item['url'])
        except Exception as e:
            print("异常--", e)
        return item


