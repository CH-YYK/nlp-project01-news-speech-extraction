# -*- coding: utf-8 -*-
import pymysql
from news_spider import settings


class MySql(object):

    def __init__(self):
        self.db = pymysql.connect("cdb-q1mnsxjb.gz.tencentcdb.com", settings.DB_USERNAME, settings.DB_PASSWORD, settings.DB_NAME, charset='utf8')
        self.cursor = self.db.cursor()

    def insert_db(self,tablename,*args):
        arg=args[0]
        columns = ','.join([k for k, v in arg.items()])  # 列名称
        values = ','.join([ "'"+v+"'" for k, v in arg.items()])  # 拼接sql字符串
        sql = "insert into "+tablename+' ('+columns+') values ('+values+');'
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise

    def select(self,tablename):
        sql = "SELECT * FROM "+tablename
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        return result

    def select_num_db(self,tablename):
        sql = "SELECT COUNT(id) FROM "+tablename
        self.cursor.execute(sql)
        result = self.cursor.fetchone()
        return result

    def __del__(self):
        self.db.close()
        pass