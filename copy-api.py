# -*- coding: utf-8 -*-

#@author:yangsong
from flask import Flask,request
from sqlalchemy import text
from app.controller.Model import Model
import time
import datetime
import json
from app import db
from app import app
from collections import defaultdict

# mysql_db=app.mysql_db
@app.route('/parse_sentence',methods=['GET','POST'])

#解析并返回句子
def parse_sentence():
    inserts=defaultdict(list)
    if request.method=='POST':
        sentence=request.form['sentence']
        user_ip = request.remote_addr
        sentence_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model=Model()
        name_says=model.sentence_process(sentence)
        ns_json=json.dumps(name_says, ensure_ascii=False)
        inserts['sentence']=sentence
        inserts['parse_sentence']=ns_json
        inserts['user_ip']=user_ip
        inserts['time']=sentence_time
        res = insert('history',inserts)
        if res:
            return ns_json
    return None
@app.route('/knowlege_graph',methods=['GET','POST'])
def knowlege_graph():
    res=select('word_sim')
    graph={}
    nodes=[]
    links=[]
    related_words=defaultdict(list)
    total_keywords=[]
    words=[]
    _keywords=[]
    for id, name_says, name_entity, url, category, content, says_entity in res:
        total_keywords.append(name_entity)
        name_entity=name_entity.split()
        _keywords+=name_entity
    total_keywords = [t.split() for t in total_keywords]

    for w in _keywords:
        for keywords in total_keywords:
            if w in keywords:
                for k in keywords:
                    if k!=w:
                        related_words[w].append(k)
    for r in related_words.keys():
            words.append(r)
            names={}
            names["name"]=r
            nodes.append(names)

    words = list(set(words))
    graph["nodes"] = nodes
    for k,v in enumerate(related_words.keys()):
        st={}
        st["source"]=k
        links.append(st)
        for t in total_keywords:
            if v in t:
                for x in t:
                    if x!=v:
                        index=words.index(x)
                        st["target"]=index
                        links.append(st)
    graph["links"]=links

    return json.dumps(graph, ensure_ascii=False)

#插入语句
def insert(tablename,*args):
    args=list(args)[0]
    print(args)
    columns=','.join([k for k,v in args.items()]) #列名称
    values=','.join(["'"+v+"'" for k,v in args.items()]) #拼接sql字符串
    sql='insert into '+tablename+' ('+columns+') values ('+values+');'
    print(sql)
    result=db.engine.execute(text(sql))
    result.close()
    return result.rowcount

#查询语句
def select(tablename):
    sql='select * from '+tablename
    result = db.engine.execute(text(sql))
    lists=result.fetchall()
    result.close()
    return lists


# knowlege_graph()


