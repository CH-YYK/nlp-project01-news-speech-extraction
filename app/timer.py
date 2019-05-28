# -*- coding: utf-8 -*-

#@author:yangsong
import pymysql
from app.controller.Model import Model
from collections import defaultdict
import os
import re
from PIL import Image
import numpy as np
import json
from wordcloud import WordCloud, STOPWORDS
wc_img_path='/home/yangsong'
db = pymysql.connect("localhost", 'root', '', 'nlp01', charset='utf8')
cursor = db.cursor()

#从新闻表news中获取未更新的数据进行处理，将结果插入到word_sim表中
def word_sim():
    cursor.execute("SELECT url FROM word_sim;")
    url_wordsim_res=cursor.fetchall() #获取已处理的新闻的url
    url_wordsim=[x[0] for x in url_wordsim_res]
    cursor.execute("SELECT * FROM news;")
    res_news = cursor.fetchall()
    obj=Model()
    inserts=defaultdict(list)
    total_content=[content for id,url,content,source,category in res_news] #所有文章
    for id,url,content,source,category in res_news:
        if url not in url_wordsim:
            print("正在处理第{}个，共计{}个".format(id,len(res_news)))
            name_says=obj.sentence_process(content)
            # print(name_says.values())
            name_entity=defaultdict(list)
            if not name_says:
                print("当前句子未获取到人物言论！")

            else:
                for name,says in name_says.items():
                    if says[0] is not None:
                        says=' '.join([' '.join(x) for x in says])
                        name_entity[name]=obj.get_news_ne(says)

            keywords=obj.get_keywords_of_a_ducment(content,total_content)
            inserts['url']=url
            inserts['name_says']=json.dumps(name_says,ensure_ascii=False)
            inserts['name_entity']=keywords
            inserts['says_entity']=json.dumps(name_entity,ensure_ascii=False)
            inserts['content']=content
            inserts['category']=category
            columns = ','.join(inserts.keys())  # 列名称
            values = ','.join(["'" + str(v) + "'" for  v in inserts.values()])  # 拼接sql字符串
            sql = 'insert into word_sim '+' (' + columns + ') values (' + values + ');'
            # print(sql)
            try:
                cursor.execute(sql)
                db.commit()  # 提交数据
            except Exception as e:
                print("**************************异常***************************", e)
        else:
            print('该条记录已处理')
    cursor.close()

    # 更新词云图片
def update_wordcloud_img():

    sql="SELECT name_entity FROM word_sim"
    cursor.execute(sql)
    words=cursor.fetchall()
    words=' '.join([re.sub(r'(新华社|新华网)','',''.join(w)) for w in words])
    d = os.getcwd()
    wc_mask = np.array(Image.open(os.path.join(d,"wc_mask.png")))
    # font_path = d.join("msyh.ttf")
    font_path = os.path.join(d,"msyh.ttf")
    wc = WordCloud(background_color="white",  # 设置背景颜色
                   collocations=False,
                   width=700,
                   height=530,
                   max_words=2000,  # 词云显示的最大词数
                   mask=wc_mask,  # 设置背景图片
                   font_path=font_path,  # 兼容中文字体，不然中文会显示乱码
                   random_state=30  # 设置有多少种随机生成状态，即有多少种配色方案
                   )
    # 生成词云
    wc.generate(words)

    # 生成的词云图像保存到本地
    wc.to_file(d+"/static/images/word_cloud.png")

    cursor.close()



#执行
if __name__ == "__main__":
    print("正在进行word_sim表更新....")
    word_sim()
    print("word_sim表更新完毕！")
    print("正在更新词云图片...")
    update_wordcloud_img()
    print("词云图片更新完毕！")

