# -*- coding: utf-8 -*-

#@author:yangsong
import pymysql
import os
from PIL import Image
import numpy as np
import re
from wordcloud import WordCloud
#db = pymysql.connect("localhost", 'root', '', 'nlp01', charset='utf8')
db = pymysql.connect("cdb-q1mnsxjb.gz.tencentcdb.com", 'root', 'A1@2019@me', 'DayandNight', 10102, charset='utf8')
cursor = db.cursor()
#从新闻表news中获取未更新的数据进行处理，将结果插入到word_sim表中

    # 更新词云图片
def update_wordcloud_img():

    sql="SELECT name_entity FROM word_sim"
    cursor.execute(sql)
    words=cursor.fetchall()
    words=' '.join([w[0] for w in words])
    d = os.getcwd()
    wc_mask = np.array(Image.open(os.path.join(d,"wc_mask.png")))
    # font_path = d.join("msyh.ttf")
    font_path = os.path.join(d,"msyh.ttf")
    wc = WordCloud(background_color="white",  # 设置背景颜色
                   collocations=False,
                   width=1000,
                   height=600,
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
    # print("正在进行word_sim表更新....")
    # word_sim()
    # print("word_sim表更新完毕！")
    print("正在更新词云图片...")
    update_wordcloud_img()
    print("词云图片更新完毕！")

