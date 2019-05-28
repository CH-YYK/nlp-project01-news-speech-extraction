# -*- coding: utf-8 -*-

#@author:yangsong
# from app import app
# from flask import current_app
# from flask import g
# ctx=app.app_context()
# ctx.push()
# from flask import g
# print(g.name)

total=[]
for i in range(10):
    num = {}
    num["nums"]=i
    total.append(num)
print(total)