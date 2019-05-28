# -*- coding: utf-8 -*-

#@author:yangsong
from flask import render_template
from app import app


@app.route('/index')
@app.route('/index/<name>')
def index(name=None):
    if name==None:name='index.html'
    return render_template(name, name=name)

@app.route('/about')
def about():
    return render_template('about.html')

