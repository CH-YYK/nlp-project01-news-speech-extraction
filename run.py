# -*- coding: utf-8 -*-

#@author:yangsong
# from flask import Flask
# from flask import render_template
# app = Flask(__name__)
#
# # from flask import Flask
# # from flask_bootstrap import Bootstrap
# #
# # app = Flask(__name__)
# # bootstrap = Bootstrap(app)
# app.logger.debug('A value for debugging')
# app.logger.warning('A warning occurred (%d apples)', 42)
# app.logger.error('An error occurred')
# @app.route('/hello/')
# @app.route('/hello/<name>')
# def hello(name=None):
#     if name==None:name='index.html'
#     return render_template(name, name=name)
from app import app
app.run(debug=True,port=8050)
# if __name__ == '__main__':
#     app.debug = True
#     app.run(host='localhost')
