import controller as controller
import Log.logutil2 as log

from flask import Flask
from flask import request
from flask import jsonify
from flask import session
from flask import redirect
from flask import url_for
from flask import escape
from flask_cors import CORS

import os
import sys
import json
import random

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['JSON_SORT_KEYS'] = False

# session
# 随机生成SECRET_KEY
app.config['SECRET_KEY'] = os.urandom(24)

# 启动日志服务
logger = log.logs()

'''
# 强制 指定session时间，如果不指定则关闭浏览器自动清除
session.permanent = True
# session 删除时间 15 mins
app.permanent_session_lifetime = timedelta(minutes = 15) 
'''


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    根目录-首页，无论任何方法都可以返回一个英文的测试json
    """
    # 首页
    return jsonify({
        "index": "There is no index page, just the json."
    })



if __name__ == '__main__':
    # 启动Flask服务
    app.run(debug=True, host='127.0.0.1', port=5000)  # 内部测试
    # app.run(debug=False, host='0.0.0.0', port=80)  # 外部访问
