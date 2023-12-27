#!/usr/bin/python
# -*- coding: UTF-8 -*-
import logging

from flask import Blueprint, render_template, request, session

from server.const import CONFIG
from utils.utils import load_params, get_IP

_CONF = load_params(CONFIG)
logger = logging.getLogger(__name__)

app = Blueprint('login', __name__, url_prefix='/login')

"""
此controller仅用于html页面导航，真正的各种action操作，都转移到controller_api里面去了（json api）
"""


@app.route('/', methods=["GET"])
def login_page():
    return render_template('login.html')


@app.route('/', methods=["POST"])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == _CONF.username and password == _CONF.password:
        logger.info("用户登录成功")
        session['username'] = username
        return render_template('/index.html')
    else:
        ip = get_IP()
        logger.warning("用户登录失败,username:%s,password:%s,IP:%s", username, password, ip)
        return "用户登录失败,username:%s,password:%s,IP:%s" % (username, password, ip)
