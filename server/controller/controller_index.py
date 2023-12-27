#!/usr/bin/python
# -*- coding: UTF-8 -*-
import logging

from flask import Blueprint, render_template

from server.const import CONFIG
from utils.utils import load_params

_CONF = load_params(CONFIG)
logger = logging.getLogger(__name__)

app = Blueprint('index', __name__, url_prefix='/')

"""
首页
"""


@app.route('/', methods=["GET"])
def login_page():
    logger.debug("访问首页！")
    return render_template('index.html')