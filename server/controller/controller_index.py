#!/usr/bin/python
# -*- coding: UTF-8 -*-
import logging

from flask import Blueprint, render_template

from server.const import CONFIG
from utils import data_loader
from utils.plot import plot
from utils.utils import load_params

logger = logging.getLogger(__name__)
_CONF = load_params(CONFIG)

app = Blueprint('index', __name__, url_prefix='/')

"""
首页
"""


@app.route('/', methods=["GET"])
def login_page():
    logger.debug("访问首页！")
    df = data_loader.load_accounts(_CONF)
    df_baselines = [data_loader.load_index(code) for code in _CONF.baseline]
    plot(df,df_baselines, _CONF)
    return render_template('index.html')