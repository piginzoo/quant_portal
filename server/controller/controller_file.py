#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import logging
import os.path

from flask import Blueprint, request

from utils.utils import load_params, create_dir

logger = logging.getLogger(__name__)

app = Blueprint('file', __name__, url_prefix="/file")

"""
此controller仅用于文件上传
"""
_CONF = load_params('conf/config.yml')

# /file
@app.route('/', methods=["POST"])
def query():
    if request.method == 'POST':
        file = request.files['file']
        # 文件名
        file_name = file.filename
        create_dir(_CONF.data_dir)
        file_path = os.path.join(_CONF.data_dir,file_name)
        # 文件写入磁盘
        file.save(file_path)
        logger.info("上传文件被保存到：%s",file_path)

        # 将结果返回客户端
        resp = {"filename": file_name}
        return json.dumps(resp)