#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import logging
from json import JSONDecodeError

from flask import Blueprint, jsonify, request
from quant_trader.utils import utils

from server import const
from trade.trade_list import load_trades
from utils.utils import load_params

logger = logging.getLogger(__name__)

app = Blueprint('api', __name__, url_prefix="/api")
_CONF = load_params(const.CONFIG)


def request2json(request):
    try:
        json_data = request.get_data()

        if request.headers.get('content-type') == 'application/json' and len(json_data) > 0:
            logger.debug("接收到Json数据，长度：%d", len(json_data))
            data = json_data.decode('utf-8')
            data = data.replace('\r\n', '')
            data = data.replace('\n', '')
            if data.strip() == "": return {}
            data = json.loads(data)
            return data
        if len(request.form) > 0:
            logger.debug("接收到表单Form数据，长度：%d", len(request.form))
            return request.form
        return None
    except JSONDecodeError as e:
        logger.exception("JSon数据格式错误")
        raise Exception("JSon数据格式错误:" + str(e))


@app.route('/', methods=["GET", "POST"])
def api():
    try:
        # if flask.sessions['username'] is None:
        #     logger.warning("未登录的访问,ip:%s", get_IP())
        #     return "无效的访问", 400

        # params = request2json(request)

        action = request.args.get('action', None)

        # /api?action=trade
        if action == 'trade':
            """
            把服务器上的dataframe返回回去
            """
            logger.info("处理请求[%s]", action)

            df = load_trades(_CONF)

            return jsonify({
                'code': 0,
                'msg': 'ok',
                'title': action,
                'data': {
                    'title': '交易记录',
                    'type': 'table',
                    'data': utils.dataframe_to_dict_list(df)
                }
            }), 200

        logger.error("无效的访问参数：%r", request.args.get)
        return jsonify({'code': -1, 'msg': f'Invalid request:{request.args}'}), 200

    except Exception as e:
        logger.exception("处理过程中出现问题：%r", e)
        return jsonify({'code': -1, 'msg': f'Exception happened: {str(e)}'}), 200
