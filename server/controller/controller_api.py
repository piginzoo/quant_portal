#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import logging
from json import JSONDecodeError

from flask import Blueprint, jsonify, request
from quant_trader.utils import utils

from server import const
from utils.data_loader import load_trades, load_accounts
from utils.utils import load_params, today

logger = logging.getLogger(__name__)

app = Blueprint('api', __name__, url_prefix="/api")
_CONF = load_params(const.CONFIG)


@app.route('/', methods=["GET", "POST"])
def api():
    try:
        # if flask.sessions['username'] is None:
        #     logger.warning("未登录的访问,ip:%s", get_IP())
        #     return "无效的访问", 400

        action = request.args.get('action', None)

        # /api?action=trade
        if action == 'trade':
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

        # /api?action=market_value
        if action == 'account':
            # account_id,cash,total_value,total_position_value,date
            df = load_accounts(_CONF)
            df1 = df[df.date==today()]
            s = df1.iloc[-1]
            return jsonify({
                'code': 0,
                'msg': 'ok',
                'title': action,
                'data': {
                    'title': '今日账户情况',
                    'type': 'dict',
                    'data': {
                        '当前日期': s.date,
                        '持有现金': int(s.cash),
                        '仓位市值': int(s.total_position_value),
                        '总资产额': int(s.total_value)
                    }
                }
            }), 200

        load_accounts

        logger.error("无效的访问参数：%r", request.args.get)
        return jsonify({'code': -1, 'msg': f'Invalid request:{request.args}'}), 200

    except Exception as e:
        logger.exception("处理过程中出现问题：%r", e)
        return jsonify({'code': -1, 'msg': f'Exception happened: {str(e)}'}), 200
