#!/usr/bin/python
# -*- coding: UTF-8 -*-
import logging

from flask import Blueprint, jsonify, request, session

from server import const
from utils import data_loader, stat
from utils.data_loader import load_trades, load_accounts, load_formated_raw_trades, load_formated_trades
from utils.utils import load_params, today, date2str, dataframe_to_dict_list, get_IP

logger = logging.getLogger(__name__)

app = Blueprint('api', __name__, url_prefix="/api")
_CONF = load_params(const.CONFIG)


@app.route('/', methods=["GET", "POST"])
def api():
    try:
        if session.get('username',None) is None:
            logger.warning("未登录的访问,ip:%s", get_IP())
            return "无效的访问", 400

        action = request.args.get('action', None)

        # /api?action=trade
        if action == 'trade':
            df = load_formated_trades(_CONF)
            return jsonify({
                'code': 0,
                'msg': 'ok',
                'title': action,
                'data': {
                    'title': '交易记录',
                    'type': 'table',
                    'data': dataframe_to_dict_list(df)
                }
            }), 200

        # /api?action=raw_trade
        if action == 'raw_trade':
            df = load_formated_raw_trades(_CONF)
            return jsonify({
                'code': 0,
                'msg': 'ok',
                'title': action,
                'data': {
                    'title': '原始交易记录',
                    'type': 'table',
                    'data': dataframe_to_dict_list(df)
                }
            }), 200

        # /api?action=account
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
                        '当前日期': date2str(s.date),
                        '持有现金': int(s.cash),
                        '仓位市值': int(s.total_position_value),
                        '总资产额': int(s.total_value)
                    }
                }
            }), 200

        # /api?action=trade_stat
        if action == 'trade_stat':
            # account_id,cash,total_value,total_position_value,date
            df = data_loader.load_trades(_CONF)
            s = stat.stat_trade(df)
            logger.debug("交易统计：%r",s)
            return jsonify({
                'code': 0,
                'msg': 'ok',
                'title': action,
                'data': {
                    'title': '交易统计',
                    'type': 'dict',
                    'data':s
                }
            }), 200


        # /api?action=market_value_stat
        if action == 'market_value_stat':
            # account_id,cash,total_value,total_position_value,date
            df = data_loader.load_accounts(_CONF)
            df_baselines = [data_loader.load_index(code) for code in _CONF.baseline]
            s = stat.stat_market_value(df,df_baselines)
            logger.debug("市值统计：%r", s)
            return jsonify({
                'code': 0,
                'msg': 'ok',
                'title': action,
                'data': {
                    'title': '市值统计',
                    'type': 'dict',
                    'data':s
                }
            }), 200


        logger.error("无效的访问参数：%r", request.args.get)
        return jsonify({'code': -1, 'msg': f'Invalid request:{request.args}'}), 500

    except Exception as e:
        logger.exception("处理过程中出现问题：%r", e)
        return jsonify({'code': -1, 'msg': f'Exception happened: {str(e)}'}), 500
