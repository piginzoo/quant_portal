import os
import akshare as ak

import pandas as pd

from server import const
from test import test_generate_data
from utils.utils import load_params, duration, init_logger, date2str
import logging

logger = logging.getLogger(__name__)


def load_accounts(conf):
    """
    加载市值信息:
    account_id,cash,total_value,total_position_value,date
    """
    path = os.path.join(conf.data_dir, "market_value.csv")
    if not os.path.exists(path):
        logger.warning("市值文件%s不存在", path)
        return None
    df = pd.read_csv(path, float_precision='round_trip', dtype={'date': str})  # ,encoding='gb2312')
    df['date'] = pd.to_datetime(df.date)
    return df


def load_index(index_code):
    """
    # https://www.akshare.xyz/data/index/index.html
    sh000001: 上证
    """
    df_stock_index = ak.stock_zh_index_daily(symbol=index_code)
    df_stock_index['date'] = pd.to_datetime(df_stock_index['date'], format='%Y-%m-%d')
    df_stock_index['code'] = index_code  # 都追加一个code字段
    return df_stock_index


def load_raw_trades(conf):
    """
    文件约定为：<日期>.trade.csv
    列：
    http://dict.thinktrader.net/nativeApi/xttrader.html?id=e2M5nZ#%E6%88%90%E4%BA%A4xttrade
    account_id,code,order_type,traded_id,traded_time,traded_price,
    traded_volume,traded_amount,order_id,order_sysid,strategy_name,order_remark
    """
    _files = []
    for root, dirs, files in os.walk(conf.data_dir):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        for f in files:
            if not 'trade' in f: continue
            path = os.path.join(root, f)
            df = pd.read_csv(path, float_precision='round_trip')  # ,encoding='gb2312')
            _files.append(df)
    df_trade = pd.concat(_files)

    # 临时测试代码，生成假交易，方便测试
    # df_trade = test_generate_data.generate_trades()
    # 交易时间格式是timestamp的，转成date：1703640608 =>2023-12-27 01:30:08
    # pd.set_option('display.width',5000)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('max_colwidth', 100)

    df_trade['traded_time'] = pd.to_datetime(df_trade['traded_time'], unit='s')
    df_trade.sort_values(by='traded_time', inplace=True,ascending=True)
    return df_trade

def load_formated_raw_trades(conf):
    df_trade = load_raw_trades(conf)
    df_trade.sort_values(by='traded_time', inplace=True, ascending=False) # 最新的在最前面，方便查看
    df_trade['order_type'] = df_trade.order_type.apply(lambda x: '买' if x==23 else '卖')
    df_trade['traded_time'] = df_trade.order_type.apply(lambda x: date2str(x))
    return df_trade[["code","order_type","traded_time","traded_price","traded_volume","traded_amount"]]

def load_trades(conf):
    df_trade  = load_raw_trades(conf)

    # 把QMT的df_trade，变成我们交易系统的trade，我们的是要有一个买，加上一个卖，才是一个完整的交易
    # 实现方法是，从上向下扫描表，遇到一个未处理的买，就去后面找他的对应的卖，2个合成一个trade，并标示为已处理，
    # 遇到已处理的忽略，直到处理完全表
    """
    qmt的交易：
    code,order_type,traded_id,traded_time,traded_price,traded_volume,traded_amount
    我们的交易：
    self.code = order.code  # 股票代码
    self.open_date = open_date
    self.close_date = None
    self.amount = order.amount  # 买了多少钱
    self.position = order.position  # 买了多少份
    self.pnl = None  # 利润率
    self.profit = None  # 利润额
    self.days = 0
    self.orders = []  # 用于记录历史的订单
    self.status = status
    """
    our_trades = []
    df_trade['status'] = 'pending'

    df_trade.reset_index(inplace=True)
    for i, s in df_trade.iterrows():
        # code,order_type,traded_id,traded_time,traded_price,traded_volume,traded_amount
        if s.order_type == 23 and s.status == 'pending':
            # 找到买单了，然后从这个位置开始，向后去找卖单
            # logger.debug("开始查找%s,%s开始的交易", s.code, s.traded_time)
            pos = i + 1
            if pos > len(df_trade) - 1: break
            # 一直向后找，直到找到
            while not (df_trade.loc[pos].code == s.code and \
                       df_trade.loc[pos].order_type == 24 and \
                       df_trade.loc[pos].status == 'pending'):
                pos += 1
                if pos > len(df_trade) - 1: break
            if pos > len(df_trade) - 1:
                logger.debug("此条交易[%s,%s]未查找到卖出记录", s.code, s.traded_time)
                continue
            e = df_trade.iloc[pos]  # 结束日期
            df_trade.loc[pos, 'status'] = 'done'
            our_trades.append({
                'code': s.code,  # 股票代码
                'open_date': s.traded_time,
                'close_date': e.traded_time,
                'amount': s.traded_amount,  # 买了多少钱
                'position': s.traded_volume,  # 买了多少份
                'pnl': (e.traded_amount - s.traded_amount) / s.traded_amount,  # 利润率
                'profit': e.traded_amount - s.traded_amount,  # 利润额
                'days': duration(s.traded_time, e.traded_time, unit='day')
            })
            # logger.debug("创建一条交易：%s：%s~%s", s.code, date2str(s.traded_time), date2str(e.traded_time))

    df_new_trades = pd.DataFrame.from_records(our_trades)

    return df_new_trades


# python -m utils.data_loader
if __name__ == '__main__':
    init_logger()
    conf = load_params(const.CONFIG)
    print(load_trades(conf))
