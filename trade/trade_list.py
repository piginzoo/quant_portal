import os

import pandas as pd

from server import const
from utils.utils import load_params
import logging

logger = logging.getLogger(__name__)

def load_trades(conf):
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
            path = os.path.join(root,f)
            df = pd.read_csv(path,float_precision='round_trip')#,encoding='gb2312')
            _files.append(df)
    df_trade = pd.concat(_files)
    # 交易时间格式是timestamp的，转成date：1703640608 =>2023-12-27 01:30:08
    df_trade['traded_time'] = pd.to_datetime(df_trade['traded_time'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    df_trade['order_type'] = df_trade.order_type.apply(lambda x: '买' if  x==23 else '卖')
    df_trade.sort_values(by='traded_time',ascending=False,inplace=True)

    # pd.set_option('display.width',5000)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('max_colwidth', 100)
    return df_trade[["code","order_type","traded_time","traded_price","traded_volume","traded_amount"]]

# python -m trade.trade_list
if __name__ == '__main__':

    conf = load_params(const.CONFIG)
    print(load_trades(conf))