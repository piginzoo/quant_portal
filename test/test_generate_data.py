"""造一些市值的假数据"""
# account_id,cash,total_value,total_position_value,date
# 55003329,19963138.44,19978195.44,15066.0,20231227
import datetime
import os.path
from random import randint

import pandas as pd
import akshare as ak


def generate_trades():
    """
    code,order_type,traded_id,traded_time,traded_price,traded_volume,traded_amount
    ----
    000622.SZ,23,1703640608,4.22,400,1688.0

    造一个假的交易单出来，方便测试
    有100只股票，这个从akshare上获得，
    然后100要现有买、后有卖，
    然后就可以挑日子把他们分不到不同的交易日里，

    :return:
    """

    if os.path.exists("debug/df_trade.csv"):
        return pd.read_csv("debug/df_trade.csv", dtype={'code': str, 'date': str})

    # 生成日期
    start_date = datetime.date(2022, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    days = end_date - start_date
    valid_date_list = [(start_date + datetime.timedelta(days=x)).strftime('%Y%m%d')
                       for x in range(days.days + 1)
                       if (start_date + datetime.timedelta(days=x)).isoweekday() <= 5]

    # 每天随机生成10个交易
    df = pd.DataFrame()
    for d in valid_date_list:
        # 随机生成每天多少个交易
        trade_num = randint(0, 10)
        # code,order_type,traded_id,traded_time,traded_price,traded_volume,traded_amount
        _data = {'code': '',
                 'order_type': 0,
                 'traded_id': randint(100000, 200000),
                 'traded_time': d,
                 'traded_price': randint(5, 20),
                 'traded_volume': randint(500, 3000),
                 'traded_amount': -1}
        # 生成一个空交易的占位，后面会用
        df = df.append([_data] * trade_num, ignore_index=True)
    print("交易记录：", len(df))

    # 生成所有的1只股票的配对交易
    df_stock_list = ak.stock_info_sz_name_code(indicator='A股列表')
    df_stock_list = df_stock_list.sample(frac=1)[:200]  # 挑200只出来
    df_stock_list.sort_values(by='A股代码', inplace=True)
    df_stock_list.reset_index(inplace=True)
    df_stock_list = df_stock_list[['A股代码']]
    df_stock_list.rename(columns={'A股代码': 'code'}, inplace=True)
    df_stock_list['order_type'] = 0
    df_stock_list.loc[::2, 'order_type'] = 23  # 买，奇数行
    df_stock_list.loc[1::2, 'order_type'] = 24  # 卖，偶数行
    print("股票记录：", len(df_stock_list))

    # 遍历股票，插入到日期交易中，插2次，1次买，1次卖
    for i in range(len(df_stock_list)):

        # 把这些股票两两拿出，放到那些日期的占位符里，但是一定是后一条比前一条，日期要靠后
        if i == len(df_stock_list) - 1: continue

        s = df_stock_list.iloc[i]
        pos = randint(0, len(df) - 100)  # 先挑买位置
        while df.iloc[pos].code != '':  # 如果被占了，就往后挪
            pos += 1
        df.loc[pos, 'code'] = s.code
        df.loc[pos, 'order_type'] = 23 # 买
        start_date = df.loc[pos].traded_time

        # 接下来，存放卖出交易
        next_pos = pos + randint(1, 5)  # 再找卖位置，至少1~5天后
        # 如果被占了，就往后挪；如果日期相同，也往后挪
        while df.loc[next_pos].code != '' or df.loc[next_pos].traded_time==start_date:
            next_pos += 1
        df.loc[next_pos, 'code'] = s.code
        df.loc[next_pos, 'order_type'] = 24 # 卖
        if start_date > df.loc[next_pos].traded_time:
            print("pos,next_pos:",start_date,df.loc[next_pos].traded_time)

    df.drop(df[df.code == ''].index, inplace=True)
    df['traded_time'] = pd.to_datetime(df['traded_time'], format='%Y%m%d').astype(int).div(10 ** 9).astype(int)
    df['traded_amount'] = df.traded_price * df.traded_volume
    print(df.iloc[:100])
    print('最终交易记录：', len(df))
    df.to_csv('debug/df_trade.csv', index=False)
    return df


def generate_accounts():
    # isoweekday: Monday is 1 and Sunday is 7
    start_date = datetime.date(2022, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    days = end_date - start_date
    valid_date_list = {(start_date + datetime.timedelta(days=x)).strftime('%Y%m%d')
                       for x in range(days.days + 1)
                       if (start_date + datetime.timedelta(days=x)).isoweekday() <= 5}
    data = []
    for d in valid_date_list:
        data.append([55003329, randint(10000, 50000), randint(-10000, 10000), randint(1000, 3000), d])
    df = pd.DataFrame(data, columns=["account_id", "cash", "total_value", "total_position_value", "date"])
    df.sort_values(by='date', ascending=True, inplace=True)
    df['total_value'] = df.total_value.cumsum()
    print(df)
    df.to_csv('data/market_value.csv', index=False)


# python -m test.test_generate_data
if __name__ == '__main__':
    generate_trades()
