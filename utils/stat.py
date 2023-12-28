import os
from collections import OrderedDict
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame
from tabulate import tabulate

from utils import metrics
from utils.metrics import annually_profit
from utils import data_loader, utils
from utils.utils import date2str, str2date

logger = logging.getLogger(__name__)
"""
统计显示一下信息：
     基准指数 : sh000001
     投资起始 : 20190102
     投资结束 : 20230616
     投资天数 : 1083
     定投起始 : 20190222
     定投结束 : 20230616
     期初资金 : 200000
     期末现金 : 452273.7250400002
     期末持仓 : 0.0
     期末总值 : 452273.7250400002
     组合赢利 : 232751.4500800002
     组合收益 : 1.163757250400001
     组合年化 : 0.16692167634955224
     基准收益 : 0.32776779698623804
     基准年化 : 0.05833808858727685
     夏普比率 : 0.47920245545936047
    索提诺比率 : 12.45
     卡玛比率 : 0.21649800353588733
     最大回撤 : -0.97
       佣金 : 19522.274960000002
     股票收益 : 0.5985990110666353
     股票年化 : 0.098368091993293
     股票代码 : 002439
       成本 : -1
       持仓 : -1
       现价 : 169.88
       买次 : 44
       卖次 : 44
       赢次 : 24
       输次 : 20
       胜率 : 0.55
       赢均 : 16081.57
       输均 : -5859.75
      赢亏比 : 2.74
     借钱次数 : N/A
     借钱总额 : N/A
"""


def calculate_stat(df_baselines, broker, params, df_stock=None):
    start_date = str2date(params.start_date)
    end_date = str2date(params.end_date)

    df_portfolio = broker.get_total_values()

    # 计算各项指标
    stat = OrderedDict()

    if df_baselines is not None:
        for df_baseline in df_baselines:
            start_value = df_baseline.iloc[0].close
            end_value = df_baseline.iloc[-1].close
            baseline_code = df_baseline.iloc[0].code
            stat[f"基准{baseline_code}收益"] = end_value / start_value - 1
            stat[f"基准{baseline_code}年化"] = annually_profit(start_value, end_value, start_date, end_date)

    stat["投资起始"] = date2str(df_portfolio.index.min())
    stat["投资结束"] = date2str(df_portfolio.index.max())
    stat["投资天数"] = len(df_portfolio)

    # 如果有无限投资额，就用累计投资额，否则，就用初始现金
    start_value = broker.banker.debt if broker.banker else params.cash
    end_value = broker.get_total_value() - broker.total_commission

    stat["期初资金"] = start_value
    stat["期末现金"] = broker.total_cash
    stat["期末持仓"] = broker.get_total_position_value()
    stat["期末总值"] = broker.get_total_value()
    stat["组合赢利"] = end_value - start_value
    stat["组合收益"] = end_value / start_value - 1
    stat["组合年化"] = annually_profit(start_value, end_value, start_date, end_date)

    """
    接下来考察，仅投资用的现金的收益率，不考虑闲置资金了
    # 赢利 = 总卖出现金 + 持有市值 - 总投入现金 - 佣金
    """
    stat["夏普比率"] = metrics.sharp_ratio(df_portfolio.total_value, params.period)
    stat["索提诺比率"] = metrics.sortino_ratio(df_portfolio.total_value, params.period)
    stat["卡玛比率"] = metrics.calmar_ratio(df_portfolio.total_value, params.period)
    _drawback,draw_start,draw_end = metrics.max_drawback(df_portfolio.total_value, params.period)
    stat["最大回撤"] = _drawback
    stat["最大回撤开始"] = utils.date2str(draw_start)
    stat["最大回撤结束"] = utils.date2str(draw_end)
    stat["最大回撤天数"] = utils.duration(draw_start,draw_end)
    stat["佣金"] = broker.total_commission

    if df_stock is not None:
        code = df_stock.iloc[0].code
        start_value = df_stock.iloc[0].close
        end_value = df_stock.iloc[-1].close
        stat['最新价格'] = end_value
        stat["股票收益"] = end_value / start_value - 1
        stat["股票年化"] = annually_profit(start_value, end_value, start_date, end_date)
        stat["股票代码"] = code
        stat["成本"] = -1 if broker.positions.get(code, None) is None else broker.positions[code].cost
        stat["持仓"] = -1 if broker.positions.get(code, None) is None else broker.positions[code].position

    if broker.banker is not None:
        stat["借钱次数"] = broker.banker.debt_num
        stat["借钱总额"] = broker.banker.debt
        stat["借钱次数"] = broker.banker.debt_num

    df_trade = broker.get_trade_history()
    _stat_trade = stat_trade(df_trade)
    stat.update(_stat_trade)

    return stat


def stat_trade(df_trade, start=None, end=None):
    """
    统计交易df_trade情况，做以下统计
    - 交易次数
    - 赢次数
    - 输次数
    - 胜率
    - 盈亏比
    - 平均收益
    - 平均正收益
    - 平均负收益
    - 最大收益
    - 最大收益与平均正收益比
    - 最小收益占正收益比
    - 最大收益与平均负收益比
    - 连续正收益次数
    - 连续负收益次数
    """

    if start:
        df_trade = df_trade[(df_trade.open_date > start) & (df_trade.open_date < end)]
    df_win = df_trade[df_trade.profit > 0]
    df_loss = df_trade[df_trade.profit <= 0]
    win_avg_amount = df_win.profit.sum() / len(df_win) if len(df_win) > 0 else 0
    loss_avg_amount = df_loss.profit.sum() / len(df_loss) if len(df_loss) > 0 else 0
    win_avg_percent = df_win.pnl.mean()
    loss_avg_percent = df_loss.pnl.mean()

    # 计算连续出现盈利和损失的最大次数
    # test = [1.12, 212.1, 1, -131.1, -22, -33, 44, 55, -66,-66,1,1,2,2,2,2,2,2,2,2,2,-1,-1]
    # ((~(test>0)).cumsum()).groupby((~(test>0)).cumsum()).count().max() - 1
    # 结果：11
    p1 = (~(df_trade.profit > 0)).cumsum()
    positive_occur_max = p1.groupby(p1).count().max()
    p1 = (~(df_trade.profit < 0)).cumsum()
    negtive_occur_max = p1.groupby(p1).count().max()

    stat = {}
    stat["交易次数"] = len(df_trade)
    stat["盈利次数"] = len(df_win)
    stat["连续盈利最大次数"] = positive_occur_max
    stat["亏损次数"] = len(df_loss)
    stat["连续亏损最大次数"] = negtive_occur_max
    stat["平均收益"] = win_avg_percent
    stat["平均正收益"] = win_avg_percent
    stat["平均负收益"] = loss_avg_percent
    stat["胜率"] = round(len(df_win) / (len(df_win) + len(df_loss)), 4) if len(df_win) + len(df_loss) > 0 else 0
    stat["赢亏比"] = win_avg_amount / abs(loss_avg_amount) if loss_avg_amount else 0
    stat["期望"] = stat["胜率"] * stat["平均正收益"] + (1 - stat["胜率"]) * stat["平均负收益"]
    stat["最大收益"] = df_win.pnl.max()
    stat["最大收益/平均正收益"] = df_win.pnl.max() / win_avg_percent
    stat["最小收益"] = df_loss.pnl.min()
    stat["最小收益/平均负收益"] = df_loss.pnl.min() / loss_avg_percent
    stat["持仓均天"] = df_trade.days.mean()
    stat["持仓最长"] = df_trade.days.max()
    stat["持仓最短"] = df_trade.days.min()

    return stat


def show(df_baselines, broker, params, df=None):
    """
    用来统计一只股票的统计结果
    """

    df_stat = DataFrame()
    df_trade = broker.get_trade_history()

    start_date = params.start_date
    end_date = params.end_date

    code = 'summary' if df is None else df.iloc[0].code

    # 统计这只基金的收益情况
    stat = calculate_stat(df_baselines, broker, params, df)

    df_stat = df_stat.append(stat, ignore_index=True)

    log_stat_and_summary(stat, f'股票{code}的统计信息')

    stat_by_period(df_trade)

    if len(df_stat) == 0:
        return df_stat

    stat_file_name = os.path.join(params.debug_dir, f"{code}_{start_date}_{end_date}_stat.csv")
    trade_file_name = os.path.join(params.debug_dir, f"{code}_{start_date}_{end_date}_trade.csv")

    # 把统计结果df_stat写入到csv文件
    # logger.info("交易统计：")
    # with pd.option_context('display.max_rows', 100, 'display.max_columns', 100):
    #     print(tabulate(df, headers='keys', tablefmt='psql'))
    df_stat.to_csv(stat_file_name)


    # 打印交易记录
    logger.info("交易记录：")
    # 暂时先注释了，不打印到屏幕上了
    # print(tabulate(df_order, headers='keys', tablefmt='psql'))
    df_trade.to_csv(trade_file_name)

    write_stat_and_summary_file(f'{code}.txt', params, stat, f'股票{code}的回测统计信息')

    # 打印期末持仓情况
    # logger.info("期末持仓：")
    # df = DataFrame([p.to_dict() for code, p in broker.positions.items()])
    # print(tabulate(df, headers='keys', tablefmt='psql'))


    return df_stat


def stat_and_trade_summary(df_stat, df_trade, params):
    """
    用来统计多只股票汇总到一起的统计信息,
    这个是用于多进程把所有的独立的股票都跑一遍后的信息汇总，
    他不适合ConstrainBackTester和PoolBackTester，因为这两者，是可以计算组合的夏普、年化等组合信息的
    """

    # 回测统计
    summary = OrderedDict()
    # summary["股票"] = len(df_stocks)
    summary["开始"] = params.start_date
    summary["结束"] = params.end_date
    summary["期初"] = df_stat['期初资金'].sum()
    summary["期末"] = df_stat['期末总值'].sum()
    summary["佣金%"] = (df_stat['佣金'].sum() / df_stat['期初资金'].sum()) * 100
    summary["赢利%"] = (df_stat['期末总值'].sum() / df_stat['期初资金'].sum() - 1) * 100
    summary["持仓均天"] = df_stat['持仓均天'].mean()
    summary["年化%"] = 100 * annually_profit(df_stat['期初资金'].sum(),
                                             df_stat['期末总值'].sum(),
                                             params.start_date,
                                             params.end_date)

    # 统计交易
    _stat_trade = stat_trade(df_trade)
    summary.update(_stat_trade)

    log_stat_and_summary(summary, '所有交易的汇总信息')



def log_stat_and_summary(_dict,title):
    logger.debug(f"\n{'-' * 80}")
    logger.info(title)
    logger.info("-" * 80)
    for k, v in _dict.items():
        logger.info(f"\t{k}:\t{v}")
    logger.info("-" * 80)