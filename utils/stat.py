import logging

import pandas as pd

from utils import metrics
from utils import utils
from utils.metrics import annually_profit
from utils.utils import date2str, duration, format_dict

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



def stat_market_value(df_account, df_baselines):
    df_account = df_account.set_index(df_account.date)

    # 计算各项指标
    stat = {}
    start_date = df_account.date[0]
    end_date = df_account.date[-1]
    stat["投资起始"] = date2str(start_date)
    stat["投资结束"] = date2str(end_date)

    if df_baselines is not None:
        for df_baseline in df_baselines:
            df_baseline = df_baseline[(df_baseline.date >= stat["投资起始"]) & (df_baseline.date <= stat["投资结束"])]
            code = df_baseline.iloc[0].code
            stat[f"基准{code}收益"] = float(df_baseline.iloc[0].close / df_baseline.iloc[-1].close - 1)
            stat[f"基准{code}年化"] = float(annually_profit(df_baseline.iloc[0].close,
                                                            df_baseline.iloc[-1].close,
                                                            df_baseline.iloc[0].date,
                                                            df_baseline.iloc[-1].date))

    stat["投资天数"] = duration(start_date, end_date)

    # account_id,cash,total_value,total_position_value,date
    stat["期初资金"] = float(df_account.iloc[0].total_value)
    stat["期末现金"] = float(df_account.iloc[-1].cash)
    stat["期末持仓"] = float(df_account.iloc[0].total_position_value)
    stat["期末总值"] = float(df_account.iloc[0].total_value)
    stat["组合赢利"] = float(df_account.iloc[-1].total_value - df_account.iloc[0].total_value)
    stat["组合收益"] = float(stat["组合赢利"] / stat["期初资金"])
    stat["组合年化"] = float(annually_profit(stat["期初资金"], stat["期末总值"], start_date, end_date))

    """
    接下来考察，仅投资用的现金的收益率，不考虑闲置资金了
    # 赢利 = 总卖出现金 + 持有市值 - 总投入现金 - 佣金
    """
    stat["夏普比率"] = float(metrics.sharp_ratio(df_account.total_value))
    # stat["索提诺比率"] = float(metrics.sortino_ratio(df_account.total_value))
    stat["卡玛比率"] = float(metrics.calmar_ratio(df_account.total_value))
    _drawback, draw_start, draw_end = metrics.max_drawback(df_account.total_value)
    stat["最大回撤"] = float(_drawback)
    stat["最大回撤开始"] = float(utils.date2str(draw_start))
    stat["最大回撤结束"] = float(utils.date2str(draw_end))
    stat["最大回撤天数"] = float(utils.duration(draw_start, draw_end))

    return format_dict(stat)


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
    if len(df_trade)==0: return {'交易情况':'无交易'}
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
    stat["连续盈利最大次数"] = int(positive_occur_max)
    stat["亏损次数"] = len(df_loss)
    stat["连续亏损最大次数"] = int(negtive_occur_max)
    stat["平均收益"] = float(win_avg_percent)
    stat["平均正收益"] = float(win_avg_percent)
    stat["平均负收益"] = float(loss_avg_percent)
    stat["胜率"] = round(len(df_win) / (len(df_win) + len(df_loss)), 4) if len(df_win) + len(df_loss) > 0 else 0
    stat["赢亏比"] = win_avg_amount / abs(loss_avg_amount) if loss_avg_amount else 0
    stat["期望"] = float(stat["胜率"] * stat["平均正收益"] + (1 - stat["胜率"]) * stat["平均负收益"])
    stat["最大收益"] = float(df_win.pnl.max())
    stat["最大收益/平均正收益"] = float(df_win.pnl.max() / win_avg_percent)
    stat["最小收益"] = float(df_loss.pnl.min())
    stat["最小收益/平均负收益"] = float(df_loss.pnl.min() / loss_avg_percent)
    stat["持仓均天"] = float(df_trade.days.mean())
    stat["持仓最长"] = int(df_trade.days.max())
    stat["持仓最短"] = int(df_trade.days.min())

    # float保留小数点3位
    return format_dict(stat)
