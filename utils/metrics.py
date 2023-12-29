import logging
import math

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from empyrical import calmar_ratio as _calmar_ratio
from empyrical import sortino_ratio as _sortino_ratio

from utils import utils, data_loader

logger = logging.getLogger(__name__)

# 可以使用LPR、1年期国债收益率、不过我还是更愿意用工行的1年期定存利率，更贴近现实：https://www.icbc.com.cn/column/1438058341686722587.html
RISK_FREE_ANNUALLY_RETRUN = 0.0155  # 这个随时更新，但是不会很频繁


def annually_profit(start_value, end_value, start_date, end_date):
    """
    细节：earn是一个百分比，不是收益/投入，而是终值/投入，这个是年化公式要求的，
    所以返回的时候，最终减去了1
    参考：https://www.pynote.net/archives/1667
    :param earn:
    :param broker:
    :return:
    """
    if type(start_date) == str:
        start_date = utils.str2date(start_date)
    if type(end_date) == str:
        end_date = utils.str2date(end_date)
    earn = end_value / start_value
    years = relativedelta(dt1=end_date, dt2=start_date).years
    months = relativedelta(dt1=end_date, dt2=start_date).months % 12
    days = relativedelta(dt1=end_date, dt2=start_date).days % 365
    years = years + months / 12 + days/365
    return earn ** (1 / years) - 1


def _daily_pct(value, period):
    """把不同时期的市值，转变成按照日计算的收益率"""
    # 如果是15分钟，每天4x4=16个15分钟
    if period == '15min':
        return value.pct_change(periods=16).dropna()
    # 如果是15分钟，每天4x12=48个5分钟
    if period == '5min':
        return value.pct_change(periods=48).dropna()
    if period == 'daily':
        return value.pct_change(periods=1).dropna()
    raise ValueError(f"未实现的周期{period}")


def sharp_ratio(value, period='daily'):
    """
    @period:  daily|15min|5min
    夏普比率 = 收益均值-无风险收益率 / 收益方差
    无风险收益率,在我国无风险收益率一般取值十年期国债收益
    https://rich01.com/what-sharpe-ratio/

                (每日報酬率平均值- 無風險利率) x (252平方根)
        夏普率= ----------------------------------------------
                    每日報酬的標準差

    一個好的策略，取任何一段時間的夏普率，數值不應該有巨大的落差
     (la.mean()- 0.0285/252)/la.std()*np.sqrt(252)

    我自己理解，夏普比率至少不能为负，至少也应该>0.5吧，>1应该就是一个合格的策略（或者是优秀的？）
    """
    pct = _daily_pct(value, period)
    return math.sqrt(250) * (pct.mean() - RISK_FREE_ANNUALLY_RETRUN / 250) / pct.std()


def sortino_ratio(value, period='daily'):
    """
    - https://xueqiu.com/4197676503/203091694
    - https://blog.csdn.net/The_Time_Runner/article/details/99569365
    - https://baike.baidu.com/item/%E7%B4%A2%E6%8F%90%E8%AF%BA%E6%AF%94%E7%8E%87/10776291
    Sortino Ratio = (年化收益率- 无风险利率) / 标准差(不良收益率)。

    索提诺比率是一种衡量投资组合相对表现的方法。与夏普比率(Sharpe Ratio)有相似之处，但索提诺比率运用下偏标准差而不是总标准差
    这一比率越高，表明基金承担相同单位下行风险能获得更高的超额回报率
    下偏标准差：
        在所有亏损的日子中，亏损的标准差，这个标准差越大，说明当出现亏损的时候，发生大幅度亏损的可能性越大。所以这个下行风险越小越好

    """
    pct = _daily_pct(value, period)
    return _sortino_ratio(pct)


def calmar_ratio(value, period='daily'):
    """
    https://www.zhihu.com/question/46517863
    卡玛比率 = 超额收益/最大回撤(风险)
    """
    # https://blog.csdn.net/The_Time_Runner/article/details/99569365
    pct = _daily_pct(value, period)
    cr = _calmar_ratio(pct)
    if np.isnan(cr): return -1
    return cr


def drawbacks(value, period='daily'):
    pct = _daily_pct(value, period)

    # 之前使用 emperial.max_drawdown(pct)，但是没有最大回撤期，现在改成下面的实现了
    # draw1 = max_drawdown(pct)
    # chatgpt写的，和emperial对比过，一致

    # 先算每天位置的累计收益率
    cumulative_returns = (1 + pct).cumprod().dropna()
    # 历史的最大收益：[1,2,3,4,3,2,1,5,4,3] ===> np.maximum.accumulate ==> [1,2,3,4,4,4,4,5,5,5]
    previous_peaks = np.maximum.accumulate(cumulative_returns)
    # 所有的回撤收益（负的）
    drawdowns = (cumulative_returns - previous_peaks) / previous_peaks

    return drawdowns, cumulative_returns


def drawback_duration(close, period='daily', min_days=10):
    """
    返回每段回撤的时间
    :param close:
    :param period:
    :param min_days:
    :return:
    """

    # 使用pandas.groupby分割 Series,这个写法太酷了，就是按照0分组
    drawdowns, cumulative_returns = drawbacks(close, period)
    # 以回撤为0，作为分割，分割成一段一段的，例：[-0.05,-0.15,0,-0.1,-0.25,0,-0.5] => [-0.05,-0.15], [0,-0.1,-0.25], [0,-0.5]
    segments = [group for _, group in drawdowns.groupby((drawdowns == 0).cumsum())]
    drawback_duration = []
    # 对每一个0分割的回撤分段，从这个分段里找出回撤最小值，就是回撤结束的日期，然后0开始的地方，就是回撤的开始
    for s in segments:
        drawback_start = s.index[0]
        drawback_end = s.idxmin()
        if pd.isnull(drawback_end): continue
        duration_days = utils.duration(drawback_start, drawback_end)
        # 如果回撤时间小于10天，就忽略，太短的回撤可以忍
        if duration_days<min_days: continue
        # 开始回撤时间~最大回撤时间
        drawback_duration.append([drawback_start,drawback_end,duration_days,s[drawback_end]])
    return pd.DataFrame(drawback_duration, columns=['start','end','days','drawback'])

def max_drawback(value, period='daily'):
    """
    value，是close或者市值
    最大回撤，https://www.yht7.com/news/30845
    """
    drawdowns, cumulative_returns = drawbacks(value, period)
    # 所有回撤里最大的1个
    _max_drawdown = drawdowns.min()
    # 最大回撤对应的结束日期，回撤值最大（负值绝对值最大，值最小idxmin)
    end_date = drawdowns.idxmin()
    # 最大回撤对应的开始日期，累计收益最大的那天
    start_date = cumulative_returns.loc[:end_date].idxmax()
    return _max_drawdown, start_date, end_date

def yearly_pct(s):
    """
    按照自然年，计算每年的收益
    """
    # 按年份分组
    grouped_by_year = s.groupby(pd.Grouper(freq='Y'))
    result = []
    for year, s1 in grouped_by_year:
        year_pct = (s1[-1] - s1[0])/s1[0]
        result.append([year.year,year_pct])
    return pd.DataFrame(result,columns=['year','pct'])




# python -m backtest.stat.metrics
if __name__ == '__main__':
    df = data_loader.load_stock('000002')
    df = df.iloc[:2000]

    # _, s, e = max_drawback(df.close, 'daily')
    # print(s, e, df.loc[s])
    # import matplotlib.pyplot as plt
    #
    # plt.plot(df.index, df.close)
    # plt.scatter(s, df.loc[s].close)
    # plt.scatter(e, df.loc[e].close)
    # plt.show()

    r = yearly_pct(df.close)
    print(r)