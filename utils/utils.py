import calendar
import datetime
import functools
import json
import logging
import os
import sys
import time

import dask
import numpy as np
import pandas as pd
import psutil
import statsmodels.api as sm
import yaml
from dask import compute, delayed
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


class AttributeDict(dict):

    def __init__(self, _dict):
        """_dict是一个字典"""
        for k, v in _dict.items():
            if type(v) == dict:
                self.__setitem__(k, AttributeDict(v))
            else:
                self.__setitem__(k, v)

    """
    class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    data=AttributeDict()
    data['a']='a'
    pickle.dump(data, open("test.txt","wb"))
    # 程序报错：
    # KeyError: '__getstate__'
    # 如何修复？
    修复：必须从新定义__getstate__和__setstate__
    """
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    # __getattr__ = dict.__getitem__
    def __getattr__(self, name):
        """重载这个方法，是为了防止KeyError，而只返回None"""
        try:
            return self[name]
        except KeyError:
            return None

    def copy(self):
        """默认的copy会返回dict，我要的是AttributeDict"""
        return AttributeDict(super().copy())

def get_IP(request):
    if request.headers.get("X-Real-Ip"):
        ip = request.headers.get("X-Real-Ip")
    else:
        ip = request.remote_addr
    return ip

def dataframe_to_dict_list(df):
    """
    把dataframe变成一个每行都带着列名的json
    :param df:
    :return:
    """
    df = df.fillna(0) # 有nan转化成json，会解析报错
    data = df.values.tolist()
    columns = df.columns.tolist()
    return [
        dict(zip(columns, datum)) for datum in data
    ]

def get_value(df, index, col=None):
    try:
        if col is None: return df.loc[index]
        if col not in df.columns: return None
        return df.loc[index][col]
    except KeyError:
        return None


def get_yearly_duration(start_date, end_date):
    """
    把开始日期到结束日期，分割成每年的信息
    比如20210301~20220501 => [[20210301,20211231],[20220101,20220501]]
    """
    start_date = str2date(start_date)
    end_date = str2date(end_date)
    years = list(range(start_date.year, end_date.year + 1))
    scopes = [[f'{year}0101', f'{year}1231'] for year in years]

    if start_date.year == years[0]:
        scopes[0][0] = date2str(start_date)
    if end_date.year == years[-1]:
        scopes[-1][1] = date2str(end_date)

    return scopes


def now(level='second'):
    if level == 'millisecond':
        return datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S%f")
    else:
        return datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")


def now2str(level='second'):
    if level == 'millisecond':
        return datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S%f")
    else:
        return datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")


def date2str(date, format="%Y%m%d"):
    return datetime.datetime.strftime(date, format)


def datetime2str(date, format="%Y%m%d %H%M"):
    return datetime.datetime.strftime(date, format)


def duration(start, end, unit='day'):
    d0 = str2date(start) if type(start) == str else start
    d1 = str2date(end) if type(end) == str else end
    delta = d1 - d0
    if unit == 'day': return delta.days
    return None


def str2date(s_date, format="%Y%m%d"):
    return datetime.datetime.strptime(s_date, format)


def str2date(s_date, format="%Y%m%d"):
    return datetime.datetime.strptime(s_date, format)


def is_sameday(date1, date2):
    """
    只判断是不是一天，而忽略小时和分钟
    """
    date1 = datetime2date2datetime(date1)
    date2 = datetime2date2datetime(date2)
    return date1 == date2


def datetime2date2datetime(dt):
    """把datetime转成date，再转成datetime，但是都变成00:00
    2011-01-01 13:55 => 2011-01-01 0:0
    """
    return date2datetime(dt.date())


def dataframe_str2date(df, col, format="%Y-%m-%d"):
    df[col] = pd.to_datetime(df[col], format=format)


def split_periods(start_date, end_date, window_years, roll_stride_months):
    """
        用来生成一个日期滚动数组，每个元素是开始日期和结束日期，每隔一个周期向前滚动

        比如:split('20120605','20151215',window_years=2,roll_stride_months=3)
        2012-06-15 00:00:00 2014-06-15 00:00:00
        2012-09-15 00:00:00 2014-09-15 00:00:00
        2012-12-15 00:00:00 2014-12-15 00:00:00
        2013-03-15 00:00:00 2015-03-15 00:00:00
        2013-06-15 00:00:00 2015-06-15 00:00:00
        2013-09-15 00:00:00 2015-09-15 00:00:00
        2013-12-15 00:00:00 2015-12-15 00:00:00

        :param start_date:
        :param end_date:
        :param window_years:
        :param roll_stride_months:
        :return:
        """

    all_ranges = []

    # 第一个范围
    start_roll_date = start_date
    end_roll_date = start_date + relativedelta(years=window_years)
    if end_roll_date > end_date:
        end_roll_date = end_date

    all_ranges.append([date2str(start_roll_date),
                       date2str(end_roll_date)])

    # while滚动期间的结束日期end_roll_date，小于总的结束日期end_date
    # 滚动获取范围
    start_roll_date = start_roll_date + relativedelta(months=roll_stride_months)
    while end_roll_date < end_date:
        # 滚动
        end_roll_date = start_roll_date + relativedelta(years=window_years)

        if end_roll_date > end_date:
            end_roll_date = end_date

        all_ranges.append([date2str(start_roll_date),
                           date2str(end_roll_date)])

        start_roll_date = start_roll_date + relativedelta(months=roll_stride_months)

    return all_ranges


def fit(data_x, data_y):
    """
    https://blog.csdn.net/zzu_Flyer/article/details/107634620
    :param data_x:
    :param data_y:
    :return:
    """

    # print(data_x,data_y)

    m = len(data_y)
    x_bar = np.mean(data_x)
    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0
    for i in range(m):
        x = data_x[i]
        y = data_y[i]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    # 根据公式计算w
    w = sum_yx / (sum_x2 - m * (x_bar ** 2))

    for i in range(m):
        x = data_x[i]
        y = data_y[i]
        sum_delta += (y - w * x)
    b = sum_delta / m
    return w, b


def init_logger(file=False, simple=False, log_level=logging.DEBUG):
    print("开始初始化日志：file=%r, simple=%r" % (file, simple))

    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.colorbar').disabled = True
    logging.getLogger('matplotlib').disabled = True
    logging.getLogger('fontTools.ttLib.ttFont').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('asyncio').disabled = True

    if simple:
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d|%(levelname)s|%(filename)s:%(lineno)d: %(message)s',
                                      datefmt='%H:%M:%S')

    root_logger = logging.getLogger()
    root_logger.setLevel(level=log_level)

    def is_any_handler(handlers, cls):
        for t in handlers:
            if type(t) == cls: return True
        return False

    # 加入控制台
    if not is_any_handler(root_logger.handlers, logging.StreamHandler):
        stream_handler = logging.StreamHandler(sys.stdout)
        root_logger.addHandler(stream_handler)
        print("日志：创建控制台处理器")

    # 加入日志文件
    if file and not is_any_handler(root_logger.handlers, logging.FileHandler):
        if not os.path.exists("./logs"): os.makedirs("./logs")
        filename = "./logs/{}.log".format(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
        t_handler = logging.FileHandler(filename, encoding='utf-8')
        root_logger.addHandler(t_handler)
        print("日志：创建文件处理器", filename)

    handlers = root_logger.handlers
    for handler in handlers:
        handler.setLevel(level=log_level)
        handler.setFormatter(formatter)


def serialize(obj, file_path):
    # pickle是二进制的，不喜欢，改成json序列化了
    # f = open(file_path, 'wb')
    # pickle.dump(obj, f)
    # f.close()
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=2)


def unserialize(file_path):
    # f = open(file_path, 'rb')
    # obj = pickle.load(f)
    # f.close()
    with open(file_path, 'r') as f:
        obj = json.load(f)
    return obj


def is_process_running(process_name):
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'].lower() == process_name.lower():
            return True
    return False


def this_month():
    now = datetime.datetime.now()
    return datetime.datetime(now.year, now.month, 1)


def calc_size(cash, price, commission_rate):
    """
    用来计算可以购买的股数：
    1、刨除手续费
    2、要是100的整数倍
    为了保守起见，用涨停价格来买，这样可能会少买一些。
    之前我用当天的close价格来算size，如果不打富余，第二天价格上涨一些，都会导致购买失败。
    """

    # 按照一个保守价格来买入
    size = math.ceil(cash * (1 - commission_rate) / price)

    # 要是100的整数倍
    size = (size // 100) * 100
    return size


def calc_k(close, previous_days=None, index=None):
    """计算斜率，只需要股价
    把股价归一化，按照天数来归一化，
    比如价格是50~150，天数是20天，
    会把价格压缩到0~20之间，这样形成一个正方形，
    然后再算斜率
    """
    if index:
        # import pdb;pdb.set_trace()
        close = close[close.index <= index]

    if previous_days:
        if len(close) < previous_days: return None
        close = close[-previous_days:]

    X = list(range(len(close)))  # 变成0~len(close)的数字序列，比如20个close，就是[0,1,.....,19]
    Y = nomalize(close, len(close))  # 要把收盘价压缩到[0~len(close)]的范围，比如[0,20]

    # 防止Y里面有NAN
    if Y.count() != len(close): return None

    params, _ = OLS(X, Y)
    # logger.debug("%r\n%r\n%r", X, Y, params[1])
    return params[1]


def nomalize(s, scope):  # 标准化，s 序列，scope，标准化到的范围
    return (s - s.min()) * scope / (s.max() - s.min())


def unomalize(x, s, scope):  # 反标准化，s 序列，scope，标准化到的范围
    return (x * (s.max() - s.min())) / scope + s.min()


def OLS(X, y):
    """
    做线性回归，返回 β0（截距）、β1（系数）和残差
    y = β0 + x1*β1 + epsilon
    参考：https://blog.csdn.net/chongminglun/article/details/104242342
    :param X: shape(N,M)，M位X的维度，一般M=1
    :param y: shape(N)
    :return:参数[β0、β1]，R2
    """
    assert not np.isnan(X).any(), f'X序列包含nan:{X}'
    assert not np.isnan(y).any(), f'y序列包含nan:{y}'

    # 增加一个截距项
    X = sm.add_constant(X)
    # 定义模型
    model = sm.OLS(y, X)  # 定义x，y
    results = model.fit()
    # 参数[β0、β1]，R2
    return results.params, results.rsquared

def create_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def load_params(name='params.yml'):
    if not os.path.exists(name):
        raise ValueError(f"参数文件[{name}]不存在，请检查路径")
    params = yaml.load(open(name, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    params = AttributeDict(params)
    return params


def date2datetime(date):
    return datetime.datetime.combine(date, datetime.datetime.min.time())


def get_trade_periods(start='20120101', end=None, period='daily'):
    """
    返回从since开始的交易市场的周期，用于循环用,
    使用的是第三方的pip包，来返回市场交易日期和时间
    注意，时间都是收盘时间，但是日是开始时间，
        日： 2013-01-04 00:00
        15min: 09:45~11:30 13:15~15:00
        5min:  09:35~11:30 13:05~15:00
    注意：
        xcals是从2003.11.17号开始的，而chinese_calendar是从2004.1.4号开始的
    20231120，xcals+chinese还是有bug，转向了tushare的[trade_cal](https://tushare.pro/document/2?doc_id=26)
    :param period: daily|5min|15min|30min|60m
    :return:
    """

    # 唉，最终放弃了xcal+chinese_calendar组合，他们获取交易日历还是有问题，比如20230502，明显5.1长假，都被识别成了交易日
    # 最终转向了tushare的[trade_cal接口](https://tushare.pro/document/2?doc_id=26)
    # if end is None: end = datetime.datetime.now()
    # if type(start) == str: start = str2date(start)
    # if type(end) == str: end = str2date(end)
    # xshg = xcals.get_calendar("XSHG")
    # dates = xshg.schedule.loc[start:end].index
    # xcals出现了端午节还是交易日的bug，所以有用chinese_calender过滤一下比如是工作日和周一~周五，另外chinese_calender只查2004后的
    # dates = [d for d in dates if d.year>2004 and chinese_calendar.is_workday(d) and datetime.datetime.isoweekday(d) < 6]

    from utils import data_loader
    if end is None: end = date2str(now())
    df_cal = data_loader.load_trade_calendar(start, end)
    dates = df_cal.date

    if period == 'daily':
        logger.debug("返回日历[%s]~[%s] %d天", start, end, len(dates))
        # pandas._libs.tslibs.timestamps.Timestamp 转成 datetime.date
        return [d.to_pydatetime() for d in dates]

    def generate_span(minute):
        datetime_array = []
        time_delta = datetime.timedelta(minutes=minute)  # 设置时间间隔

        for date in dates:
            # 把date转成datetime, 2019-1-1=>2019-1-1 00:00
            start_time = date2datetime(date)
            # end_time为明天日期
            end_time = start_time + datetime.timedelta(hours=24)
            current_time = start_time
            # 遍历所有的15分钟，只保留 9:00~11:30 & 13:00~15:00，剔除9:00和13:00两个开头
            while current_time <= end_time:
                # 如果
                if current_time.time() < datetime.time(hour=9, minute=31):
                    current_time += time_delta
                    continue
                if current_time.time() > datetime.time(hour=15, minute=1):
                    current_time += time_delta
                    continue
                if datetime.time(hour=13, minute=1) > current_time.time() > datetime.time(hour=11, minute=30):
                    current_time += time_delta
                    continue
                datetime_array.append(current_time)
                current_time += time_delta
        logger.debug("返回日历[%s]~[%s] %d分钟级别，共%d条", start, end, minute, len(datetime_array))
        return datetime_array

    if period == '15min':
        return generate_span(15)

    if period == '5min':
        return generate_span(5)

    raise ValueError("period参数应该：daily,5min,15min，您给了：", period)


def get_monthly_duration(start_date, end_date):
    """
    把开始日期到结束日期，分割成每月的信息
    比如20210301~20220515 =>
    [   [20210301,20210331],
        [20210401,20210430],
        ...,
        [20220401,20220430],
        [20220501,20220515]
    ]
    """

    start_date = str2date(start_date)
    end_date = str2date(end_date)
    years = list(range(start_date.year, end_date.year + 1))
    scopes = []
    for year in years:
        if start_date.year == year:
            start_month = start_date.month
        else:
            start_month = 1

        if end_date.year == year:
            end_month = end_date.month + 1
        else:
            end_month = 12 + 1

        for month in range(start_month, end_month):

            if start_date.year == year and start_date.month == month:
                s_start_date = date2str(datetime.date(year=year, month=month, day=start_date.day))
            else:
                s_start_date = date2str(datetime.date(year=year, month=month, day=1))

            if end_date.year == year and end_date.month == month:
                s_end_date = date2str(datetime.date(year=year, month=month, day=end_date.day))
            else:
                _, last_day = calendar.monthrange(year, month)
                s_end_date = date2str(datetime.date(year=year, month=month, day=last_day))

            scopes.append([s_start_date, s_end_date])

    return scopes


def get_series(df, index_key, num):
    """
    # 先前key之前或者之后的series
    """
    try:
        loc = df.index.get_loc(index_key)
        s = df.iloc[loc + num]
        return s
    except KeyError:
        return None


def generate_svg_html(target_svg_path, template_path='utils/svg_template.html', patten_str='$SVG_PATH$'):
    """生成嵌入了svg路径的html"""

    # 1.修改svg内容
    html_path = target_svg_path.replace('svg', 'html')
    svg_filename = target_svg_path.split('/')[-1]  # debug/002438.svg => 002438.svg
    # 打开文件并读取内容
    with open(template_path, 'r') as file:
        content = file.read()
    # 使用replace()函数替换匹配字符串的内容
    new_content = content.replace(patten_str, svg_filename)

    # 2.修改统计的内容
    stat_path = target_svg_path.replace('svg', 'txt')
    if os.path.exists(stat_path):
        with open(stat_path, 'r') as file:
            stat_content = file.read()
        new_content = new_content.replace('$STAT$', stat_content)

    # 写入替换后的内容到文件
    with open(html_path, 'w') as file:
        file.write(new_content)

    return html_path


def get_previous_df(df, index_key, num, offset=0):
    """
    按照index-key，取前N天的子dataframe（包含索引index-key日）
    offset是可以再往前几天，为正数：例如，如果是2，就是往前2天
    """
    try:
        num = abs(num)  # 防止传入的是个负数
        offset = abs(offset)  # 防止传入的是个负数
        loc = df.index.get_loc(index_key)
        df1 = df.iloc[loc - num - offset + 1:loc + 1 - offset]
        return df1
    except KeyError:
        return None


# def calc_k(p1, p2):
#     """计算2个点的斜率"""
#     return (p1[1] - p2[1]) / (p1[0] - p2[0])


def calc_y(p1, p2, x):
    """根据两点计算直线上其他的x对应的y值"""
    x1, y1 = p1
    x2, y2 = p2
    # 计算斜率m
    m = (y2 - y1) / (x2 - x1)
    # 计算截距b
    b = y1 - m * x1
    # 计算y值
    y = m * x + b
    return y


def concat(column_name, df, new_series):
    """
    赋值给一个新列的时候，报性能错误：
     PerformanceWarning: DataFrame is highly fragmented.
     This is usually the result of calling `frame.insert` many times,
     which has poor performance.
     Consider joining all columns at once using pd.concat(axis=1) instead.
     To get a de-fragmented frame, use `newframe = frame.copy()`
    所以，把df['new_col'] = ...，改成了 concat('new_col', old_df, new_series)
    """
    df1 = pd.DataFrame(new_series)
    df1 = df1.rename(columns={df1.columns[0]: f'{column_name}'})
    return pd.concat([df, df1], axis=1)


import math


def tan2angle(tan_value):
    # 计算反正切值
    atan_value = math.atan(tan_value)

    # 将反正切值转换为角度
    angle = atan_value * 180 / math.pi
    return angle


def is_in_same_week(current_datetime, friday_datetime):
    """是否是同一周，gpt生成的"""
    year1, weeknum1, _ = current_datetime.isocalendar()
    year2, weeknum2, _ = friday_datetime.isocalendar()
    return year1 == year2 and weeknum1 == weeknum2


def is_in_same_month(current_datetime, month_end_datetime):
    # 比较年份和月份是否相同
    if current_datetime.year == month_end_datetime.year and current_datetime.month == month_end_datetime.month:
        return True
    else:
        return False


def compare_date(d1, d2):
    """
    d1、d2可能是date，也可能是datetime，
    比较时候都转成datetime比较，date会被转成00:00来对待
    :return: 1: d1>d2, -1: d1<d2, 0: d1=d2
    """
    if type(d1) == datetime.date:
        d1 = datetime.datetime.combine(d1, datetime.datetime.min.time())
    if type(d2) == datetime.date:
        d2 = datetime.datetime.combine(d1, datetime.datetime.min.time())
    if d1 == d2: return 0
    return 1 if d1 > d2 else -1


def __date_span(date_type, unit, direction, s_date):
    """
    last('year',1,'2020.1.3')=> '2019.1.3'
    :param unit:
    :param date_type: year|month|day
    :return:
    """
    the_date = str2date(s_date)
    if date_type == 'year':
        return date2str(the_date + relativedelta(years=unit) * direction)
    elif date_type == 'month':
        return date2str(the_date + relativedelta(months=unit) * direction)
    elif date_type == 'week':
        return date2str(the_date + relativedelta(weeks=unit) * direction)
    elif date_type == 'day':
        return date2str(the_date + relativedelta(days=unit) * direction)
    else:
        raise ValueError(f"无法识别的date_type:{date_type}")


def last(date_type, unit, s_date):
    return __date_span(date_type, unit, -1, s_date)


def last_year(s_date, num=1):
    return last('year', num, s_date)


def last_month(s_date, num=1):
    return last('month', num, s_date)


def last_week(s_date, num=1):
    return last('week', num, s_date)


def last_day(s_date, num=1):
    return last('day', num, s_date)


def next(date_type, unit, s_date):
    return __date_span(date_type, unit, 1, s_date)


def next_year(s_date, num=1):
    return next('year', num, s_date)


def next_month(s_date, num=1):
    return next('month', num, s_date)


def next_week(s_date, num=1):
    return next('week', num, s_date)


def next_day(s_date, num=1):
    return next('day', num, s_date)


def next_trade_date(today):
    """入参和返回都是str类型"""
    _next_week = next_week(today)
    dates = get_trade_periods(start=today, end=_next_week, period='daily')
    # 日期=>str
    dates = [date2str(d) for d in dates]
    if not today in dates:
        logger.warning("今日%s不是交易日", today)
        return None
    return dates[dates.index(today) + 1]


def tomorrow(s_date=None):
    if s_date is None: s_date = today()
    return future('day', 1, s_date)


def yesterday(s_date=None):
    if s_date is None: s_date = today()
    return last_day(s_date, 1)


def last(date_type, unit, s_date):
    return __date_span(date_type, unit, -1, s_date)


def last_year(s_date, num=1):
    return last('year', num, s_date)


def last_month(s_date, num=1):
    return last('month', num, s_date)


def last_week(s_date, num=1):
    return last('week', num, s_date)


def last_day(s_date, num=1):
    return last('day', num, s_date)


def today():
    now = datetime.datetime.now()
    return datetime.datetime.strftime(now, "%Y%m%d")


def future(date_type, unit, s_date):
    return __date_span(date_type, unit, 1, s_date)


def __date_span(date_type, unit, direction, s_date):
    """
    last('year',1,'2020.1.3')=> '2019.1.3'
    :param unit:
    :param date_type: year|month|day
    :return:
    """
    the_date = str2date(s_date)
    if date_type == 'year':
        return date2str(the_date + relativedelta(years=unit) * direction)
    elif date_type == 'month':
        return date2str(the_date + relativedelta(months=unit) * direction)
    elif date_type == 'week':
        return date2str(the_date + relativedelta(weeks=unit) * direction)
    elif date_type == 'day':
        return date2str(the_date + relativedelta(days=unit) * direction)
    else:
        raise ValueError(f"无法识别的date_type:{date_type}")


# python -m utils.utils
if __name__ == '__main__':
    init_logger()
    # p = load_params('triples/params.yml')
    # print(p)
    # print(p.start_date)
    #
    # # p = get_monthly_duration('20140101', '20230201')
    # # print(p)
    #
    # print(nomalize(np.array([1, 2, 3, 4, 5]), 10))
    # print(nomalize(np.array([5, 4, 3, 2, 1]), 10))
    #
    # # 测试OLS
    # y = np.array([62.35, 66.80, 67.50, 66.85, 66.50, 67.65, 73.85, 70.45, 71.00, 69.45, 69.05])
    # x = list(range(len(y)))
    # beta, _ = utils.utils.OLS(x, y)
    # y1 = [beta[0] + beta[1] * i for i in x]
    # print("回归测试：")
    # print("beta:", beta)
    # print("原x：", x)
    # print("原y：", y)
    # print("拟y：", y1)
    # import matplotlib.pyplot as plt
    #
    # plt.plot(x, y)
    # plt.plot(x, y1)
    # plt.show()

    # print(get_trade_periods(start='20120104', end='20120201', period='daily'))
    # print(get_trade_periods(start='20120104', end='20120104', period='15min'))
    # print(get_trade_periods(start='20120104', end='20120104', period='5min'))
    # ds = get_trade_periods(start='19910101', end='20231101', period='daily')
    # print("19910101~20231101交易天数:", len(ds))
    # print(get_trade_periods(start='19930104', end='19930201', period='daily'))
    # # 靠！，验证了，日期是从2003.11.17号开始的
    # print(get_trade_periods(start='20030104', end='20040101', period='daily'))
    # # 20230502

    # d = {
    #     'a':1,
    #     'b':{
    #         'b1':'111111',
    #         'b2':222222,
    #         'b3':{
    #             'b31':31,
    #             'b32':32
    #         }
    #     }
    # }
    # new_d = AttributeDict(d)
    # print(new_d.b)
    # print(new_d.b.b3.b31)

    print(next_trade_date('20231215'))
