import logging
import os

import matplotlib
import matplotlib.pyplot as plt

from server import const
from utils import data_loader
from utils.metrics import drawback_duration
from utils.utils import create_dir, init_logger, load_params


logger = logging.getLogger(__name__)

FIGSIZE = (20, 25)


def plot(df, df_baselines, params):
    """
    报错，所以加上matplotlib.use("Agg")
    UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
    fig = plt.figure(figsize=(15, 5))
    *** Terminating app due to uncaught exception 'NSInternalInconsistencyException',
    reason: 'NSWindow drag regions should only be invalidated on the Main Thread!'
    """
    matplotlib.use('Agg')

    start_date = df.iloc[0].date
    end_date = df.iloc[-1].date

    df.set_index('date', inplace=True)
    df_drawback = drawback_duration(df.total_value, 'daily')

    fig = plt.figure(figsize=(15, 5))
    size = (1, 1)
    ax = plt.subplot2grid(size, (0, 0))
    ax.set_title(f'组合收益')
    ax.set_xlabel('日期')
    ax.set_ylabel('市值')
    # 画组合的总市值（仓位+现金）变化
    ax.plot(df.index, df.total_value, 'c', label='总市值')
    ax.fill_between(df.index, df.total_value, df.total_value.min(), alpha=0.1, label='组合市值')
    # 画所有回撤
    for _, drawback in df_drawback.iterrows():  # columns=['start','end','days','drawback']
        start = drawback.start
        end = drawback.end
        df_draw = df.loc[start:end]
        ax.fill_between(df_draw.index, df_draw.total_value, df_draw.total_value.min(), alpha=0.1, color='green')
    # 画最大回撤的期间
    s = df_drawback.loc[df_drawback.drawback.idxmax()]
    df_draw = df.loc[s.start:s.end]
    ax.fill_between(df_draw.index, df_draw.total_value, df_draw.total_value.min(), alpha=0.3, color='red',
                    label='最大回撤')
    # 画最长回撤的期间
    s = df_drawback.loc[df_drawback.days.idxmax()]
    df_draw = df.loc[s.start:s.end]
    ax.fill_between(df_draw.index, df_draw.total_value, df_draw.total_value.min(), alpha=0.3, color='brown',
                    label='最长回撤')
    ax.legend(loc='upper right')

    # 画基准
    handles = []
    for df_baseline in df_baselines:
        ax2 = ax.twinx()
        ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler  # 让颜色可以自动循环起来
        df_baseline = df_baseline[(df_baseline.date > start_date) & (df_baseline.date < end_date)]
        h, = ax2.plot(df_baseline.date, df_baseline.close, linewidth=1, label=f'基准{df_baseline.iloc[0].code}')
        handles.append(h)
    ax2.legend(handles=handles, loc='upper left')

    # 保存图形
    fig.tight_layout()
    create_dir('data')
    svg_path = os.path.join(params.asset_dir, 'summary.svg')
    plt.savefig(svg_path)
    logger.debug("保存图表为svg文件：%s", svg_path)
    plt.close()


# python -m utils.plot
if __name__ == '__main__':
    init_logger()
    conf = load_params(const.CONFIG)
    df = data_loader.load_accounts(conf)
    plot(df, conf)
    print("data/summary.svg")
