from server import const
from utils import data_loader, stat
from utils.utils import load_params

conf  = load_params(const.CONFIG)
df = data_loader.load_trades(conf)
s = stat.stat_trade(df)
print(s)

# python -m test.test_trade_stat