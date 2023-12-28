"""造一些市值的假数据"""
# account_id,cash,total_value,total_position_value,date
# 55003329,19963138.44,19978195.44,15066.0,20231227
import datetime
from random import randint

import pandas as pd

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
df = pd.DataFrame(data,columns=["account_id", "cash", "total_value", "total_position_value", "date"])
df.sort_values(by='date',ascending=True,inplace=True)
df['total_value'] = df.total_value.cumsum()
print(df)
df.to_csv('data/market_value.csv',index=False)
# python -m test.test_generate_market_value
