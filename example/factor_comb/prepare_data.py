import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "../")
import alphalens


from factor_cal.config_loader import basic_config as cfg
from factor_cal.table.ddb_table import PriceTable, SecLevelFacTable
from factor_cal.utils import ddb_utils as du

# read config file
config = cfg.BasicConfig('config/config.yml')
# obtain the ddb session
s = du.DDBSessionSingleton().session

factor_names = []
for facType in config['factors']:
    factor_names += list(config['factors'][facType].keys())
    

excluded_factors = ['close_ret']
factor_names = [x for x in factor_names if x not in excluded_factors]

## For testing purpose, only use the first two factors
# factor_names = factor_names[:2]

start_date = '2023.09.21'
end_date = '2024.02.20'
start_time = '09:45:00'
end_time = '14:45:00'
dates = pd.date_range(start_date, end_date)




for date in dates:
    date = date.strftime('%Y.%m.%d')
    print("processing date: ", date, "...")
    
    price_info = config['price_info']
    pc_tb = PriceTable(price_info['price_dbPath'], price_info['price_tbName'], price_info['time_col'], price_info['sec_col'], price_info['price_cols'])
    price = pc_tb.load_price(date, config['start_time'], config['end_time'], sec_list=None)
    price_df = s.loadTable(tableName=price).toDF()
    if price_df.empty:
        print('\tNo data for date: ', date)
        continue
    price_df = price_df.set_index(['tradetime', 'securityid'])
    prices = price_df['close'].unstack()
    
    ret_df = None
    facs_df = None
    for fac_name in factor_names:
        fac_tb = SecLevelFacTable(config['factor_dbPath'], config['factor_tbName'])
        fac = fac_tb.load_factor(fac_name, date, start_time, end_time, sec_list=None)
        fac_df = s.loadTable(tableName=fac).toDF()
        fac_df.rename(columns={'value': fac_name}, inplace=True)
        fac_df.drop('factorname', axis=1, inplace=True)
        if facs_df is None:
            facs_df = fac_df
        else:
            facs_df = facs_df.merge(fac_df, on=['tradetime', 'securityid'], how='outer')
        
        if ret_df is None:
            fac_df.set_index(['tradetime', 'securityid'], inplace=True)
            fac_df = fac_df.sort_index(level=0)
            ret_df = alphalens.utils.compute_forward_returns(fac_df, prices, periods=[20, 60, 100], filter_zscore=20, cumulative_returns=True)
            ret_df.index.set_names(['tradetime', 'securityid'], inplace=True)
            ret_df.reset_index(inplace=True)
            ret_df = ret_df[~(ret_df.isna().sum(axis=1) == 3)]
        
    df = ret_df.merge(facs_df, on=['tradetime', 'securityid'], how='left')
    df.to_pickle(f'/home/wangzirui/workspace/data/fac_ret_{date}.pkl')