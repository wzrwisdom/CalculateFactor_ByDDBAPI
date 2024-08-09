import warnings
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, "../")
from scipy.stats import mode
from alphalens.utils import timedelta_to_string
from datetime import datetime

from factor_cal.config_loader import basic_config as cfg
from factor_cal.table.ddb_table import PriceTable, SecLevelFacTable
from factor_cal.utils import ddb_utils as du


# igmore the warning of ConstantInputWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
# obtain the ddb session
s = du.DDBSessionSingleton().session

def compute_forward_return_v1(prices, periods=(1, 5, 10), filter_zscore=None, cumulative_returns=True):
    dateindex = prices.index

    raw_values_dict = {}
    column_list = []
    for period in sorted(periods):
        if cumulative_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()
    
        forward_returns = \
            returns.shift(-period).reindex(dateindex)
        
        if filter_zscore is not None:
            mask = abs(
                forward_returns - forward_returns.mean()
            ) > (filter_zscore * forward_returns.std())
            forward_returns[mask] = forward_returns.mean() + np.sign(forward_returns[mask] - forward_returns.mean()) * (filter_zscore * forward_returns.std())
        #
        # Find the period length, which will be the column name. We'll test
        # several entries in order to find out the correct period length as
        # there could be non-trading days which would make the computation
        # wrong if made only one test
        #
        entries_to_test = min(
            30,
            len(forward_returns.index),
            len(prices.index) - period
        )
        times_diffs = []
        for i in range(entries_to_test):
            p_idx = prices.index.get_loc(forward_returns.index[i])
            start = prices.index[p_idx]
            end = prices.index[p_idx + period]
            period_len = end-start
            times_diffs.append(period_len)
        period_len = mode(times_diffs, keepdims=True).mode[0]
        label = timedelta_to_string(period_len)
        
        column_list.append(label)
        
        raw_values_dict[label] = np.concatenate(forward_returns.values)
    
    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        pd.MultiIndex.from_product(
            [dateindex, prices.columns],
            names=['tradetime', 'securityid']
        ),
        inplace=True
    )

    df = df[column_list]
    df.index.set_names(['tradetime', 'securityid'], inplace=True)
    
    return df


def compute_forward_return_v2(bid1, ask1, periods=(1, 5, 10), filter_zscore=None, cumulative_returns=True):
    dateindex = bid1.index
    dateindex = dateindex.intersection(ask1.index)

    raw_values_dict = {}
    column_list = []
    for period in sorted(periods):
        if cumulative_returns:
            returns = bid1 / ask1.shift(period) - 1
        else:
            returns = bid1 / ask1.shift() - 1
    
        forward_returns = \
            returns.shift(-period).reindex(dateindex)
        
        if filter_zscore is not None:
            mask = abs(
                forward_returns - forward_returns.mean()
            ) > (filter_zscore * forward_returns.std())
            forward_returns[mask] = forward_returns.mean() + np.sign(forward_returns[mask] - forward_returns.mean()) * (filter_zscore * forward_returns.std())
        #
        # Find the period length, which will be the column name. We'll test
        # several entries in order to find out the correct period length as
        # there could be non-trading days which would make the computation
        # wrong if made only one test
        #
        entries_to_test = min(
            30,
            len(forward_returns.index),
            len(bid1.index) - period
        )
        times_diffs = []
        for i in range(entries_to_test):
            p_idx = bid1.index.get_loc(forward_returns.index[i])
            start = bid1.index[p_idx]
            end = bid1.index[p_idx + period]
            period_len = end-start
            times_diffs.append(period_len)
        period_len = mode(times_diffs, keepdims=True).mode[0]
        label = timedelta_to_string(period_len)
        
        column_list.append(label)
        
        raw_values_dict[label] = np.concatenate(forward_returns.values)
    
    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        pd.MultiIndex.from_product(
            [dateindex, bid1.columns],
            names=['tradetime', 'securityid']
        ),
        inplace=True
    )

    df = df[column_list]
    df.index.set_names(['tradetime', 'securityid'], inplace=True)
    
    return df
    
    
def prepare_bid_ask_return(date, config):
    price_info = config['snap_price_info']
    pc_tb = PriceTable(price_info['price_dbPath'], price_info['price_tbName'], price_info['time_col'], price_info['sec_col'], price_info['price_cols'])
    price = pc_tb.load_price(date, config['start_time'], config['end_time'], sec_list=None)
    price_df = s.loadTable(tableName=price).toDF()
    if price_df.empty:
        print('\tNo data for date: ', date)
        return None
    
    price_df = price_df.set_index(['tradetime', 'securityid'])
    bid1 = price_df['b1'].unstack()
    ask1 = price_df['s1'].unstack()
    
    ret_df = compute_forward_return_v2(bid1, ask1, periods=(20, 60, 100), filter_zscore=None, cumulative_returns=True)
    
    date_tmp = datetime.strptime(date, '%Y.%m.%d')
    date = date_tmp.strftime('%Y-%m-%d')
    start_time = f'{date} 09:30:00'
    end_time = f'{date} 14:57:00'
    mask = (ret_df.index.get_level_values('tradetime') >= start_time) & (ret_df.index.get_level_values('tradetime') <= end_time)
    return ret_df[mask]

def prepare_simple_ret(date, config, price_type='price_info', price_col='close'):
    price_info = config[price_type]
    pc_tb = PriceTable(price_info['price_dbPath'], price_info['price_tbName'], price_info['time_col'], price_info['sec_col'], price_info['price_cols'])
    price = pc_tb.load_price(date, config['start_time'], config['end_time'], sec_list=None)
    price_df = s.loadTable(tableName=price).toDF()
    if price_df.empty:
        print('\tNo data for date: ', date)
        return None
    price_df = price_df.set_index(['tradetime', 'securityid'])
    prices = price_df[price_col].unstack()
    
    ret_df = compute_forward_return_v1(prices, periods=(20, 60, 100), filter_zscore=None, cumulative_returns=True)
    
    date_tmp = datetime.strptime(date, '%Y.%m.%d')
    date = date_tmp.strftime('%Y-%m-%d')
    start_time = f'{date} 09:30:00'
    end_time = f'{date} 14:57:00'
    mask = (ret_df.index.get_level_values('tradetime') >= start_time) & (ret_df.index.get_level_values('tradetime') <= end_time)
    return ret_df[mask]

if __name__ == "__main__":
    # read config file
    config = cfg.BasicConfig('config/config.yml')
    
    base_dir = "/home/wangzirui/workspace/data"
    dates = pd.date_range(config['start_date'], config['end_date'])
    
    for date in dates:
        date = date.strftime('%Y.%m.%d')
        
        print("Calculating return by bid1 and ask1 ...")
        ret_df = prepare_bid_ask_return(date, config)
        if ret_df is not None:
            save_dir = f'{base_dir}/bid_ask_return'
            os.makedirs(save_dir, exist_ok=True)
            save_filepath = f'{save_dir}/{date}.pkl'
            ret_df.to_pickle(save_filepath)
    
        print("Calculating return by close...")
        ret_df = prepare_simple_ret(date, config, 'price_info', 'close')
        if ret_df is not None:
            save_dir = f'{base_dir}/close_return'
            os.makedirs(save_dir, exist_ok=True)
            save_filepath = f'{save_dir}/{date}.pkl'
            ret_df.to_pickle(save_filepath)
        
        print("Calculating return by bs_avg_price...")
        ret_df = prepare_simple_ret(date, config, 'snap_price_info', 'bs_avg_price')
        if ret_df is not None:
            save_dir = f'{base_dir}/bs_avg_price_return'
            os.makedirs(save_dir, exist_ok=True)
            save_filepath = f'{save_dir}/{date}.pkl'
            ret_df.to_pickle(save_filepath)
