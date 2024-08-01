import warnings
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ConstantInputWarning
import os, sys
sys.path.insert(0, "../")
from alphalens.utils import get_forward_returns_columns

from factor_cal.config_loader import basic_config as cfg
from factor_cal.table.ddb_table import PriceTable, SecLevelFacTable
from factor_cal.utils import ddb_utils as du
from factor_cal.factor_eval.basic_evaluate import get_clean_data, factor_portfolio_return,\
    factor_information_coefficient, get_factor_ic_summary_info, plot_quantile_ic
# igmore the warning of ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)
# obtain the ddb session
s = du.DDBSessionSingleton().session   

def get_trade_close_bydate(stat_date, config):
    price_info = config['price_info']
    pc_tb = PriceTable(price_info['price_dbPath'], price_info['price_tbName'], price_info['time_col'], price_info['sec_col'], price_info['price_cols'])
    price = pc_tb.load_price(stat_date, config['start_time'], config['end_time'], sec_list=None)
    return s.loadTable(tableName=price).toDF()

def get_trade_return(close):
    close.set_index(['tradetime', 'securityid'], inplace=True)
    close.sort_index(inplace=True)
    res = close.groupby(close.index.get_level_values('securityid')).apply(lambda x: x.pct_change(1))
    res.rename(columns={'close': 'ret'}, inplace=True)
    return res

def load_factor_and_return(stat_date, factor_name, config):
    base_dirpath = '/home/wangzirui/workspace/data'
    factors_filepath = f'{base_dirpath}/fac_ret_{stat_date}.pkl'
    if not os.path.exists(factors_filepath):
        print(f'[warning]: There is no file:{factors_filepath}')
    factors_df = pd.read_pickle(factors_filepath)
    factors_df.set_index(['tradetime', 'securityid'], inplace=True)
    # factors_df = factors_df.sort_index(level=0)
    
    # get the dataframe with factor and return information
    sel_cols = factors_df.columns[:3]
    sel_cols = np.append(sel_cols, [factor_name])
    factor_and_ret = factors_df[sel_cols].copy()
    factor_and_ret.rename(columns={factor_name: "factor"}, inplace=True)
    factor_and_ret.replace(np.inf, np.nan, inplace=True)
    
    # get trade close and return for each tradetime and each symbol
    # close = get_trade_close_bydate(stat_date, config)
    # td_ret = get_trade_return(close)
    
    # tidy the dataframe and quantile it
    factor_and_ret = get_clean_data(factor_and_ret, config['evaluation'])
    
    # calculate IC infomation in total or by group
    ic_data = factor_information_coefficient(factor_and_ret)
    ic_summary = get_factor_ic_summary_info(ic_data)
    base_dirpath = '/home/wangzirui/workspace/factor_eval_summary/top_n'
    table_dirpath = f'{base_dirpath}/ICtable'
    os.makedirs(table_dirpath, exist_ok=True)
    ic_summary.to_csv(f'{table_dirpath}/{factor_name}.csv')
    
    
    ic_data_bygroup = factor_information_coefficient(factor_and_ret, by_group=True)
    ic_summary_bygroup = get_factor_ic_summary_info(ic_data_bygroup, by_group=True)
    ic_summary_bygroup.to_csv(f'{table_dirpath}/{factor_name}_group.csv')
    
    plot_dirpath = f'{base_dirpath}/ICplot'
    os.makedirs(plot_dirpath, exist_ok=True)
    ic_group_filepath = f"{plot_dirpath}/{factor_name}.png"
    plot_quantile_ic(ic_summary_bygroup.loc['IC Mean'], ic_group_filepath)
    
    # calculate pnl curve
    port_metric_summary, port_netvalue_summary = factor_portfolio_return(factor_and_ret, td_ret, holding_time=20, long_short=True)
    port_metric_summary['factor_name'] = factor_name
    table_dirpath = f'{base_dirpath}/BTtable'
    os.makedirs(table_dirpath, exist_ok=True)
    port_metric_summary.to_csv(f'{table_dirpath}/{factor_name}_metric.csv')
    port_netvalue_summary.to_csv(f'{table_dirpath}/{factor_name}_netvalue.csv')
    
    plot_dirpath = f'{base_dirpath}/BTplot'
    os.makedirs(plot_dirpath, exist_ok=True)
    nv_group_filepath = f"{plot_dirpath}/{factor_name}.png"
    plot_quantile_netvalue(port_netvalue_summary, nv_group_filepath)
    
    
    



if __name__ == "__main__":
    # read config file
    config = cfg.BasicConfig('config/config.yml')

    cur_date = '2023.09.25'
    factor_name = 'ret_v_prod_5min'
    load_factor_and_return(cur_date, factor_name, config)