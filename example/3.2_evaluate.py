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
    factor_information_coefficient, get_factor_ic_summary_info, plot_quantile_info, plot_quantile_netvalue
# igmore the warning of ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)
# obtain the ddb session
s = du.DDBSessionSingleton().session  


def load_factor_from_ddb(stat_date, factor_name, config):
    fac_tb = SecLevelFacTable(config['factor_dbPath'], config['factor_tbName'])
    fac = fac_tb.load_factor(factor_name, stat_date, config['start_time'], config['end_time'], sec_list=None)
    factor_df = s.loadTable(tableName=fac).toDF()
    if (factor_df.empty):
        return None
    # filter the dataframe
    start_time = pd.to_datetime('9:30').time()
    end_time = pd.to_datetime('14:57').time()
    factor_df = factor_df.set_index('tradetime').between_time(start_time, end_time).reset_index()
    factor_df.set_index(['tradetime', 'securityid'], inplace=True)
    
    factor_df.rename(columns={'value': 'factor'}, inplace=True)
    factor_df.replace(np.inf, np.nan, inplace=True)
    factor_df = factor_df.sort_index(level=0)
    
    factor_df = get_clean_data(factor_df, config['evaluation'])
    return factor_df
    
    
def load_factor(stat_date, factor_name, config):
    base_dirpath = '/home/wangzirui/workspace/data'
    factors_filepath = f'{base_dirpath}/fac_ret_{stat_date}.pkl'
    if not os.path.exists(factors_filepath):
        print(f'[warning]: There is no file:{factors_filepath}')
        return None
    print(f"\t Processing {stat_date} ...", flush=True)
    factors_df = pd.read_pickle(factors_filepath)
    factors_df.set_index(['tradetime', 'securityid'], inplace=True)
    # factors_df = factors_df.sort_index(level=0)
    
    # get the dataframe with factor and return information
    sel_cols = factors_df.columns[:3]
    sel_cols = np.append(sel_cols, [factor_name])
    factor_and_ret = factors_df[sel_cols].copy()
    factor_and_ret.rename(columns={factor_name: "factor"}, inplace=True)
    factor_and_ret.replace(np.inf, np.nan, inplace=True)
    
    # tidy the dataframe and quantile it
    factor_and_ret = get_clean_data(factor_and_ret, config['evaluation'])
    factor = factor_and_ret[['factor', 'factor_quantile']].copy()
    return factor

def do_pnl_curve(data, td_ret, factor_name):
    base_dirpath = '/home/wangzirui/workspace/factor_eval_summary/top_n'
    # calculate pnl curve
    factor = data[['factor', 'factor_quantile']].copy()
    port_metric_summary, port_netvalue_summary = factor_portfolio_return(factor, td_ret, holding_time=20, long_short=True)
    port_metric_summary['factor_name'] = factor_name
    table_dirpath = f'{base_dirpath}/BTtable'
    os.makedirs(table_dirpath, exist_ok=True)
    port_metric_summary.to_csv(f'{table_dirpath}/{factor_name}_metric.csv')
    port_netvalue_summary.to_csv(f'{table_dirpath}/{factor_name}_netvalue.csv')
    
    plot_dirpath = f'{base_dirpath}/BTplot'
    os.makedirs(plot_dirpath, exist_ok=True)
    nv_group_filepath = f"{plot_dirpath}/{factor_name}.png"
    plot_quantile_netvalue(port_netvalue_summary, nv_group_filepath)
    
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

# def process_perFactor(factor_name, start_date, end_date, config):
#     datas = []
#     for date in pd.date_range(start_date, end_date):
#         date = date.strftime('%Y.%m.%d')
#         data = load_factor_and_return(date, factor_name, config)
#         if data is not None:
#             datas.append(data)
#     if len(datas) > 0:
#         data = pd.concat(datas, axis=0)

#         do_ic_evaluate(data, factor_name)

if __name__ == "__main__":
    # read config file
    config = cfg.BasicConfig('config/config.yml')

    start_date = '2023.09.22'
    end_date = '2023.09.22'
    
    for facType in config['factors']:
        for i, factor_name in enumerate(config['factors'][facType]):
            if (i >= 1):
                break
            print("Evaluating factor: ", factor_name, " ...", flush=True)
            for date in pd.date_range(start_date, end_date):
                date = date.strftime('%Y.%m.%d')
                # Source 1: from a prepared pickle file
                # data = load_factor(date, factor_name, config)
                # Source 2: from dolphindb 
                data = load_factor_from_ddb(date, factor_name, config)
                
                close = get_trade_close_bydate(date, config)
                td_ret = get_trade_return(close)
                if data is not None:
                    do_pnl_curve(data, td_ret, factor_name)
    
    # import yaml
    # pred_type='1m'
    # base_dir = r'/home/wangzirui/workspace/factor_ic_summary/factor_comb_top_n/bid_ask_price'
    # factor_filepath = os.path.join(base_dir, f'satisfied_factors_{pred_type}_without_hcorr.yml')
    # with open(factor_filepath, 'r') as f:
    #     factor_names = yaml.load(f, Loader=yaml.FullLoader)
    # for factor_name in factor_names:
    #     print("Evaluating factor: ", factor_name, " ...", flush=True)
    #     process_perFactor(factor_name, start_date, end_date, config)