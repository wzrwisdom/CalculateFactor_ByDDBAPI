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
    factor_information_coefficient, get_factor_ic_summary_info, plot_quantile_info, plot_quantile_netvalue,\
    factor_timeSeries_information_coefficient, factor_group_rough_return
# igmore the warning of ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)
# obtain the ddb session
s = du.DDBSessionSingleton().session  

def load_factor_and_return_from_ddb(stat_date, factor_name, config, ret_type='close_return'):
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
    
   
    base_dir = f'/home/wangzirui/workspace/data/{ret_type}'
    ret_filepath = f'{base_dir}/{stat_date}.pkl'
    ret_df = pd.read_pickle(ret_filepath)
    
    factor_and_ret = factor_df.merge(ret_df, how='left', on=['tradetime', 'securityid'], sort=True)
    return factor_and_ret 


def load_factor_and_return(stat_date, factor_name, config):
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
    return factor_and_ret

def do_ic_evaluate(data, factor_name, base_dirpath):
    # calculate IC infomation in total or by group
    ic_data = factor_information_coefficient(data)
    ic_summary = get_factor_ic_summary_info(ic_data)

    table_dirpath = f'{base_dirpath}/ICtable'
    os.makedirs(table_dirpath, exist_ok=True)
    ic_summary.to_csv(f'{table_dirpath}/{factor_name}.csv')
    
    
    ic_data_bygroup = factor_information_coefficient(data, by_group=True)
    ic_summary_bygroup = get_factor_ic_summary_info(ic_data_bygroup, by_group=True)
    ic_summary_bygroup.to_csv(f'{table_dirpath}/{factor_name}_group.csv')
    
    plot_dirpath = f'{base_dirpath}/ICplot'
    os.makedirs(plot_dirpath, exist_ok=True)
    ic_group_filepath = f"{plot_dirpath}/{factor_name}.png"
    plot_quantile_info(ic_summary_bygroup.loc['IC Mean'], ic_group_filepath)

def do_portforlio_rough_return(data, factor_name, date, base_dirpath):
    dict_col_holding_time = {'1m': 20, '3m': 60, '5m': 100}
    rough_return_bygroup = factor_group_rough_return(data, dict_col_holding_time)
    
    table_dirpath = f'{base_dirpath}/ReturnTable/{factor_name}'
    os.makedirs(table_dirpath, exist_ok=True)
    rough_return_bygroup.to_csv(f'{table_dirpath}/{date}.csv')
    
    plot_dirpath = f'{base_dirpath}/ReturnPlot/{factor_name}'
    os.makedirs(plot_dirpath, exist_ok=True)
    group_return_filepath = f"{plot_dirpath}/{date}.png"
    plot_quantile_info(rough_return_bygroup, group_return_filepath)
    

def do_ts_ic_evaluate(data, factor_name, date, base_dirpath):
    ic_data = factor_timeSeries_information_coefficient(data)
    ic_summary = get_factor_ic_summary_info(ic_data)
    
    table_dirpath = f'{base_dirpath}/tsICtable/{factor_name}'
    os.makedirs(table_dirpath, exist_ok=True)
    ic_data.to_csv(f'{table_dirpath}/{date}_byCode.csv')
    ic_summary.to_csv(f'{table_dirpath}/{date}.csv')


def process_perFactor(factor_name, start_date, end_date, config, ret_type):
    base_dirpath = '/home/wangzirui/workspace/factor_eval_summary/param_optimized'
    datas = []
    for date in pd.date_range(start_date, end_date):
        date = date.strftime('%Y.%m.%d')
        ### Method One
        ### data = load_factor_and_return(date, factor_name, config)
        # Method Two
        data = load_factor_and_return_from_ddb(date, factor_name, config, ret_type)
        if data is not None:
            datas.append(data)
            # 时序截面的IC
            do_ts_ic_evaluate(data, factor_name, date, base_dirpath)
            # 计算分组的收益率
            do_portforlio_rough_return(data, factor_name, date, base_dirpath)
    
    # if len(datas) > 0:
    #     data = pd.concat(datas, axis=0)
    #     # 横截面的IC
    #     do_ic_evaluate(data, factor_name, base_dirpath)
            
            
if __name__ == "__main__":
    # read config file
    config = cfg.BasicConfig('config/config_scan.yml')

    start_date = '2023.10.09'
    end_date = '2023.10.09'
    
    ret_type = 'close_return'
    
    # import multiprocessing
    # PROCESSES = 2
    # print('Creating pool with %d processes\n' % PROCESSES)
    # pool = multiprocessing.Pool(PROCESSES)
    
    # args_list = []
    for facType in config['factors']:
        for i, factor_name in enumerate(config['factors'][facType]):
            # if (i >= 1):
            #     break
            print("Evaluating factor: ", factor_name, " ...", flush=True)
            # args_list.append((factor_name, start_date, end_date, config, ret_type))
            # pool.apply_async(process_perFactor, args=(factor_name, start_date, end_date, config, ret_type))
            process_perFactor(factor_name, start_date, end_date, config, ret_type)
    # pool.starmap_async(process_perFactor, args_list)
    
    # pool.close()
    # pool.join()
    
    # import yaml
    # pred_type='1m'
    # base_dir = r'/home/wangzirui/workspace/factor_ic_summary/factor_comb_top_n/bid_ask_price'
    # factor_filepath = os.path.join(base_dir, f'satisfied_factors_{pred_type}_without_hcorr.yml')
    # with open(factor_filepath, 'r') as f:
    #     factor_names = yaml.load(f, Loader=yaml.FullLoader)
    # for factor_name in factor_names:
    #     print("Evaluating factor: ", factor_name, " ...", flush=True)
    #     process_perFactor(factor_name, start_date, end_date, config)