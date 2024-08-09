import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from alphalens import performance as perf
from scipy import stats
import os
import sys
sys.path.insert(0, "../../")
import yaml

from factor_cal.config_loader import basic_config as cfg
from factor_cal.factor_eval.basic_evaluate import get_clean_data, factor_portfolio_return,\
    factor_information_coefficient, get_factor_ic_summary_info, plot_quantile_info, plot_quantile_netvalue,\
    factor_timeSeries_information_coefficient, factor_group_rough_return


# read config file
config = cfg.BasicConfig('../config/config_scan.yml')

def preprocess(df):
    df.dropna(inplace=True)
    df= df.replace([np.inf, -np.inf], np.nan).dropna(subset=df.columns)
    return df


def do_ts_ic_evaluate(data, factor_name, date, base_dirpath):
    ic_data = factor_timeSeries_information_coefficient(data)
    ic_summary = get_factor_ic_summary_info(ic_data)
    
    table_dirpath = f'{base_dirpath}/tsICtable/{factor_name}'
    os.makedirs(table_dirpath, exist_ok=True)
    ic_data.to_csv(f'{table_dirpath}/{date}_byCode.csv')
    ic_summary.to_csv(f'{table_dirpath}/{date}.csv')

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

if __name__ == "__main__":
    

    start_date = '2023.09.22'
    end_date = '2023.09.30'
    base_dirpath = '/home/wangzirui/workspace/factor_eval_summary/param_optimized'
    
    dates = pd.date_range(start_date, end_date)
        
    with open(f'{base_dirpath}/satisfied_factors_nohl.yml', 'r') as f:
        selected_col = yaml.load(f, yaml.FullLoader)
    
    # load model
    model = xgb.Booster()
    model.load_model(f'{base_dirpath}/model/xgboost_model_no_hcorr.json')

    for date in pd.date_range(start_date, end_date):
        date = date.strftime('%Y.%m.%d')
        fac_filepath = f'/home/wangzirui/workspace/data/param_optimized/fac_{date}.pkl'
        ret_filepath = f'/home/wangzirui/workspace/data/bid_ask_return/{date}.pkl'
        if os.path.exists(fac_filepath) and os.path.exists(ret_filepath):
            print("processing date: ", date, "...")

            ret_df = pd.read_pickle(ret_filepath)
            fac_df = pd.read_pickle(fac_filepath)
            fac_df = fac_df[['tradetime', 'securityid'] + selected_col].copy()
            
            cur_df = ret_df.merge(fac_df, how='right', on=['tradetime', 'securityid'], sort=True)
            cur_df= preprocess(cur_df)
            
            X_test, y_test = cur_df.drop(columns=cur_df.columns[:5]), cur_df[['1m']]


            dtest_reg = xgb.DMatrix(X_test, y_test)
            
            preds = model.predict(dtest_reg)

            cur_df['factor'] = preds
            
            cols = list(cur_df.columns[:5])
            cols.append('factor')
            eval_df = cur_df[cols]
            
            eval_df.set_index(['tradetime', 'securityid'], inplace=True)
            eval_df = get_clean_data(eval_df, config['evaluation'])
            
            # do_ts_ic_evaluate(eval_df, 'xgboost', date, base_dirpath)
            
            do_portforlio_rough_return(eval_df, 'xgboost', date, base_dirpath)
            
