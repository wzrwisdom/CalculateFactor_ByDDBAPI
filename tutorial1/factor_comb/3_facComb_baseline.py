import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import sys
sys.path.insert(0, "../../")
import pickle
import yaml

from factor_cal.config_loader import basic_config as cfg
from factor_cal.factor_eval.basic_evaluate import get_clean_data, factor_portfolio_return,\
    factor_information_coefficient, get_factor_ic_summary_info, plot_quantile_info, plot_quantile_netvalue,\
    factor_timeSeries_information_coefficient, factor_group_rough_return


# read config file
config = cfg.BasicConfig('../config/config_scan.yml')
    
def linear_regression(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model


def equal_weight_combination(X):
    X.mean(axis=1)
    res = X.mean(axis=1)
    return res

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

def do_linear_regression(fac_and_ret, base_dir, date=None, train_flag=False):
    X, y = fac_and_ret.drop(columns=fac_and_ret.columns[:5]), fac_and_ret[['1m']]
    if (train_flag):
        model = linear_regression(X, y)
        os.makedirs(f'{base_dir}/model', exist_ok=True)
        with open(f'{base_dir}/model/linear_model.pkl', 'wb') as file:
            pickle.dump(model, file)
    else:
        model_filepath = f'{base_dir}/model/linear_model.pkl'
        with open(model_filepath, 'rb') as file:
            loaded_model = pickle.load(file)
        X = sm.add_constant(X)
        y_pred = loaded_model.predict(X)

        fac_and_ret['factor'] = y_pred
        cols = list(fac_and_ret.columns[:5])
        cols.append('factor')
        eval_df = fac_and_ret[cols]

        eval_df.set_index(['tradetime', 'securityid'], inplace=True)
        eval_df = get_clean_data(eval_df, config['evaluation'])
        
        # do_ts_ic_evaluate(eval_df, 'lin_reg', date, base_dirpath)
        
        do_portforlio_rough_return(eval_df, 'lin_reg', date, base_dirpath)
        
def do_equal_weight_comb(fac_and_ret, base_dirpath, date):
    X = fac_and_ret.drop(columns=fac_and_ret.columns[:5])
    y_ewc = equal_weight_combination(X)
    
    fac_and_ret['factor'] = y_ewc
    cols = list(fac_and_ret.columns[:5])
    cols.append('factor')
    eval_df = fac_and_ret[cols]
    
    eval_df.set_index(['tradetime', 'securityid'], inplace=True)
    eval_df = get_clean_data(eval_df, config['evaluation'])
    
    # do_ts_ic_evaluate(eval_df, 'ewc', date, base_dirpath) 
    
    do_portforlio_rough_return(eval_df, 'ewc', date, base_dirpath)
        

def construct_baseline(fac_and_ret, base_dirpath, date):
    do_linear_regression(fac_and_ret, base_dirpath, date)
    
    do_equal_weight_comb(fac_and_ret, base_dirpath, date)

if __name__ == "__main__":    
    
    
    
    base_dirpath = '/home/wangzirui/workspace/factor_eval_summary/param_optimized'
    start_date = '2023.09.22'
    # end_date = '2023.09.22'
    end_date = '2023.09.30'
    train_flag = True
    dates = pd.date_range(start_date, end_date)
    
    def preprocess(df):
        df= df.replace([np.inf, -np.inf], np.nan).dropna(subset=df.columns)
        return df
    
    with open(f'{base_dirpath}/satisfied_factors_nohl.yml', 'r') as f:
        selected_col = yaml.load(f, yaml.FullLoader)
    
    index = 0
    for date in dates:
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
            # total_df = pd.concat([total_df, cur_df])

            if index == 0 and train_flag:
                do_linear_regression(cur_df, base_dirpath, train_flag=train_flag)
                index += 1
            construct_baseline(cur_df, base_dirpath, date)
            
            
    
    print("done!")