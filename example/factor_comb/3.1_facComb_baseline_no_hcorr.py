import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import sys
sys.path.insert(0, "../")
import pickle
import yaml

from factor_cal.factor_eval.basic_evaluate import get_factor_ic_summary_info

def linear_regression(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model


def equal_weight_combination(X):
    X.mean(axis=1)
    res = X.mean(axis=1)
    return res


if __name__ == "__main__":
    save_path = '/home/wangzirui/workspace/factor_ic_summary/factor_comb_top_n'
    os.makedirs(save_path, exist_ok=True)
    
    start_date = '2023.09.22'
    end_date = '2023.09.22'
    # end_date = '2024.02.18'
    dates = pd.date_range(start_date, end_date)
    
    def preprocess(df):
        df= df.replace([np.inf, -np.inf], np.nan).dropna(subset=df.columns)
        return df
    
    total_df = pd.DataFrame()
    for date in dates:
        date = date.strftime('%Y.%m.%d')
        pickle_filepath = f'/data2/prepared_data/fac_ret_{date}.pkl'
        if os.path.exists(pickle_filepath):
            print("processing date: ", date, "...")
            cur_df = pd.read_pickle(pickle_filepath)
            cur_df= preprocess(cur_df)
            total_df = pd.concat([total_df, cur_df])
        
    X, y = total_df.drop(columns=total_df.columns[:5]), total_df[['1m']]
    
    
    with open(f'{save_path}/satisfied_factors_1m_no_hcorr.yml', 'r') as f:
        selected_col = yaml.load(f, yaml.FullLoader)
    X = X[selected_col]
    # baseline: linear regression
    model = linear_regression(X, y)
    
    with open(f'{save_path}/linear_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # # 从文件中加载模型
    # with open(f'{save_path}/linear_model.pkl', 'rb') as file:
    #     loaded_model = pickle.load(file)
    y_lr = model.fittedvalues
    
    eval_df = total_df[total_df.columns[:5]]
    eval_df['factor'] = y_lr
    eval_df.set_index(['tradetime', 'securityid'], inplace=True)
    eval_df.index.set_names(['date', 'asset'], inplace=True)
    ic_summary_table = get_factor_ic_summary_info(eval_df)
    ic_summary_table.to_csv(f'{save_path}/baseline_lg_no_hcorr.csv')
     
    # baseline: equal weight combination   
    y_ewc = equal_weight_combination(X)
    eval_df = total_df[total_df.columns[:5]]
    eval_df['factor'] = y_ewc
    eval_df.set_index(['tradetime', 'securityid'], inplace=True)
    eval_df.index.set_names(['date', 'asset'], inplace=True)
    ic_summary_table = get_factor_ic_summary_info(eval_df)
    ic_summary_table.to_csv(f'{save_path}/baseline_equal_weight_comb_no_hcorr.csv')
    print("done!")