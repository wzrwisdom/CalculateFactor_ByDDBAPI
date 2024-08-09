import sys, os
sys.path.insert(0, '../')
import dolphindb as ddb
import numpy as np
import pandas as pd
import warnings
from scipy.stats import ConstantInputWarning

# igmore the warning of ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)

from factor_cal.feature import features as fe
from factor_cal.factor import factors as fa
from factor_cal.config_loader import basic_config as cfg
from factor_cal.utils import ddb_utils as du
from factor_cal.factor_eval.basic_evaluate import factor_timeSeries_information_coefficient


def get_return_bydate(date):
    # ret_type = 'bid_ask_return'
    ret_type = 'close_return'
    base_dir = f'/home/wangzirui/workspace/data/{ret_type}'
    ret_filepath = f'{base_dir}/{date}.pkl'
    if not os.path.exists(ret_filepath):
        return None
    ret_df = pd.read_pickle(ret_filepath)
    return ret_df

def evaluate_back(factor, ret):
    factor.set_index(['tradetime', 'securityid'], inplace=True)
    factor.rename(columns={'value': 'factor'}, inplace=True)
    factor.replace(np.inf, np.nan, inplace=True)
    
    factor_and_ret = factor.merge(ret, how='left', on=['tradetime', 'securityid'], sort=True)
    factor_and_ret = factor_and_ret.dropna()
      
    ic_data = factor_timeSeries_information_coefficient(factor_and_ret)
    return abs(ic_data.mean()['1m'])

def evaluate(factor, ret):
    factor.rename(columns={'value': 'factor'}, inplace=True)
    factor.replace(np.inf, np.nan, inplace=True)

    ret.reset_index(inplace=True)
    
    fac_arr = factor.pivot(index='tradetime', columns='securityid', values='factor')
    ret_arr = ret.pivot(index='tradetime', columns='securityid', values='1m')
    
    common_index = fac_arr.index.intersection(ret_arr.index)
    common_col = fac_arr.columns.intersection(ret_arr.columns)
    fac_com = fac_arr.loc[common_index, common_col]
    ret_com = ret_arr.loc[common_index, common_col]
    
    return abs(fac_com.corrwith(ret_com).mean())


if __name__ == "__main__":
    # read config file
    config = cfg.CalculateConfig('config/config_scan.yml')
    # obtain the ddb session
    s = du.DDBSessionSingleton().session
    
    features = fe.Features(config)
    factors = fa.Factors(config, features)
    
    factors.scan_factor_params(get_return_bydate, evaluate, base_dir="/home/wangzirui/workspace/data/param_scan")


    