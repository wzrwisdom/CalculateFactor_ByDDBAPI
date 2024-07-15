import warnings
import alphalens
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ConstantInputWarning
import sys
sys.path.insert(0, "../")
from alphalens import performance as perf
import dolphindb as ddb 

from factor_cal.config_loader import basic_config as cfg
from factor_cal.table.ddb_table import PriceTable, SecLevelFacTable
from factor_cal.utils import ddb_utils as du


# igmore the warning of ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)
# obtain the ddb session
s = du.DDBSessionSingleton().session

def get_clean_data(factor_name, date, config):
    fac_tb = SecLevelFacTable(config['factor_dbPath'], config['factor_tbName'])
    fac = fac_tb.load_factor(factor_name, date, config['start_time'], config['end_time'], sec_list=None)
    
    price_info = config['price_info']
    pc_tb = PriceTable(price_info['price_dbPath'], price_info['price_tbName'], price_info['time_col'], price_info['sec_col'], price_info['price_cols'])
    price = pc_tb.load_price(date, config['start_time'], config['end_time'], sec_list=None)
    
    fac_df = s.loadTable(tableName=fac).toDF()
    if (fac_df.empty):
        return None
    price_df = s.loadTable(tableName=price).toDF()
    
    # filter the dataframe
    start_time = pd.to_datetime('9:45').time()
    end_time = pd.to_datetime('14:45').time()
    price_df = price_df.set_index('tradetime').between_time(start_time, end_time).reset_index()
    fac_df = fac_df.set_index('tradetime').between_time(start_time, end_time).reset_index()
    
    
    fac_df.set_index(['tradetime', 'securityid'], inplace=True)
    fac_df = fac_df['value']
    fac_df = fac_df.sort_index(level=0)

    price_df = price_df.set_index(['tradetime', 'securityid'])
    prices = price_df['close'].unstack()
    
    
    data=alphalens.utils.get_clean_factor_and_forward_returns(
        fac_df, prices, quantiles=1, periods=(20, 60, 100))
    return data


def get_factor_ic_summary_info(data):
    group_neutral = False
    ic_data = perf.factor_information_coefficient(data, group_neutral)


    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = \
        ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0, nan_policy='omit')
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data, nan_policy='omit')
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data, nan_policy='omit')
    ic_summary_table['IC win rate'] = (ic_data > 0).sum() / ic_data.count()
    return ic_summary_table



def do_evaluate(factor_name):
    date1 = '2023.09.22'
    # date2 = '2023.09.22'
    date2 = '2024.02.20'

    datas = []
    for i in pd.date_range(date1, date2):
        data = get_clean_data(factor_name, i.strftime('%Y.%m.%d'), config)
        if data is not None:
            datas.append(data)

    data = pd.concat(datas, axis=0)

    ic_summary_table = get_factor_ic_summary_info(data)

    save_path = '/home/wangzirui/workspace/factor_ic_summary'
    ic_summary_table.to_csv(f'{save_path}/{factor_name}.csv')



# read config file
config = cfg.BasicConfig('config/config.yml')
# factor_name = "pv_corr"

for facType in config['factors']:
    for facName in config['factors'][facType]:
        print("Evaluating factor: ", facName, " ...", flush=True)
        do_evaluate(facName)

s.close()