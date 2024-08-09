import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "../../")
import alphalens
import yaml
import os

from factor_cal.config_loader import basic_config as cfg
from factor_cal.table.ddb_table import PriceTable, SecLevelFacTable
from factor_cal.utils import ddb_utils as du



if __name__ == "__main__":
    # read config file
    config = cfg.BasicConfig('../config/config_scan.yml')
    # obtain the ddb session
    s = du.DDBSessionSingleton().session

    base_dirpath = "/home/wangzirui/workspace/factor_eval_summary/param_optimized"
    factor_filepath = f"{base_dirpath}/satisfied_factors.yml"
    with open(factor_filepath, 'r') as f:
        factor_names = yaml.load(f, Loader=yaml.FullLoader)
    
    start_date = '2023.09.22'
    # end_date = '2023.09.25'
    end_date = '2023.09.30'
    start_time = '09:30:00'
    # end_time = '09:59:00'
    end_time = '14:57:00'
    dates = pd.date_range(start_date, end_date)
    
    output_base_dir = '/home/wangzirui/workspace/data/param_optimized'
    os.makedirs(output_base_dir, exist_ok=True)
    for date in dates:
        date = date.strftime('%Y.%m.%d')
        print("processing date: ", date, "...")

        facs_df = None
        
       
        for fac_name in factor_names:
            print("\tprocessing factor: ", fac_name, "...")
            fac_tb = SecLevelFacTable(config['factor_dbPath'], config['factor_tbName'])
            fac = fac_tb.load_factor(fac_name, date, start_time, end_time, sec_list=None)
            fac_df = s.loadTable(tableName=fac).toDF()
            if fac_df.empty:
                facs_df = None
                break
            
            fac_df.rename(columns={'value': fac_name}, inplace=True)
            fac_df.drop('factorname', axis=1, inplace=True)
            
            if facs_df is None:
                facs_df = fac_df
            else:
                facs_df = facs_df.merge(fac_df, on=['tradetime', 'securityid'], how='outer')
        if facs_df is None:
            print('\tNo data for date: ', date)
            continue        
        
        facs_df.to_pickle(f"{output_base_dir}/fac_{date}.pkl")
    
    