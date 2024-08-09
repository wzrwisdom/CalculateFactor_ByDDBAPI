import sys
sys.path.insert(0, "../")
import dolphindb as ddb 
import numpy as np
import pandas as pd

from factor_cal.feature import features as fe
from factor_cal.factor import factors as fa
from factor_cal.config_loader import basic_config as cfg
from factor_cal.utils import ddb_utils as du
from factor_cal.utils import tools as tl
from factor_cal.factor import factor_func as ff

# read config file
config = cfg.CalculateConfig('config/config_scan.yml')
# obtain the ddb session
s = du.DDBSessionSingleton().session

features = fe.Features(config)
factors = fa.Factors(config, features)

factors.set_best_param(base_dir="/home/wangzirui/workspace/data/param_scan", start_date='2023.09.22', end_date='2023.09.30')

factors.process()

s.close()

# def 


# if __name__ == "__main__":
#     import multiprocessing
#     PROCESSES = 2
#     print('Creating pool with %d processes\n' % PROCESSES)
#     pool = multiprocessing.Pool(PROCESSES)
#     # read config file
#     config = cfg.CalculateConfig('config/config_scan.yml')
    
#     start_date, end_date = config['start_date'], config['end_date']
    
#     for date in pd.date_range(start_date, end_date):
#         date = date.strftime('%Y.%m.%d')

#         config['start_date'] = date
#         config['end_date'] = date
    

    
    
