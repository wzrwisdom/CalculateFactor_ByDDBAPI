import sys
sys.path.insert(0, "../")
import dolphindb as ddb 
import numpy as np
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

factors.set_best_param(base_dir="/home/wangzirui/workspace/data/param_scan", date="2023.09.22")

factors.process()

s.close()
