import sys
sys.path.insert(0, '../')
import dolphindb as ddb
import numpy as np

from factor_cal.feature import features as fe
from factor_cal.factor import factors as fa
from factor_cal.config_loader import basic_config as cfg
from factor_cal.utils import ddb_utils as du


def get_return_bydate(date):
    pass

def evaluate(factor, ret):
    pass


if __name__ == "__main__":
    # read config file
    config = cfg.CalculateConfig('config/config.yml')
    # obtain the ddb session
    s = du.DDBSessionSingleton().session
    
    features = fe.Features(config)
    factors = fa.Factors(config, features)
    
    factors.scan_factor_params(get_return_bydate, evaluate)


    