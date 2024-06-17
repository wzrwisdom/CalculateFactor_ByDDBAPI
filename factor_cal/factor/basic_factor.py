import warnings
import pandas as pd
import numpy as np

from factor_cal.utils.tools import get_func_info
from factor_cal.config import regist_config, get_config
from factor_cal.table.ddb_table import FactorTable

FACTOR_FUNC = "factor_func"
regist_config(FACTOR_FUNC, {})


def register_facFunc(name=None, f=None):
    """register a factor function

    :params name: function name
    :params f: factor function itself
    """

    def regist(g):
        if name is None:
            my_name = g.__name__
        else:
            my_name = name
        config = get_config(FACTOR_FUNC)
        if my_name in config:
            warnings.warn("Override factor func {}".format(my_name))
        config[my_name] = g
        return g

    if f is None:
        return regist
    return regist(f)


class BasicFactor:
    def __init__(self, name, func_name, fac_args):
        self.name = name
        self.func = get_config(FACTOR_FUNC)[func_name] # according to the name to get the function
        self.fac_args = fac_args
        func_info = get_func_info(self.func)
        self.args_info = func_info['args']
        self.kwargs_info = func_info['kwargs']
        self.return_type = func_info['return_type']
        self.data = None  # nmpy array
        self.output_data = None  # pandas dataframe

    def set_args(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def set_dates_and_secs(self, dates, secs):
        self.dates = dates
        self.secs = secs

    def calculate(self):
        self.data = self.func(*self.args, **self.kwargs)
    
    
    
    def prepare_data(self):
        factor_vals = self.data.flatten(order='F')
        
        # Calculate the Cartesian product of dates and secs
        pos = np.meshgrid(self.dates, self.secs)
        date_vals, sec_vals = pos[0].flatten(), pos[1].flatten()

        df = pd.DataFrame({
            'tradetime': date_vals, 
            'securityid': sec_vals, 
            'factorname': self.name,
            'value': factor_vals
            })
        self.output_data = df

    def save(self, table: FactorTable):
        self.prepare_data()
        table.save(self.output_data)