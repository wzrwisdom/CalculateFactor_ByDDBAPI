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
    def __init__(self, name, func_name, arg_names):
        self.name = name
        self.func = get_config(FACTOR_FUNC)[func_name] # according to the name to get the function
        func_info = get_func_info(self.func)
        self.arg_names = arg_names
        self.args_info = func_info['args']
        self.kwargs_info = func_info['kwargs']
        self.return_type = func_info['return_type']
        self.data = None  # nmpy array
        self.output_data = None  # pandas dataframe
    
    def prepare_args(self, args, features):
        ret = []
        for arg in args:
            if isinstance(arg, BasicFactor):
                arg.calculate()
                arg = arg.get_data()
            elif isinstance(arg, str):
                arg = features.get_feature(arg).get_data()
            ret.append(arg)
        self.args = ret
        
    def prepare_kwargs(self, kwargs):
        ret = {}
        for i in self.kwargs_info:
            kwarg_name = i[0]
            kwarg_type = i[1]
            if (kwarg_type == str) and (kwarg_name in kwargs):
                ret[kwarg_name] = kwargs.get(kwarg_name, "")
            elif kwarg_name in kwargs:
                ret[kwarg_name] = eval(kwargs[kwarg_name])
        self.kwargs = ret
        
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
    
    def get_data(self):
        return self.data
        

def create_factor_by_str(fac_name, fac_str, features):
    # fac_str="ret(close, corr(close, volume), shift=50)"
    def parse_fac_str(fac_name, fac_str) -> BasicFactor:

        # Split the fac_str into function name and arguments
        func_name, args_str = fac_str.split("(", 1)
        args_str = args_str.rstrip(")")
        
        # Split the arguments string into individual arguments
        args = args_str.split(",")
        
        # Recursively parse each argument
        parsed_args = []
        parsed_kwargs = {}
        for arg in args:
            arg = arg.strip()
            if "(" in arg:
                parsed_arg = parse_fac_str("", arg)
            elif "=" in arg:
                parsed_kwargs[arg.split("=")[0].strip()] = arg.split("=")[1].strip()
            else:
                # Handle other arguments
                parsed_arg = arg
                parsed_args.append(parsed_arg)

        # Get the function from the config
        fac = BasicFactor(fac_name, func_name, arg_names=parsed_args)
        fac.prepare_args(parsed_args, features)
        fac.prepare_kwargs(parsed_kwargs)
        return fac
    return parse_fac_str(fac_name, fac_str)