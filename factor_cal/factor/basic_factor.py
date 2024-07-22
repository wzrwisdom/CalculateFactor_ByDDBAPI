import warnings
import pandas as pd
import numpy as np
import re

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
    def __init__(self, name, func_name, args_info, kwargs_info, desc=None):
        self.name = name
        self._desc = desc
        self.func = get_config(FACTOR_FUNC)[func_name] # according to the name to get the function
        func_type = get_func_info(self.func)
        self.args_info = args_info
        self.kwargs_info = kwargs_info
        self.args_type = func_type['args']
        self.kwargs_type = func_type['kwargs']
        self.return_type = func_type['return_type']
        self.data = None  # nmpy array
        self.output_data = None  # pandas dataframe
    
    @property
    def desc(self):
        return self._desc
    
    @desc.setter
    def desc(self, desc):
        self._desc = desc
        
    def __str__(self):
        args_str = []
        kwargs_str = []
        for i in self.args_info:
            if isinstance(i, BasicFactor):
                args_str.append(str(i))
            else:
                args_str.append(i)
        for k, v in self.kwargs_info.items():
            kwargs_str.append(f"{k}={v}")
        if len(kwargs_str) > 0:
            return self.func.__name__ + "(" + ", ".join(args_str) + ", " + ", ".join(kwargs_str) + ")"
        else:
            return self.func.__name__ + "(" + ", ".join(args_str) + ")"
        
    
    def prepare_args(self, args, features, date):
        ret = []
        parsed_args = []
        for arg in args:
            try: 
                if isinstance(arg, BasicFactor):
                    arg.calculate(features, date)
                    arg.set_dates_and_secs_v2(features)
                    parsed_args.append(arg)
                elif isinstance(arg, str):
                    parsed_args.append(features.get_feature(arg, date))
            except TypeError:
                raise ValueError(f"Feature {arg} is not available ")
        
        common_secs, common_dates = None, None
        try:
            for arg in parsed_args:
                if common_dates is None:
                    common_dates = arg.get_dates()
                else:
                    common_dates = np.intersect1d(common_dates, arg.get_dates())
                if common_secs is None:
                    common_secs = arg.get_secs()
                else:
                    common_secs = np.intersect1d(common_secs, arg.get_secs())
        except TypeError:
                raise ValueError(f"Feature {arg} is not available ")  

        for arg in parsed_args:
            cur_dates = arg.get_dates()
            cur_secs = arg.get_secs()
            common_dates_indices = [np.where(cur_dates == date)[0][0] for date in common_dates]
            common_secs_indices = [np.where(cur_secs == sec)[0][0] for sec in common_secs]
            arg.set_dates_and_secs(common_dates, common_secs)
            arg.set_data(arg.get_data()[common_dates_indices][:, common_secs_indices])
            ret.append(arg.get_data())
            
        self.args = ret
            
        
    def prepare_kwargs(self, kwargs):
        ret = {}
        for i in self.kwargs_type:
            kwarg_name = i[0]
            kwarg_type = i[1]
            kwarg_default = i[2]
            if (kwarg_type != str) and (kwarg_name in kwargs) and isinstance(kwargs[kwarg_name], str):
                ret[kwarg_name] = kwarg_type(eval(kwargs[kwarg_name]))
            else:
                ret[kwarg_name] = kwargs.get(kwarg_name, kwarg_default)
        self.kwargs = ret
        
    def set_args(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def set_dates_and_secs(self, dates, secs):
        self.dates = dates
        self.secs = secs
        
    def set_dates_and_secs_v2(self, features):
        if isinstance(self.args_info[0], BasicFactor):
            features.set_dates_and_secs(self.args_info[0].get_dates(), self.args_info[0].get_secs())
        else:
            features.set_dates_and_secs_by_feat(self.args_info[0])
        self.set_dates_and_secs(features.get_dates(), features.get_secs())

    def get_dates(self):
        return self.dates
    
    def get_secs(self):
        return self.secs

    def calculate(self, features, date):
        self.prepare_args(self.args_info, features, date)
        self.prepare_kwargs(self.kwargs_info)
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
    
    def set_data(self, data):
        self.data = data
        

def create_factor_by_str(fac_name, fac_info):
    # fac_str="ret(close, corr(close, volume), shift=50)"
    if isinstance(fac_info, dict):
        fac_str = fac_info['formula']
    else:
        fac_str = fac_info
    def parse_args(s):
        args = []
        balance = 0
        current_arg = []
        for char in s:
            if char == ',' and balance == 0:
                args.append(''.join(current_arg).strip())
                current_arg = []
            else:
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                current_arg.append(char)
        if current_arg:
            args.append(''.join(current_arg).strip())
        return args
    
    match = re.match(r'(\w+)\((.*)\)', fac_str)
    if match:
        func_name = match.group(1)
        args_str = match.group(2)
        args = parse_args(args_str)
        args = [create_factor_by_str("", arg) for arg in args]
        parsed_args = []
        parsed_kwargs = {}
        for arg in args:
            if isinstance(arg, BasicFactor):
                parsed_args.append(arg)
            elif isinstance(arg, str):
                arg = arg.strip()
                if "=" in arg:
                    parsed_kwargs[arg.split("=")[0].strip()] = arg.split("=")[1].strip()
                else:
                    parsed_args.append(arg)
        return BasicFactor(fac_name, func_name, parsed_args, parsed_kwargs, desc=fac_info['desc'] if isinstance(fac_info, dict) else None)
    else:
        return fac_str
