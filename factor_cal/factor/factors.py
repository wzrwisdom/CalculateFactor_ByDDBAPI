import gc
import datetime
import pandas as pd
from itertools import product
from copy import deepcopy
import os

from .basic_factor import BasicFactor, create_factor_by_str
from factor_cal.table.ddb_table import SecLevelFacTable
from factor_cal.utils.tools import show_memory


class Factors:
    def __init__(self, config, features):
        self.config = config
        self.factors_info = config['factors']
        self.features = features
        self.fac_category = dict()  # key is the factor category name, value is a list of factors' names in each category
        self.fac_dict = dict()  # key is factor name, value is BasicFactor object
        self.load_factors()
        self.create_factor_table()

    def create_factor_table(self):
        self.factor_table = SecLevelFacTable(self.config['factor_dbPath'], self.config['factor_tbName'])
        self.factor_table.create()

    def load_factors(self):
        for cat_name, cat_facs in self.factors_info.items():
            # get all factors' names in each category
            self.fac_category[cat_name] = []
            self._load_oneCategory_factors(cat_name, cat_facs)
            self.fac_category[cat_name] = list(cat_facs.keys())

    def _load_oneCategory_factors(self, cat_name, cat_facs):
        for fac_name, fac_info in cat_facs.items():
            if (fac_name in self.fac_category[cat_name]):
                raise Warning(f"[Factor]{fac_name} already exists in [Fac Catetory]{cat_name}")
            
            if isinstance(fac_info, dict) and 'formula' in fac_info:
                fac = create_factor_by_str(fac_name, fac_info)
            else:
                fac = BasicFactor(fac_name, fac_info['func_name'], fac_info['args'], fac_info['kwargs'], fac_info['desc'])
                # args = self.features.get_data_by_featList(fac_info['args'])
                # fac.set_args(*args, **fac_info['kwargs'])

            if isinstance(fac_info, dict) and 'param_scan' in fac_info:
                fac.param_scan = fac_info['param_scan']
            
            self.fac_category[cat_name].append(fac_name)
            self.fac_dict[fac_name] = fac

    def set_best_param(self, base_dir, start_date, end_date):
        for fac_name, fac in self.fac_dict.items():
            dfs = []
            for date in pd.date_range(start_date, end_date):
                date = date.strftime('%Y.%m.%d')
                param_filepath = f'{base_dir}/{fac_name}/{date}.pkl'
                if os.path.exists(param_filepath):
                    df = pd.read_pickle(param_filepath)
                    dfs.append(df)
            try:
                params_all = pd.concat([df['metric'] for df in dfs], axis=1)
            except ValueError as e:
                print(f"There is no params scan file factor:{fac_name}")
                continue
                
            params_avg = dfs[0].copy()
            params_avg['metric'] = params_all.T.mean()
            
                
            idx = params_avg['metric'].argmax()
            metric = params_avg.iloc[idx]['metric']
            s = params_avg.iloc[idx].drop('metric')
            
            s_index = list(s.index)
            m = max([int(i[-1]) for i in s_index])
            params = [{} for i in range(m)]
            for i in s_index:
                key = i[:-1]
                pos = int(i[-1])-1
                value = s[i]
                params[pos][key] = value
            fac.reset_kwargs_info(params)
            if metric < 0:
                fac.negative = True
                
            
    
    def process(self):
        start_date = datetime.datetime.strptime(self.config['start_date'], "%Y.%m.%d")
        end_date = datetime.datetime.strptime(self.config['end_date'], "%Y.%m.%d")

        cur_date = start_date
        while cur_date <= end_date:
            print("------------------------------------")
            print("Current date: ", cur_date, flush=True)
            show_memory("before calculate")
            for fac_name, fac in self.fac_dict.items():
                try:
                    fac.calculate(self.features, datetime.datetime.strftime(cur_date, "%Y.%m.%d"))
                    fac.set_dates_and_secs_v2(self.features)
                    # if isinstance(fac.args_info[0], BasicFactor):
                    #     self.features.set_dates_and_secs(fac.args_info[0].get_dates(), fac.args_info[0].get_secs())
                    # else:
                    #     self.features.set_dates_and_secs_by_feat(fac.args_info[0])
                    # fac.set_dates_and_secs(self.features.get_dates(), self.features.get_secs())
                    fac.save(self.factor_table)
                except ValueError as e:
                    print(f"\tError: {e}")
                    continue
            gc.collect()
            show_memory("after calculate")
            cur_date += datetime.timedelta(days=1)
            
    
    def scan_factor_params(self, func_get_return_bydate, func_evaluate, base_dir):
        """

        Args:
            func_get_return_bydate (function): get a DataFrame about return information by giving a date
            func_evaluate (function): With factor and return as input, it calculates a metric to evaluate this factor. Following a rule: "Larger the metric, better the factor".
        """
        start_date = datetime.datetime.strptime(self.config['start_date'], "%Y.%m.%d")
        end_date = datetime.datetime.strptime(self.config['end_date'], "%Y.%m.%d")
        
        for fac_name, fac in self.fac_dict.items():
            print(f'[Scaning parameters] factor: {fac_name}')
            if not hasattr(fac, 'param_scan'):
                print("\t There is no param_scan in its configuration!")
                continue
            save_dir = f"{base_dir}/{fac_name}"
            os.makedirs(save_dir, exist_ok=True)
            for cur_date in pd.date_range(start_date, end_date):
                # for test
                # if cur_date > start_date:
                #     break
                print("------------------------------------")
                print("Current date: ", cur_date, flush=True)
                show_memory("before calculate")
                cur_date = cur_date.strftime('%Y.%m.%d')
                
                # for test
                params_combination = get_params_combination(fac.param_scan)
                ret_data = func_get_return_bydate(cur_date)
                if ret_data is None:
                    print(f"[Warning]There is no data for {fac_name}-{cur_date}")
                    continue
                
                # scan_result = self.do_scan_params_combination_multicore(fac, ret_data, params_combination, cur_date, func_evaluate)
                scan_result = self.do_scan_params_combination(fac, ret_data, params_combination, cur_date, func_evaluate)
                
                save_filepath = f'{save_dir}/{cur_date}.csv'
                scan_result.to_csv(save_filepath)

    # def scan_process1(self, factor, return_data, features, kw_params, date, func_evaluate):
    #     factor_tmp = deepcopy(factor)
    #     factor_tmp.reset_kwargs_info(deepcopy(kw_params))
    #     factor_tmp.calculate(features, date)
    #     factor_tmp.set_dates_and_secs_v2(features)
    #     factor_data = factor_tmp.prepare_data()
    #     metric_value = func_evaluate(factor_data, return_data)
        
    #     return metric_value, kw_params
    
    # def scan_process(self, factor, return_data, features, kw_params, date, func_evaluate):
    #     factor_tmp = deepcopy(factor)
    #     factor_tmp.reset_kwargs_info(deepcopy(kw_params))
    #     factor_tmp.calculate(features['features'], date)
    #     factor_tmp.set_dates_and_secs_v2(features['features'])
    #     factor_data = factor_tmp.prepare_data()
    #     metric_value = func_evaluate(factor_data, return_data)
        
    #     return metric_value, kw_params
        
    # def do_scan_params_combination_multicore(self, factor, return_data, params_combination, date, func_evaluate):
    #     import multiprocessing, time
    #     with multiprocessing.Manager() as manager:
    #         pool = multiprocessing.Pool(1)
    #         results = []
    #         shared_obj = manager.dict()
    #         for i, kw_params in enumerate(params_combination[1:]):
    #             if i == 0:    
    #                 r =  self.scan_process1(factor, return_data, self.features, kw_params, date, func_evaluate)
    #                 shared_obj.update({'features':self.features})
    #                 results.append(r)
    #                 continue
                
    #             r = pool.apply_async(self.scan_process, args=(factor, return_data, shared_obj, kw_params, date, func_evaluate))
    #             results.append(r)
                
    #         for i in range(1, len(results)):
    #             results[i] = results[i].get()
    #             # results[i] = tmp

    #         pool.close()
    #         pool.join()
        
    #     scan_result = {}
    #     for i, r in enumerate(results):
    #         metric_value = r.get()[0]
    #         kw_params = r.get()[1]
    #         print(f'metric_value: {metric_value} | {kw_params}', flush=True)
    #         if (i == 0):
    #             for j, par in enumerate(kw_params):
    #                 for name, value in par.items():
    #                     scan_result[name+str(j+1)] = [value]
    #             scan_result['metric'] = [metric_value]
                
    #         for j, par in enumerate(kw_params):
    #             for name, value in par.items():
    #                 scan_result[name+str(j+1)].append(value)
    #         scan_result['metric'].append(metric_value)
    #     return pd.DataFrame(scan_result)
        
        
        
    
    def do_scan_params_combination(self, factor, return_data, params_combination, date, func_evaluate):
        best_one = None
        scan_result = {}
        for i, kw_params in enumerate(params_combination):
            factor.reset_kwargs_info(deepcopy(kw_params))
            factor.calculate(self.features, date)
            if (i == 0):
                factor.set_dates_and_secs_v2(self.features)
                        
            factor_data = factor.prepare_data()
            metric_value = func_evaluate(factor_data, return_data)
            
            if (i == 0):
                best_one = (metric_value, kw_params, factor.get_data())
                for j, par in enumerate(kw_params):
                    for name, value in par.items():
                        scan_result[name+str(j+1)] = [value]
                scan_result['metric'] = [metric_value]
                continue
            
            for j, par in enumerate(kw_params):
                for name, value in par.items():
                    scan_result[name+str(j+1)].append(value)
            scan_result['metric'].append(metric_value)
            if (best_one[0] is None) or (metric_value > best_one[0]):
                best_one = (metric_value, kw_params, factor.get_data())
        
        print(f"\t[Best params]: {best_one[1]}")
        print(f"\t[Best metric]: {best_one[0]}")
        # factor.reset_kwargs_info(best_one[1])
        # factor.set_data(best_one[2])
        # factor.save(self.factor_table)
        return pd.DataFrame(scan_result)

def get_params_combination(param_scan):
    params_list = []
    for item in param_scan:
        for one_param in item:
            params_list.append(one_param[1])
        
    params_combination = []
    for params in product(*params_list):
        result = []
        index = 0
        for item in param_scan:
            tmp_dict = {}
            for one_param in item:
                tmp_dict[one_param[0]] = params[index]
                index+=1
            result.append(tmp_dict)
        params_combination.append(result)
    return params_combination
                
            