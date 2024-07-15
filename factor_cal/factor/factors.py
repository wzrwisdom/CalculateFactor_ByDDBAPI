import gc
import datetime

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
            
            if isinstance(fac_info, dict):
                fac = BasicFactor(fac_name, fac_info['func_name'], fac_info['args'], fac_info['kwargs'])
                # args = self.features.get_data_by_featList(fac_info['args'])
                # fac.set_args(*args, **fac_info['kwargs'])
            elif isinstance(fac_info, str):
                fac = create_factor_by_str(fac_name, fac_info)

            self.fac_category[cat_name].append(fac_name)
            self.fac_dict[fac_name] = fac

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

