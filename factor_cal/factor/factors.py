from .basic_factor import BasicFactor
from factor_cal.table.ddb_table import SecLevelFacTable


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
            self.fac_category[cat_name] = list(cat_facs.keys())
            self._load_oneCategory_factors(cat_name, cat_facs)

    def replace_args_by_featName(self, args):
        ret = []
        for arg in args:
            if (arg in self.features.get_feat_names()):
                ret.append(self.features.get_feature(arg).get_data())
            else:
                raise KeyError(f"[Feature]{arg} not found in features")
        return ret

    def _load_oneCategory_factors(self, cat_name, cat_facs):
        for fac_name, fac_info in cat_facs.items():
            if (fac_name in self.fac_category[cat_name]):
                raise Warning(f"[Factor]{fac_name} already exists in [Fac Catetory]{cat_name}")
            
            fac = BasicFactor(fac_name, fac_info['func_name'], fac_info['args'])
            args = self.replace_args_by_featName(fac_info['args'])
            fac.set_args(*args, **fac_info['kwargs'])

            self.fac_category[cat_name].append(fac_name)
            self.fac_dict[fac_name] = fac

    def process(self):
        for fac_name, fac in self.fac_dict.items():
            fac.calculate()
            self.features.set_dates_and_secs(fac.fac_args[0])
            fac.set_dates_and_secs(self.features.get_dates(), self.features.get_secs())
            fac.save(self.factor_table)

