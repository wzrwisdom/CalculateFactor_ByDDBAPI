import functools
from factor_cal.utils.ddb_utils import s
from factor_cal.feature.feat_table import DDB_FeatTable

class Features:
    def __init__(self, config):
        """
        Initializes a Features object.

        Args:
            config (dict): A dictionary containing configuration parameters.

        Attributes:
            config (dict): The configuration parameters.
            start_time (str): The start time.
            end_time (str): The end time.
            sec_list (tuple): A tuple of security codes.
            feat_dict (dict): A dictionary of feature names and Feature objects.
        """
        self.config = config
        self.start_time = config['start_time']
        self.end_time = config['end_time']
        self.sec_list = tuple(config['sec_list'])
        self.feat_dict = {}
        self.load_features()
    
    def get_feat_names(self):
        """
        Returns a list of feature names.

        Returns:
            list: A list of feature names.
        """
        return list(self.feat_dict.keys())

    def load_features(self):
        """
        Loads the features from the configuration.

        Returns:
            list: A list of feature names.
        """
        for ddb_info in self.config['features']:
            ddb_name = ddb_info['ddb_name']
            tbs = ddb_info['tb_features']
            for tb_name, tb_info in tbs.items():
                # Create a DDB_Table object for each table
                ddb_table = DDB_FeatTable(ddb_name, tb_name, tb_info['time_col'], tb_info['sec_col'])

                for feat_nkname, feat_colname in tb_info['feat_cols'].items():
                    feat = Feature(ddb_table, feat_colname)
                    if (feat_nkname in self.feat_dict):
                        raise Warning(f"Feature name {feat_nkname} already exists")
                    self.feat_dict[feat_nkname] = feat
        return self.get_feat_names()
    
    def set_dates_and_secs(self, feat_name):
        if not self.feat_dict:
            raise Warning("No features loaded")
        self.dates = self.feat_dict[feat_name].get_dates()
        self.secs = self.feat_dict[feat_name].get_secs()
    
    def get_dates(self):
        return self.dates
    
    def get_secs(self):
        return self.secs
    
    def get_data_by_featList(self, feat_list):
        """
        Retrieves data for a list of features.

        Args:
            feat_list (list): A list of feature names.

        Returns:
            List: A list of feature in the given order.
        """
        ret = []
        for feat_name in feat_list:
            if (feat_name in self.get_feat_names()):
                ret.append(self.get_feature(feat_name).get_data())
            else:
                raise KeyError(f"Feature name {feat_name} not found") 
        return ret

    @functools.lru_cache(maxsize=20)
    def get_feature(self, feat_name):
        """
        Retrieves a feature by its name.

        Args:
            feat_name (str): The name of the feature.

        Returns:
            object: The Feature object.

        Raises:
            ValueError: If the feature name is not found.
        """
        if feat_name not in self.feat_dict:
            raise KeyError(f"Feature name {feat_name} not found")
        feat = self.feat_dict[feat_name]
        print(f"Loading [feature]{feat_name} from DolphinDB server")
        feat.load_data(self.start_time, self.end_time, self.sec_list)
        return feat



class Feature:
    def __init__(self, ddb_tb, feat_colname):
        self.ddb_tb = ddb_tb
        self.feat_colname = feat_colname

    def load_data(self, start_time=None, end_time=None, sec_list=None):
        self.data = self.ddb_tb.get_feature(self.feat_colname, start_time, end_time, sec_list)
        
    def get_data(self):
        return self.data[0]
    
    def get_dates(self):
        return self.data[1]
    
    def get_secs(self):
        return self.data[2]