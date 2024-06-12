import functools
from factor_cal.utils import ddb_utils as du
from factor_cal.feature.ddb_table import DDB_Table

# Obtain the session object from the singleton instance
s = du.DDBSessionSingleton().get_session()


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
                ddb_table = DDB_Table(ddb_name, tb_name, tb_info['time_col'], tb_info['sec_col'])

                for feat_nkname, feat_colname in tb_info['feat_cols'].items():
                    feat = Feature(ddb_table, feat_colname)
                    if (feat_nkname in self.feat_dict):
                        raise ValueError(f"Feature name {feat_nkname} already exists")
                    self.feat_dict[feat_nkname] = feat
        return self.get_feat_names()
    
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
            raise ValueError(f"Feature name {feat_name} not found")
        feat = self.feat_dict[feat_name]
        print(f"Loading [feature]{feat_name} from DolphinDB server")
        return feat.read_data(self.start_time, self.end_time, self.sec_list)



class Feature:
    def __init__(self, ddb_tb, feat_colname):
        self.ddb_tb = ddb_tb
        self.feat_colname = feat_colname

    @functools.lru_cache(maxsize=None)
    def read_data(self, start_time=None, end_time=None, sec_list=None):
        data = self.ddb_tb.get_feature(self.feat_colname, start_time, end_time, sec_list)
        return data[0]