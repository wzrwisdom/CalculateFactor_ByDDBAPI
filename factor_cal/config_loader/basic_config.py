import yaml
import yaml_include
import os


def get_dir(file_path):
    file_abspath = os.path.abspath(file_path)
    file_folder = os.path.dirname(file_abspath)
    return file_folder


class BasicConfig:
    def __init__(self, file_path):
        self.file_path = file_path
        self.filedir_path = get_dir(file_path)
        self.config = self.load_config()

    def load_config(self):
        yaml.add_constructor("!inc", yaml_include.Constructor(base_dir=self.filedir_path))
        with open(self.file_path, 'r') as file:
            ret = yaml.full_load(file)
        return ret

    def __getitem__(self, key):
        return self.config.get(key, None)
    
    def __setitem__(self, key, value):
        self.config[key] = value

    def __str__(self):
        return str(self.config)

