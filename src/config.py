

import inspect
from os.path import abspath, dirname, join

import torch
import yaml


class Config:
    def __init__(self, config_path: str):
        self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path):        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            setattr(self, key, value)
        
        # Override device based on availability
        self.device = 'cpu' if not torch.cuda.is_available() else self.device
        
        # Convert relative paths to absolute
        base_dir = dirname(dirname(config_path))
        for key in ['OUTPUT_DIR', 'train_datadir', 'train_csv', 'test_soundscapes', 
                    'submission_csv', 'taxonomy_csv']:
            if hasattr(self, key):
                path = getattr(self, key)
                if path.startswith('../'):
                    setattr(self, key, abspath(join(base_dir, path.lstrip('../'))))

    def to_dict(self):
        pr = {}
        for name in dir(self):
            value = getattr(self, name)
            if not name.startswith('__') and not inspect.ismethod(value):
                pr[name] = value
        return pr

    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0, 1]
