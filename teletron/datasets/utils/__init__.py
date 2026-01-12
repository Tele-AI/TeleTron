import json
import os
import pickle
import stat
from importlib import import_module

import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

def load_file(file_path, **kwargs):
    if file_path.endswith(".pkl") or file_path.endswith(".pickle"):
        data = pickle.load(open(file_path, "rb"), **kwargs)
    elif file_path.endswith(".json"):
        data = json.load(open(file_path, "r"), **kwargs)
    elif file_path.endswith(".yaml") or file_path.endswith("yml"):
        kwargs.setdefault("Loader", Loader)
        data = yaml.load(open(file_path, "r"), **kwargs)
    else:
        assert False
    return data


def save_file(file_path, data, **kwargs):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_path.endswith(".pkl") or file_path.endswith(".pickle"):
        pickle.dump(data, open(file_path, "wb"), **kwargs)
    elif file_path.endswith(".json"):
        kwargs.setdefault("indent", 4)
        json.dump(data, open(file_path, "w"), **kwargs)
    elif file_path.endswith(".yaml") or file_path.endswith("yml"):
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(data, open(file_path, "w"), **kwargs)
    else:
        assert False


def get_data_dir():
    return os.environ.get("VAST_DATASETS_DIR", "./data/")

def import_function(function_name, sep="."):
    parts = function_name.split(sep)
    module_name = ".".join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])