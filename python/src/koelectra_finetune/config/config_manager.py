from abc import ABC
import json
class ConfigManager(ABC):
    def __init__(self, **kwargs):
        pass

    def __getitem__(self, item):
        pass


class JSONConfigManager(ConfigManager):

    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            config_data: dict = json.load(f)

        for key, value in config_data.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)
