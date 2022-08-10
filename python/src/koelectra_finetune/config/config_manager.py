from abc import ABC
import json
class ConfigManager(ABC):
    def __init__(self, **kwargs):
        pass

    def get(self, key):
        pass


class JSONConfigManager(ConfigManager):

    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            config_data: dict = json.load(f)

        for key, value in config_data.items():
            setattr(self, key, value)

    def get(self, key):
        return getattr(self, key)
