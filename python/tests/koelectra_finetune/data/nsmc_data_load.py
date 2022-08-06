import unittest
from koelectra_finetune.data.nsmc_data_load import NsmcDataLoader
from pathlib import Path


class TestNsmcDataLoad(unittest.TestCase):

    def test_load(self):

        test_file_path = self.get_project_root() / "data" / "nsmc" / "ratings_test.txt"
        print(self.get_project_root())
        print(test_file_path)
        NsmcDataLoader.load(str(test_file_path))

    def get_project_root(self) -> Path:
        return Path(__file__).parent.parent.parent.parent.parent


if __name__ == 'main':
    unittest.main()
