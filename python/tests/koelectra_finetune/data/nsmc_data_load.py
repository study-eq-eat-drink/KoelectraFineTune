import unittest
from koelectra_finetune.data.nsmc_data_load import NsmcDataLoader


class TestNsmcDataLoad(unittest.TestCase):

    def test_load(self):
        data_path = "../../../../data/nsmc/ratings_train.txt"
        data = NsmcDataLoader.load(data_path)
        print(data.head())
        self.assertTrue(len(data) > 0)


if __name__ == 'main':
    unittest.main()
