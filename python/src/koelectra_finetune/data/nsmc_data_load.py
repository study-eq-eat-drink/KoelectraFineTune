from koelectra_finetune.data.data_load import DataLoader
import pandas as pd

class NsmcDataLoader(DataLoader):

    @classmethod
    def load(cls, data_path, **args):
        review_datas = pd.read_csv(data_path)
        print(review_datas.head())
