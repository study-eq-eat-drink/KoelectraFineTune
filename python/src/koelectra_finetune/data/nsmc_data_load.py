from pandas import DataFrame

from koelectra_finetune.data.data_load import DataLoader
import pandas as pd


class NsmcDataLoader(DataLoader):

    @classmethod
    def load(cls, data_path, **args) -> DataFrame:
        review_datas = pd.read_csv(data_path, sep='\t')
        return review_datas
