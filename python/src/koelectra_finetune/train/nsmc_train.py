from koelectra_finetune.model.nsmc_model import NsmcKoelectraSmallModel, NsmcKoelectraSmallTokenizer
from koelectra_finetune.data.nsmc_data_load import NsmcDataLoader
from koelectra_finetune.config.config_manager import JSONConfigManager

import numpy as np

class NsmcKoelectraSmallModelTrainer:

    def __init__(self, config_path: str):
        self.config_path = config_path

    def train(self):
        config_manager = JSONConfigManager(self.config_path)

        # 데이터 가져오기
        train_data_path = config_manager.get("DATA")["train_data_path"]
        train_data = NsmcDataLoader.load(train_data_path)

        test_data_path = config_manager.get("DATA")["test_data_path"]
        test_data = NsmcDataLoader.load(test_data_path)

        # 데이터 토크 나이징
        x_datas = NsmcKoelectraSmallTokenizer.tokenize_model_input(train_data["document"].tolist())

        # 데이터 라벨 변환
        labels = train_data["label"].to_list
        y_data = self.__parse_label_to_y_data(labels)
        print(labels)
        print(y_data)

        # 모델 로드
        # 모델 학습 설정
        # 모델 학습
        # 모델 테스트
        # 모델 저장

    def __parse_label_to_y_data(self, labels):
        y_datas = np.zeros((len(labels), 2))
        for index, label in enumerate(labels):
            y_datas[index][label] = 1
        return y_datas


