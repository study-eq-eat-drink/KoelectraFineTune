from koelectra_finetune.model.nsmc_model import NsmcKoelectraSmallModel, NsmcKoelectraSmallTokenizer
from koelectra_finetune.data.nsmc_data_load import NsmcDataLoader
from koelectra_finetune.config.config_manager import JSONConfigManager

import numpy as np
import tensorflow as tf

class NsmcKoelectraSmallModelTrainer:

    def __init__(self, config_path: str):
        self.config_path = config_path

    def train(self):
        config_manager = JSONConfigManager(self.config_path)

        # 데이터 가져오기
        train_data_path = config_manager.get("DATA")["train_data_path"]
        train_data = NsmcDataLoader.load(train_data_path)
        train_data = train_data[train_data["document"].notna()]
        train_data = train_data[:100]
        print(train_data["document"].dtype)

        test_data_path = config_manager.get("DATA")["test_data_path"]
        test_data = NsmcDataLoader.load(test_data_path)



        # 데이터 토크 나이징
        texts = train_data["document"].tolist()
        x_train_datas = NsmcKoelectraSmallTokenizer.tokenize_model_input(texts)

        # 데이터 라벨 변환
        labels = train_data["label"]
        y_train_datas = self.__parse_label_to_y_data(labels)

        # 모델 로드
        nsmc_model = NsmcKoelectraSmallModel().get_model()

        # 모델 학습 설정
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)
        loss_func = tf.keras.losses.CategoricalCrossentropy()
        nsmc_model.compile(
            optimizer=optimizer,
            loss=loss_func,
            metrics=['accuracy']
        )

        input_ids = x_train_datas['input_ids']
        attention_mask = x_train_datas['attention_mask']
        token_type_ids = x_train_datas['token_type_ids']
        print(type(input_ids), type(attention_mask), type(token_type_ids), type(y_train_datas))
        print(input_ids.shape, attention_mask.shape, token_type_ids.shape, y_train_datas.shape)
        print(input_ids.dtype, attention_mask.dtype, token_type_ids.dtype, y_train_datas.dtype)
        nsmc_model.summary()
        nsmc_model.fit([input_ids, attention_mask, token_type_ids], y_train_datas, epochs=1, batch_size=64)
        # 모델 학습
        # 모델 테스트
        # 모델 저장

    def __parse_label_to_y_data(self, labels):
        y_datas = np.zeros((len(labels), 2), dtype='int32')
        for index, label in enumerate(labels):
            y_datas[index][label] = 1
        return y_datas


