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

        data_config = config_manager["DATA"]
        train_config = config_manager["TRAIN"]

        # 데이터 가져오기
        train_data_path = data_config["train_data_path"]
        train_data = NsmcDataLoader.load(train_data_path)
        train_data = train_data[train_data["document"].notna()]

        test_data_path = data_config["test_data_path"]
        test_data = NsmcDataLoader.load(test_data_path)
        train_data = train_data[train_data["document"].notna()]

        row_limit = train_config.get("train_row_limit")
        if row_limit is not None and row_limit > 0:
            train_data = train_data[:row_limit]
            test_data = test_data[:row_limit]


        # 데이터 토크 나이징
        train_texts = train_data["document"].tolist()
        x_train_datas = NsmcKoelectraSmallTokenizer.tokenize_model_input(train_texts)

        test_texts = test_data["document"].tolist()
        x_test_datas = NsmcKoelectraSmallTokenizer.tokenize_model_input(test_texts)
        
        # 데이터 라벨 변환
        train_labels = train_data["label"]
        y_train_datas = self.__parse_label_to_y_data(train_labels)

        test_labels = train_data["label"]
        y_test_datas = self.__parse_label_to_y_data(test_labels)

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

        train_input_ids = x_train_datas['input_ids']
        train_attention_mask = x_train_datas['attention_mask']
        train_token_type_ids = x_train_datas['token_type_ids']
        print(type(train_input_ids), type(train_attention_mask), type(train_token_type_ids), type(y_train_datas))
        print(train_input_ids.shape, train_attention_mask.shape, train_token_type_ids.shape, y_train_datas.shape)
        print(train_input_ids.dtype, train_attention_mask.dtype, train_token_type_ids.dtype, y_train_datas.dtype)
        nsmc_model.summary()
        
        # 모델 학습
        nsmc_model.fit([train_input_ids, train_attention_mask, train_token_type_ids], y_train_datas
                       , epochs=train_config["epochs"], batch_size=train_config["batch_size"])
        
        # 모델 테스트
        test_input_ids = x_test_datas['input_ids']
        test_attention_mask = x_test_datas['attention_mask']
        test_token_type_ids = x_test_datas['token_type_ids']
        test_result = nsmc_model.evaluate([test_input_ids, test_attention_mask, test_token_type_ids], y_test_datas
                                          , batch_size=train_config["batch_size"])
        
        print(test_result)

        # 모델 저장
        nsmc_model.save(
            filepath=train_config["save_model_path"],
            overwrite=True
        )

    def __parse_label_to_y_data(self, labels):
        y_datas = np.zeros((len(labels), 2), dtype='int32')
        for index, label in enumerate(labels):
            y_datas[index][label] = 1
        return y_datas


