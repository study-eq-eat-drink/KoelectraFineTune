from transformers import TFElectraModel, ElectraTokenizer
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


class NsmcKoelectraSmallModel:

    def __init__(self, max_length=512):
        model_configs = {
            'max_length': max_length
        }
        self.model_configs = model_configs

    def get_model(self):
        model_configs = self.model_configs

        max_length = model_configs['max_length']
        input_token = Input((max_length,), dtype='int32')
        input_pad_mask = Input((max_length,), dtype='int32')
        input_segment = Input((max_length,), dtype='int32')
        pt_model = TFElectraModel.from_pretrained(
            "monologg/koelectra-small-v3-discriminator", from_pt=True
        )([input_token, input_pad_mask, input_segment])
        output = Dense(2, activation='softmax')(pt_model)

        nsmc_model = Model([input_token, input_pad_mask, input_segment], output)
        return nsmc_model


class NsmcKoelectraSmallTokenizer:

    __tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")

    @classmethod
    def tokenize_token(cls, text: str):
        return cls.__tokenizer.tokenize(text)

    @classmethod
    def tokenize_token_id(cls, text: str):
        tokenizer = cls.__tokenizer
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    @classmethod
    def tokenize_model_input(cls, text, max_length=512):
        model_input = cls.__tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            pad_to_max_length="max_length"
        )
        return model_input
