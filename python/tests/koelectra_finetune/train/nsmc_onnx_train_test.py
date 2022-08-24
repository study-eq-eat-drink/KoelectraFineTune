import unittest

from koelectra_finetune.model.nsmc_tensorflow_model import NsmcKoelectraSmallTokenizer
from koelectra_finetune.train.nsmc_onnx_train import NsmcKoelectraSmallOnnxModelTrainer

import onnxruntime


class TestNsmcKoelectraSmallOnnxModelTrainer(unittest.TestCase):

    def test_train(self):
        keras_model_path = '../../../../model/nsmc/test/tensorflow'
        onnx_model_path = r'..\..\..\..\model\nsmc\test\onnx\model.onnx'
        onnx_model = NsmcKoelectraSmallOnnxModelTrainer.convert_keras_to_onnx(keras_model_path, onnx_model_path)

        test_text = '한국영화 존나 재밌네'
        model_input = NsmcKoelectraSmallTokenizer.tokenize_model_input(test_text)

        onnx_runtime_session = onnxruntime.InferenceSession(onnx_model)
        onnx_runtime_session.get_inputs()[0]


if __name__ == '__main__':
    unittest.main()
