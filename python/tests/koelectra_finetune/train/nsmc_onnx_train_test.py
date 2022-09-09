import unittest

import onnx.checker

from koelectra_finetune.model.nsmc_tensorflow_model import NsmcKoelectraSmallTokenizer
from koelectra_finetune.train.nsmc_onnx_train import NsmcKoelectraSmallOnnxModelTrainer

import onnxruntime


class TestNsmcKoelectraSmallOnnxModelTrainer(unittest.TestCase):

    def test_train(self):
        keras_model_path = '../../../../model/nsmc/test/tensorflow/nsmc_model.h5'
        onnx_model_path = r'..\..\..\..\model\nsmc\test\onnx\model.onnx'
        onnx_model = NsmcKoelectraSmallOnnxModelTrainer.convert_keras_to_onnx(keras_model_path, onnx_model_path)
        onnx.checker.check_model(onnx_model)


if __name__ == '__main__':
    unittest.main()
