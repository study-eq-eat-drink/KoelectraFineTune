import unittest

from koelectra_finetune.train.nsmc_onnx_train import NsmcKoelectraSmallOnnxModelTrainer
from koelectra_finetune.train.nsmc_tensorflow_train import NsmcKoelectraSmallModelTrainer

class TestNsmcKoelectraSmallOnnxModelTrainer(unittest.TestCase):

    def test_train(self):
        keras_model_path = '../../../../model/nsmc/test/tensorflow/'
        onnx_model_path = r'..\..\..\..\model\nsmc\test\onnx\model.onnx'
        NsmcKoelectraSmallOnnxModelTrainer.convert_keras_to_onnx(keras_model_path, onnx_model_path)


if __name__ == '__main__':
    unittest.main()
