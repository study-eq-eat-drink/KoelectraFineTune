import unittest

import onnx

from koelectra_finetune.model.nsmc_tensorflow_model import NsmcKoelectraSmallTokenizer
from koelectra_finetune.train.nsmc_onnx_train import NsmcKoelectraSmallOnnxModelTrainer

import onnxruntime


class TestNsmcKoelectraSmallOnnxModelInfrencer(unittest.TestCase):

    def test_train(self):
        onnx_model_path = r'..\..\..\..\model\nsmc\test\onnx\model.onnx'
        onnx_model = onnx.load_model(onnx_model_path)
        onnx.checker.check_model(onnx_model)

        onnx_runtime_session = onnxruntime.InferenceSession(onnx_model_path)

        test_text = ['한국영화 존나 재밌네']
        model_input = NsmcKoelectraSmallTokenizer.tokenize_model_input(test_text)

        input_ids = model_input['input_ids']
        attention_mask = model_input['attention_mask']
        token_type_ids = model_input['token_type_ids']

        onnx_model_inputs = onnx_runtime_session.get_inputs()
        onnx_inputs = {
            onnx_model_inputs[0].name: input_ids,
            onnx_model_inputs[1].name: attention_mask,
            onnx_model_inputs[2].name: token_type_ids
        }

        result = onnx_runtime_session.run(None, onnx_inputs)
        print(result)


if __name__ == '__main__':
    unittest.main()
