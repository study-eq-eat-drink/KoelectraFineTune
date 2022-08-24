import unittest
from koelectra_finetune.model.nsmc_tensorflow_model import NsmcKoelectraSmallTokenizer, NsmcKoelectraSmallModel


class TestNsmcKoelectraSmallModel(unittest.TestCase):

    def test_get_model(self):

        model = NsmcKoelectraSmallModel().get_model()

        self.assertTrue(model is not None)


class TestNsmcKoelectraSmallTokenizer(unittest.TestCase):

    def test_tokenize_model_input(self):
        text = "영화 존나 재미 있다 이게 영화지"
        model_inputs = NsmcKoelectraSmallTokenizer.tokenize_model_input(text)
        print(model_inputs.keys())
        print(type(model_inputs['input_ids']))
        print(model_inputs)
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
