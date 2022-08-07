import unittest
from koelectra_finetune.model.nsmc_model import NsmcKoelectraSmallTokenizer, NsmcKoelectraSmallModel


class TestNsmcKoelectraSmallModel(unittest.TestCase):

    def test_get_model(self):
        NsmcKoelectraSmallTokenizer.tokenize_token()

        model = NsmcKoelectraSmallModel().get_model()
        self.assertTrue(model is not None)


class TestNsmcKoelectraSmallTokenizer(unittest.TestCase):

    def test_tokenize_model_input(self):
        text = "영화 존나 재미 있다 이게 영화지"
        NsmcKoelectraSmallTokenizer.tokenize_model_input(text)

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
