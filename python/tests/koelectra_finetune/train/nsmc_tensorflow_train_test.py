import unittest

from koelectra_finetune.train.nsmc_tensorflow_train import NsmcTensorflowKoelectraSmallModelTrainer


class TestNsmcKoelectraSmallModelTrain(unittest.TestCase):

    def test_train(self):
        test_config_path = "test_nsmc_config.json"
        trainer = NsmcTensorflowKoelectraSmallModelTrainer(test_config_path)
        trainer.train()


if __name__ == '__main__':
    unittest.main()
