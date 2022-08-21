from koelectra_finetune.train.nsmc_tensorflow_train import NsmcTensorflowKoelectraSmallModelTrainer


def run_nsmc_tensorflow_finetune():
    config_path = "koelectra_finetune/config/nsmc_config.json"
    trainer = NsmcTensorflowKoelectraSmallModelTrainer(config_path)
    trainer.train()


run_nsmc_tensorflow_finetune()