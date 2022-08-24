from koelectra_finetune.train.nsmc_tensorflow_train import NsmcKoelectraSmallModelTensorflowTrainer


def run_nsmc_tensorflow_finetune():
    config_path = "koelectra_finetune/config/nsmc_config.json"
    trainer = NsmcKoelectraSmallModelTensorflowTrainer(config_path)
    trainer.train()


run_nsmc_tensorflow_finetune()
