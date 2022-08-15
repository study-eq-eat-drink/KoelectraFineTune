from koelectra_finetune.train.nsmc_train import NsmcKoelectraSmallModelTrainer


def run_nsmc_finetune():
    config_path = "koelectra_finetune/config/nsmc_config.json"
    trainer = NsmcKoelectraSmallModelTrainer(config_path)
    trainer.train()


run_nsmc_finetune()