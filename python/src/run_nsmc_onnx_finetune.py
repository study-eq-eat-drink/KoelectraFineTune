from koelectra_finetune.train.nsmc_onnx_train import NsmcKoelectraSmallOnnxModelTrainer


def run_nsmc_onnx_finetune():
    keras_model_path = "../../model/nsmc/tensorflow/"
    onnx_model_path = r"..\..\model\nsmc\onnx\model.onnx"
    trainer = NsmcKoelectraSmallOnnxModelTrainer
    trainer.train(keras_model_path, onnx_model_path)


run_nsmc_onnx_finetune()
