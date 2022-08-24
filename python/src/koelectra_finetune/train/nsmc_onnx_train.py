import tensorflow as tf
import tf2onnx


class NsmcKoelectraSmallOnnxModelTrainer:

    @classmethod
    def train(cls, keras_model_path, onnx_model_path):
        cls.convert_keras_to_onnx(keras_model_path, onnx_model_path)

    @classmethod
    def convert_keras_to_onnx(cls, keras_model_path, onnx_model_path):
        keras_model = tf.keras.models.load_model(keras_model_path)
        onnx_model = tf2onnx.convert.from_keras(keras_model, output_path=onnx_model_path)
        return onnx_model
