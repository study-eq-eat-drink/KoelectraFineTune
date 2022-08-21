import keras2onnx
import tensorflow as tf

class NsmcKoelectraSmallOnnxModelTrainer:

    @classmethod
    def convert_keras_to_onnx(cls, keras_model_path, onnx_model_path):
        keras_model = tf.keras.models.load_model(keras_model_path)
        onnx_model = keras2onnx.convert_keras(keras_model)
        keras2onnx.save_model(onnx_model, onnx_model_path)

