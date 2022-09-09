import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model


class KerasModelConverter:

    def __init__(self):
        pass

    @classmethod
    def convert_keras_h5_to_pb(cls, h5_model_path: str, pb_model_path: str = None):
        # 이 코드는 반드시 keras model 로드 전에 실행되야함.
        tf.keras.backend.set_learning_phase(0)

        keras_model = load_model(h5_model_path)

        with tf.keras.backend.get_session() as tf_session:
            model_graph = tf_session.graph
            model_output_names = [output.op.name for output in keras_model.outputs]
            frozen_graph = cls.frozen_graph(model_graph, tf_session, model_output_names)

        if pb_model_path:
            graph_io.write_graph(frozen_graph, pb_model_path, 'nsmc_model.pb', as_text=False)

        return frozen_graph

    @classmethod
    def frozen_graph(cls, tf_graph, tf_session, output_name):
        with tf_graph.as_default():
            graphdef_inf = tf.graph_util.remove_training_nodes(tf_graph.as_graph_def())
            graphdef_frozen = tf.graph_util.convert_variables_to_constants(tf_session, graphdef_inf, output_name)
            return graphdef_frozen
