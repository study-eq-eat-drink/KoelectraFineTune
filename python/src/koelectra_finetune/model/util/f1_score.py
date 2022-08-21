from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, classification_report


class F1score(Callback):
    def __init__(self, value=0.0, x_val_data=None, y_val_data=None):
        super(F1score, self).__init__()
        self.value = value
        self.x_val_data = x_val_data
        self.y_val_data = y_val_data

    def on_epoch_end(self, epoch, logs={}):

        X_test = self.x_val_data
        y_test = self.y_val_data
        y_predicted = self.model.predict(X_test)

        # 0, 1 encoding
        y_predicted[y_predicted > 0.5] = 1
        y_predicted[y_predicted <= 0.5] = 0
        #         print(y_test)
        #         print(y_predicted)

        score = f1_score(y_test, y_predicted, average='weighted')

        print(' - f1: {:04.2f}'.format(score * 100))
        print(classification_report(y_test, y_predicted))

        # F1-score가 지금까지 중 가장 높은 경우
        if score > self.value:
            print('f1_score improved from %f to %f, saving model to best_model.h5' % (self.value, score))
            self.model.save_weights('best_model_통합.h5')
            self.value = score
        else:
            print('f1_score did not improve from %f' % (self.value))