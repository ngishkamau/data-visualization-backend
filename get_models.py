import keras
import numpy as np

from models_1_and_2.main.gan_util import test, train_gan
from models_1_and_2.main.knn_util import knn_test, knn_train
from models_1_and_2.main.process_data import get_feature_mean_std
from models_1_and_2.main.util import *


def right_file_format(filename: str, format:str) -> bool:
    if filename.endswith(format):
        return True
    return False

def get_model_1(filename: str) -> dict:
    if not right_file_format(filename, '.csv'):
        return {'error': 'Wrong file format. .csv file required'}
    with open(filename) as f:
        feature_data = get_feature_mean_std(f)
    model = test(feature_data)
    model = model.to_dict('list')
    return model

def get_model_2(filename: str) -> dict:
    if not right_file_format(filename, '.csv'):
        return {'error': 'Wrong file format. .csv file required'}
    with open(filename) as f:
        feature_data = get_feature_mean_std(f)
    model, round_name_k = knn_test(feature_data)
    model = model.T.to_dict()
    return model


def get_model_3(filename: str) -> dict:
    # if not right_file_format(filename, '.h5'):
        # return {'error': 'Wrong file format. .h5 file required'}
    values = np.load('model_3/data/trainX.npy')
    my_model = keras.models.load_model(filename)
    output = my_model.predict(values)
    output = [x[0] for x in output.tolist()]
    return {'output': output}
