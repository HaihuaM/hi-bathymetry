
"""
This is the sub-module to do the prediction.
"""
__author__ = "haihuam"

import sys
import numpy
import pickle
from osgeo import gdal
from tqdm import tqdm
from global_envs import *
from utility import time_stamp
from models import *
from coordinate_transform import gdal_reader, load_depth_data_parser, obj_dump
from sklearn import preprocessing
import matplotlib.pyplot as plt


def data_load_func1(func1_point_data):
    """
    Function: load_func1_data 
    parameters: pixel data generate from func1.
    returns:
    """
    data_file = op.join(OUTPUT_PATH, func1_point_data)
    with open(data_file, 'rb') as data_file_handle:
        data = pickle.load(data_file_handle)

    features = list()
    labels = list()
    for item in data: 
        features.append(item[0])
        labels.append([item[1]])

    x = np.array(features).squeeze()
    y = np.array(labels)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=33)
    return train_x, test_x, train_y, test_y


def model_predict(Model, checkpoint_dir):


    func_name = sys._getframe().f_code.co_name
    out_file_name = '.'.join([func_name, 'data'])

    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    print(xsize, ysize)
    geo_data = geo_data_loader()
    transposed_data = np.reshape(geo_data, [-1, 330])
    # print(transposed_data.shape)

    # func2_point_data = "extract_pixel_value_func2.data.20190107_043546"
    # data_load_func2 = data_load_func1
    # train_x, test_x, train_y, test_y = data_load_func2(func2_point_data)

    # ss_x = preprocessing.StandardScaler()
    # transposed_data = ss_x.fit_transform(transposed_data)

    # print(transposed_data.shape)
    # sys.exit(0)

    # print(test_x.shape)
    # sys.exit(0)
    model = Model(test_x=transposed_data)
    model.setup_net()
    prediction_value = model.test_predict(checkpoint_dir)
    print(prediction_value)
    print(prediction_value.shape)
    data = np.reshape(prediction_value, [ysize, xsize])
    print(data.shape)
    obj_dump(data, file_name=out_file_name)
    

if __name__ == "__main__":
    # model_predict(Model10, 'Model10_func2_20190115_103412')
    model_predict(Model1, 'Model1_func2_20190115_144146')

