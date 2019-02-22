
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
from utility import time_stamp, debug_print
# from models import *
from models_v2 import *
from coordinate_transform import gdal_reader, load_depth_data_parser, obj_dump
from sklearn import preprocessing
import matplotlib.pyplot as plt


def model_predict(Model, checkpoint_dir):

    func_name = sys._getframe().f_code.co_name
    out_file_name = '.'.join([func_name, 'data'])

    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    # print(xsize, ysize)
    geo_data = geo_data_loader()
    # debug_print(geo_data[:,0,0])

    features = np.zeros([ysize,xsize,raster_count])
    for i in tqdm(range(raster_count)):
        features[:,:,i] = geo_data[i,:,:]

    del geo_data
    debug_print(features.shape)
    
    features.shape = [ysize*xsize, raster_count]
    max_predict_num = 60000
    predict_cnt = ysize*xsize // max_predict_num + 1

    model = Model(no_data_loader=True)
    model.setup_cnn_model()

    predict_data = list()
    for x in tqdm(range(predict_cnt)):
        if x == predict_cnt -1 :
            model.test_features = features[x*max_predict_num:]
        else:
            model.test_features = features[x*max_predict_num: (x+1)*max_predict_num]
        prediction_value = model.predict(checkpoint_dir)
        debug_print(prediction_value.flat[:])
        predict_data.extend(prediction_value.flat[:])

    predict_data = np.array(predict_data)
    debug_print(predict_data)
    predict_data = predict_data.flatten()
    debug_print(predict_data.shape)
    debug_print(predict_data)
    data = np.reshape(predict_data, [2166, 1983])
    debug_print(data.shape)
    obj_dump(data, file_name=out_file_name)
    

if __name__ == "__main__":
    model_predict(Model2, 'Model2_20190218_205754_retune_3')

