
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
import matplotlib.pyplot as plt

def model_predict(Model, checkpoint_dir):


    func_name = sys._getframe().f_code.co_name
    out_file_name = '.'.join([func_name, 'data'])

    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    # print(xsize, ysize)
    geo_data = geo_data_loader()
    transposed_data = np.reshape(geo_data, [-1, 330])

    model = Model(test_x=transposed_data)
    model.setup_net()
    prediction_value = model.predict(checkpoint_dir)
    print(prediction_value.shape)
    data = np.reshape(prediction_value, [ysize, xsize])
    print(data.shape)
    obj_dump(data, file_name=out_file_name)
    

if __name__ == "__main__":
    model_predict(Model10, 'Model10_func2_20190115_065637')

