
"""
This is the sub-module to do the prediction.
"""
__author__ = "haihuam"

import sys
import numpy as np
import pickle
from osgeo import gdal
from tqdm import tqdm
from global_envs import *
from utility import time_stamp, debug_print
from models_v2 import *
from coordinate_transform import gdal_reader, load_depth_data_parser, obj_dump, build_patch
from extract_features import load_pickle_obj
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc


def foo():

    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    geo_data = geo_data_loader()
    conv_kernel_size = 4
    scharr = np.ones((conv_kernel_size, conv_kernel_size))/(conv_kernel_size**2)
    debug_print(scharr)
    conv_data = np.zeros((raster_count, ysize, xsize))
    for band in tqdm(range(raster_count)):
        conv_data[band, :, :] = signal.convolve2d(geo_data[band], scharr, boundary='symm', mode='same')

    debug_print(conv_data.shape)
    out_file_name = '.'.join(['conv', 'data'])
    # obj_dump(conv_data, file_name=out_file_name)
    np.save(out_file_name, conv_data)

def build_gdal_img_patch(patch_size=1):
    """
    Function: pre-process function for gdal data
    Parameters: None
    Return: GDAL data
    """
    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    patch_list = list()
    for i in tqdm(range(ysize)):
        for j in range(xsize):
            patch = build_patch((i, j), (ysize-1, xsize-1), patch_size)
            patch_list.append(patch)
    return patch_list

def dump_patch_img_data():
    """
    Function: dump patch index data from gdal img
    Parameters: patch list
    Return: None
    """
    img_patch_index_file = "img_patch_index.info"
    patch_list = build_gdal_img_patch(patch_size=1)
    out_path = obj_dump(patch_list, img_patch_index_file)
    return


def model_predict(Model, checkpoint_dir):

    func_name = sys._getframe().f_code.co_name
    out_file_name = '.'.join([func_name, 'data'])

    # xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    # geo_data = geo_data_loader()
    geo_data = np.load('conv.data.npy')
    debug_print(geo_data.shape)
    raster_count, ysize, xsize = geo_data.shape

    features = np.zeros([ysize,xsize,raster_count])
    for i in tqdm(range(raster_count)):
        features[:,:,i] = geo_data[i,:,:]

    del geo_data
    debug_print(features.shape)
    sys.exit(0)
    
    
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
    model_predict(Model2, 'Model2_20190219_143221_retune_9')
    # dump_patch_img_data()
    # points_index_file = "../out/img_patch_index.info.20190224_081446"
    # extract_pixel_value(points_index_file)
    # foo()

