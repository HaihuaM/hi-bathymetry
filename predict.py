
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
from models_v2 import *
from coordinate_transform import gdal_reader, load_depth_data_parser, obj_dump, build_patch
from extract_features import load_pickle_obj
from sklearn import preprocessing
import matplotlib.pyplot as plt


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

def extract_pixel_value(points_index_file):
    """
    Function: according to the index in the image to extract pixel values from all bands, not use patch only for single point.
    parameters: NONE
    """
    points_index = load_pickle_obj(points_index_file)
    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    geo_data = geo_data_loader()

    func_name = sys._getframe().f_code.co_name
    out_file_name = '.'.join(['img', 'data'])

    data = list()

    points_index = numpy.array(points_index)
    debug_print(points_index.shape)
    index_y = points_index[:,:,:,0]
    index_x = points_index[:,:,:,1]
    debug_print(index_y.shape)
    debug_print(index_x.shape)
    temp = geo_data[:, index_y, index_x]
    debug_print(temp.shape)
    sys.exit(0)
    # for patch in tqdm(points_index[0:1]):
    for patch in tqdm(points_index):

        debug_print(patch.shape)
        patch = numpy.array(patch)
        index_y = patch[:,:,0]
        index_x = patch[:,:,1]
        index_y = index_y.flatten()
        index_x = index_x.flatten()
        # _data = list()
        temp = geo_data[:, index_y, index_x]
        debug_print(temp.shape)
        sys.exit(0)

        # for im in geo_data:
        #     band_point_data = im[index_y, index_x]
        #     _data.append(band_point_data)

        # data.append(geo_data[:,index_y, index_x])

    # data = numpy.array(data)
    # debug_print(data.shape)

    # obj_dump(data, file_name=out_file_name)


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
    # model_predict(Model2, 'Model2_20190219_143221_retune_9')
    # dump_patch_img_data()
    points_index_file = "../out/img_patch_index.info.20190224_081446"
    extract_pixel_value(points_index_file)

