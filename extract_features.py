
"""
This is a sub-module to utilize the index data to extract pixel value from hyperspectral image.
"""

__author__ = 'haihuam'

import numpy
import pickle
from osgeo import gdal
from tqdm import tqdm
from global_envs import *
from utility import time_stamp
from coordinate_transform import gdal_reader, load_depth_data_parser
import matplotlib.pyplot as plt

global GEO_TIFF_FILE 
global DEPTH_DATA_PATH
global OUTPUT_PATH

def load_points_index(points_index_file):
    """
    Function: load the index of each point
    parameters: dumped file
    """
    with open(points_index_file, 'rb') as points_index_file_handle:
        points_index = pickle.load(points_index_file_handle)
        return points_index


def mark_labels():
    """
    Function: to print labels in the hyperspectral image.
    """
    X_test, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    geo_data = geo_data_loader()
    band = numpy.copy(geo_data[90])
    points_index_file = "points_index.info.20181229_063205"
    points_index = load_points_index(points_index_file)
    _, depths = load_depth_data_parser(DEPTH_DATA_PATH)
    print(len(depths))
    margin = 3
    index = 0
    for point in points_index:
        if point:
            index_y = point[0]
            index_x = point[1]
            depth = float(depths[index])
            band[index_y -margin:index_y+margin, index_x -margin : index_x + margin] = depth * 5000
        index += 1
    plt.figure('')
    plt.imshow(band)
    plt.show()


def test():
    points_index_file = "points_index.info.20181229_063205"
    points_index = load_points_index(points_index_file)
    # print(len(points_index))
    cnt = 0
    for x in points_index:
        if x:
            cnt += 1
    print(cnt)



if __name__ == '__main__':
    # test()
    mark_labels()
