
"""
This is a sub-module to utilize the index data to extract pixel value from hyperspectral image.
"""

__author__ = 'haihuam'

import sys
import numpy
import time
import pickle
from osgeo import gdal
from tqdm import tqdm
from global_envs import *
from utility import time_stamp
from coordinate_transform import gdal_reader, load_depth_data_parser, obj_dump
from sklearn.model_selection import train_test_split
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


def extract_pixel_value_func1(points_index_file):
    """
    Function: according to the index in the image to extract pixel values from all bands, not use patch only for single point.
    parameters: NONE
    """
    points_index = load_points_index(points_index_file)
    _, depths = load_depth_data_parser(DEPTH_DATA_PATH)
    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    geo_data = geo_data_loader()

    func_name = sys._getframe().f_code.co_name
    out_file_name = '.'.join([func_name, 'data'])

    data = list()
    margin = 1
    
    index = 0

    for point in points_index:
        if point:
                index_y = point[0]
                index_x = point[1]
                point_pixel_data = geo_data[:, index_y:index_y+margin, index_x: index_x + margin]
                point_depth_data = float(depths[index])
                data.append([point_pixel_data, point_depth_data] )
        else:
            print("Cannot locate point:", index)

        index += 1

    obj_dump(data, file_name=out_file_name)

def extract_pixel_value_func2(points_index_file):
    """
    Function: according to the index in the image to extract pixel values from all bands, not use patch only for single point.
        - compare with func1, it will keep unique [feature: label] pairs
    parameters: NONE
    """
    points_index = load_points_index(points_index_file)
    _, depths = load_depth_data_parser(DEPTH_DATA_PATH)
    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    geo_data = geo_data_loader()
    func_name = sys._getframe().f_code.co_name
    out_file_name = '.'.join([func_name, 'data'])

    data = list()
    unique_points_index = list()
    margin = 1
    
    index = 0

    for point in points_index:
        if point:
            if point in unique_points_index:
                pass
            else:
                unique_points_index.append(point)
                index_y = point[0]
                index_x = point[1]
                point_pixel_data = geo_data[:, index_y:index_y+margin, index_x: index_x + margin]
                point_depth_data = float(depths[index])
                # print(point_depth_data)
                data.append([point_pixel_data, point_depth_data] )
                print([point_pixel_data, point_depth_data])
                time.sleep(1)
        else:
            print("Cannot locate point:", index)

        index += 1

    obj_dump(data, file_name=out_file_name)


def mark_labels():
    """
    Function: to print labels in the hyperspectral image.
    """
    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    geo_data = geo_data_loader()
    band = numpy.copy(geo_data[90])
    # points_index_file = "points_index.info.20181229_063205"
    points_index_file = "points_index.info.20181229_080434"
    points_index = load_points_index(points_index_file)
    _, depths = load_depth_data_parser(DEPTH_DATA_PATH)
    margin = 4
    index = 0
    duplicate = 0
    unique_points_index = list()
    for point in points_index:
        if point:
            if point in unique_points_index:
                pass
                duplicate += 1
            else:
                unique_points_index.append(point)
                index_y = point[0]
                index_x = point[1]
                depth = float(depths[index])
                band[index_y -margin:index_y+margin, index_x -margin : index_x + margin] = depth * 5000
        else:
            # print("Cannot locate point:", index)
            pass
        index += 1
    
    print('duplicate', duplicate)
    plt.figure('')
    gci = plt.imshow(band)
    cbar = plt.colorbar(gci) 
    cbar.set_label('$Depth(m)$')  
    cbar.set_ticks(numpy.linspace(3000, 70000, 8))  
    cbar.set_ticklabels( ('0', '10', '20', '30', '40',  '50',  '60',  '70'))
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
    # points_index_file = "points_index.info.20181229_080434"
    # extract_pixel_value_func2(points_index_file)
