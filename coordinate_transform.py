
"""
This is a sub-module to transform the coordinates from lat.tiff and lon.tiff
"""

__author__ = 'haihuam'

import numpy
import pickle
from osgeo import gdal
from tqdm import tqdm
from global_envs import *
from utility import time_stamp, debug_print
import matplotlib.pyplot as plt

global GEO_TIFF_FILE 
global LAT_TIFF_FILE
global LON_TIFF_FILE
global DEPTH_DATA_PATH
global OUTPUT_PATH


def gdal_reader(tiff_file):
    """
    Function: common geotiff reader
    parameters: tiff file
    returns: xsize, ysize, raster_count, data_loader function
    """
    dataset = gdal.Open(tiff_file, gdal.GA_ReadOnly)
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    raster_count = dataset.RasterCount
    return xsize, ysize, raster_count, dataset.GetVirtualMemArray


def coordinate_data_loader(tiff_file):
    """
    Function: coordinate file reader
    parameters: tiff file
    returns: xsize, ysize, data_loader function
    """
    xsize, ysize, _, data_loader = gdal_reader(tiff_file)
    return xsize, ysize, data_loader


def locate_point(lat_value, lon_value, lat_data, lon_data, lat_xsize, lat_ysize):
    """
    Function: locate the index from lat value and lon value
    parameters: lat value, lon value, lat_data, lon_data, lat_xsize, lat_ysize
    returns: the index in tiff file
    # COORDINATE EXAMPLE:
    #     Lon: 121.3386
    #     Lat: 31.56370
    """
    mapped_lat_data = numpy.where(lat_data >= lat_value, 1, 0)
    temp_y = numpy.sum(mapped_lat_data, axis=0) - 1 
    # [temp_x] = numpy.where(temp_y < lat_ysize)
    temp_x = numpy.array(range(len(temp_y)))
    candidates_list = list(zip(temp_y, temp_x))
    len_candidates = len(candidates_list)
    y_max_len, x_max_len = lat_data.shape

    for i in range(len_candidates -1):
        index_y = candidates_list[i][0]
        index_x = candidates_list[i][1]
        temp_lon_value = lon_data[index_y][index_x]

        next_index_y = candidates_list[i+1][0]
        next_index_x = candidates_list[i+1][1]
        
        next_temp_lon_value = lon_data[next_index_y][next_index_x]
        if lon_value > temp_lon_value and lon_value < next_temp_lon_value:
            if (index_y + 1 < y_max_len ) and (index_x + 1 < x_max_len):
                _temp_lat_value = lat_data[index_y, index_x]
                _temp_next_lat_value = lat_data[index_y+1, index_x]
                _temp_lon_value = lon_data[index_y, index_x]
                _temp_next_lon_value = lon_data[index_y, index_x+1]

                index_y += (lat_value - _temp_lat_value)/(_temp_next_lat_value - _temp_lat_value)
                index_x += (lon_value - _temp_lon_value)/(_temp_next_lon_value - _temp_lon_value)
                return [index_y, index_x]
            else:
                return []
    return []


def load_depth_data_parser(depth_file):
    """
    Function: read the depth.txt
    parameters:  depth.txt
    returns: coordinates list
    """

    coordinates = list()
    depths = list()
    with open(depth_file, 'r') as depth_file_handle:
        lines = depth_file_handle.readlines()
        for ln in lines:
            ln_splitted = ln.strip('\n').split()
            lon_value = ln_splitted[0]
            lat_value = ln_splitted[1]
            depth_value = ln_splitted[2]
            depths.append(depth_value)
            coordinates.append((lat_value, lon_value))
    return coordinates, depths


def pixel_interpolation(coordinates):
    """
    Function: non-interger points, use bilinear interpolation
    parameters: coordinates
    return: pixel values
    """


def load_depth_data_coordinate(coordinates, depths, patch_size=0):
    """
    Function: read the coordinates data 
    parameters: list of the lat and lon data
    returns: list of the index for each point
    """
    lat_xsize, lat_ysize, lat_data_loader = coordinate_data_loader(LAT_TIFF_FILE)
    lat_data = lat_data_loader()
    lon_xsize, lon_ysize, lon_data_loader = coordinate_data_loader(LON_TIFF_FILE)
    lon_data = lon_data_loader()
    assert(lat_data.shape==lon_data.shape)
    lat_max = numpy.max(lat_data)
    lat_min = numpy.min(lat_data)
    lon_max = numpy.max(lon_data)
    lon_min = numpy.min(lon_data)

    x_patch_res = (lon_max - lon_min) / (lat_xsize - 1)
    y_patch_res = (lat_max - lat_min) / (lat_ysize - 1)
    assert(patch_size >=0 )

    index_list = list()
    depth_list = list()
    depth_index = 0
    # for coord in tqdm(coordinates[32700:]):
    for coord in tqdm(coordinates):
        depth = depths[depth_index]
        lat_value = float(coord[0])
        lon_value = float(coord[1])
        condition = ((lat_value <= lat_max and lat_value >= lat_min) and (lon_value <= lon_max and lon_value >= lon_min))
        if condition:
            index = locate_point(lat_value, lon_value, lat_data, lon_data, lat_xsize, lat_ysize)
            if index:
                patch = build_patch(index, patch_size=patch_size)
                index_list.append(patch)
                depth_list.append(depth)
            else:
                print("Waring: Cannot locate point:", coord)
        else:
            print("Info: Point <lat %s, lon %s> is out of scope" %(lat_value, lon_value))
        depth_index += 1

    return index_list, depth_list
        
def build_patch(index,max_index, patch_size=0):
    """
    Function: Build path index
    parameters: one coord, patch size
    return: tupple of coords list
    """
    index_y =  index[0]
    index_x =  index[1]
    max_y_index = max_index[0]
    max_x_index = max_index[1]

    patch_x_index_list = numpy.arange(-patch_size, patch_size+1, 1) + index_x
    patch_y_index_list = numpy.arange(-patch_size, patch_size+1, 1) + index_y

    corrected_patch_x_list = list()
    corrected_patch_y_list = list()

    for x in patch_x_index_list:
        if x < 0:
            corrected_patch_x_list.append(-x)
        elif x > max_x_index:
            corrected_patch_x_list.append(2*max_x_index-x)
        else:
            corrected_patch_x_list.append(x)

    for y in patch_y_index_list:
        if y < 0:
            corrected_patch_y_list.append(-y)
        elif y > max_y_index:
            corrected_patch_y_list.append(2*max_y_index-y)
        else:
            corrected_patch_y_list.append(y)

    patch_x_list = numpy.array(corrected_patch_x_list)
    patch_y_list = numpy.array(corrected_patch_y_list)

    return numpy.dstack(numpy.meshgrid(patch_y_list, patch_x_list))

def test_build_patch():
    index = [123.63463623, 326.11111]
    x = build_patch(index, patch_size=1)
    print(x)

def obj_dump(dump_obj, file_name='out.dump'):
    """
    Function: dump the index of each point
    parameters: list of the index for each point
    return: path of the file dumped
    """
    time_info  = time_stamp()
    out_path = op.join(OUTPUT_PATH, '.'.join([file_name, time_info]))
    with open(out_path, 'wb') as points_index_file_handle:
        pickle.dump(dump_obj, points_index_file_handle)
        if op.exists(out_path):
            print('Dump points index information successfully.')
            return out_path
        else:
            print('Error: Dump points index information failed.')
            return str()

def load_points_index(points_index_file):
    """
    Function: load the index of each point
    parameters: dumped file
    """
    with open(points_index_file, 'rb') as points_index_file_handle:
        points_index = pickle.load(points_index_file_handle)
        return points_index


def test():
    # lon_value = 121.3450
    # lat_value = 31.61876

    lon_value = 121.353867
    lat_value = 31.566635
    # 121.353867   31.566635
    # 31.61876', '121.3450

    # ('31.60980', '121.3506')
    lat_xsize, lat_ysize, lat_data_loader = coordinate_data_loader(LAT_TIFF_FILE)
    lat_data = lat_data_loader()
    lon_xsize, lon_ysize, lon_data_loader = coordinate_data_loader(LON_TIFF_FILE)
    lon_data = lon_data_loader()
    assert(lat_data.shape==lon_data.shape)
    lat_max = numpy.max(lat_data)
    lat_min = numpy.min(lat_data)
    lon_max = numpy.max(lon_data)
    lon_min = numpy.min(lon_data)

    print("lat_max: %s, lat_min: %s, lon_max: %s, lon_min: %s"%(lat_max, lat_min, lon_max, lon_min))
    print(lat_value, lon_value)
    condition = ((lat_value <= lat_max and lat_value >= lat_min) and (lon_value <= lon_max and lon_value >= lon_min))
    if condition:
        INDEX = locate_point(lat_value, lon_value, lat_data, lon_data, lat_xsize, lat_ysize)
        print(INDEX)
    else:
        print("Info: Point <lat %s, lon %s> is out of scope" %(lat_value, lon_value))


if __name__ == '__main__':

    coordinates, depths = load_depth_data_parser(DEPTH_DATA_PATH)
    patch_size = 2
    index_list, depth_list = load_depth_data_coordinate(coordinates, depths, patch_size)
    # print(index_list)
    # print(len(index_list))
    points_file_name = "points_index_patch_x%s.info" %patch_size
    out_path = obj_dump(index_list, points_file_name)
    # depth_file_name = 'filterd_depth.info'
    # out_path = obj_dump(depth_list, depth_file_name)
    # out_path = '../out/points_index_patch_x1.info.20190217_080559'
    # points_index = load_points_index(out_path)
    # debug_print(len(points_index))

