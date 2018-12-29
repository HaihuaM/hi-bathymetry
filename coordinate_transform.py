
"""
This is a sub-module to transform the coordinates from lat.tiff and lon.tiff
"""
import numpy
import pickle
from osgeo import gdal
from tqdm import tqdm
from global_envs import *
from utility import time_stamp
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


def locate_point(lat_value, lon_value):
    """
    Function: locate the index from lat value and lon value
    parameters: lat value, lon value
    returns: the index in tiff file
    # COORDINATE EXAMPLE:
    #     Lon: 121.3386
    #     Lat: 31.56370
    """

    _, _, lat_data_loader = coordinate_data_loader(LAT_TIFF_FILE)
    lat_data = lat_data_loader()
    _, _, lon_data_loader = coordinate_data_loader(LON_TIFF_FILE)
    lon_data = lon_data_loader()

    mapped_lat_data = numpy.where(lat_data > lat_value, 1, 0)
    # mapped_lon_data = numpy.where(lon_data > lon_value, 1, 0)
    # new_data = mapped_lat_data + mapped_lon_data
    # plt.figure('')
    # plt.imshow(new_data)
    # plt.show()
    temp_y = numpy.sum(mapped_lat_data, axis=0)
    [temp_x] = numpy.where(temp_y < 2166)
    candidates_list = list(zip(temp_y, temp_x))
    len_candidates = len(candidates_list)
    for i in range(len_candidates -1):
        index_y = candidates_list[i][0]
        index_x = candidates_list[i][1]
        temp_lon_value = lon_data[index_y][index_x]

        next_index_y = candidates_list[i+1][0]
        next_index_x = candidates_list[i+1][1]
        next_temp_lon_value = lon_data[next_index_y][next_index_x]
        if lon_value > temp_lon_value and lon_value < next_temp_lon_value:
            return [index_y, index_x]
    return []


def load_depth_data_parser(depth_file):
    """
    Function: read the depth.txt
    parameters:  depth.txt
    returns: coordinates list
    """

    coordinates = list()
    with open(depth_file, 'r') as depth_file_handle:
        lines = depth_file_handle.readlines()
        for ln in lines:
            ln_splitted = ln.strip('\n').split()
            lon_value = ln_splitted[0]
            lat_value = ln_splitted[1]
            depth_value = ln_splitted[2]
            coordinates.append((lat_value, lon_value))
    return coordinates


def load_depth_data_coordinate(coordinates):
    """
    Function: read the coordinates data 
    parameters: list of the lat and lon data
    returns: list of the index for each point
    """
    index_list = list()
    for coord in tqdm(coordinates):
        # print(coord)
        lat_value = float(coord[0])
        lon_value = float(coord[1])
        index = locate_point(lat_value, lon_value)
        index_list.append(index)
        # print(index)
    return index_list
        

def dump_points_index(index_list):
    """
    Function: dump the index of each point
    parameters: list of the index for each point
    return: path of the file dumped
    """
    time_info  = time_stamp()
    out_path = op.join(OUTPUT_PATH, 'points_index.info.'+time_info)
    with open(out_path, 'wb') as points_index_file_handle:
        pickle.dump(index_list, points_index_file_handle)
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


def test():
    LON_VALUE = 121.1890
    LAT_VALUE = 31.66390

    INDEX = locate_point(LAT_VALUE, LON_VALUE)
    print(INDEX)
    # plt.figure('')
    # plt.imshow(lat_data)
    # plt.show()
    # print(points_index)



if __name__ == '__main__':

    coordinates = load_depth_data_parser(DEPTH_DATA_PATH)
    index_list = load_depth_data_coordinate(coordinates)
    out_path = dump_points_index(index_list)
    # load_points_index(out_path)
    # print(len(index_list))
    
    




