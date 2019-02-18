
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
from utility import time_stamp, debug_print
from coordinate_transform import gdal_reader, load_depth_data_parser, obj_dump
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def load_pickle_obj(pickle_obj_file):
    """
    Function: load the index of each point
    parameters: dumped file
    """
    with open(pickle_obj_file, 'rb') as points_obj_handle:
        pickle_obj = pickle.load(points_obj_handle)
        return pickle_obj

def extract_pixel_value(points_index_file):
    """
    Function: according to the index in the image to extract pixel values from all bands, not use patch only for single point.
    parameters: NONE
    """
    points_index = load_pickle_obj(points_index_file)

    
    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    geo_data = geo_data_loader()

    func_name = sys._getframe().f_code.co_name
    out_file_name = '.'.join([func_name, 'data'])

    data = list()

    for patch in tqdm(points_index):
        patch = numpy.array(patch)
        index_y = patch[:,:,0]
        index_x = patch[:,:,1]
        index_y = index_y.flatten()
        index_x = index_x.flatten()
        _data = list()

        for im in geo_data:
            interpolation_results = bilinear_interpolate(im, index_y, index_x)
            _data.append(interpolation_results)

        data.append(_data)

    obj_dump(data, file_name=out_file_name)

def test_interpolation( points_index_file, depth_list_file):
    points_index = load_pickle_obj(points_index_file)
    depth_list = load_pickle_obj(depth_list_file)
    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    geo_data = geo_data_loader()

    index = 0
    for patch in points_index[0:1]:
        # print(patch)
        depth = depth_list[index]
        debug_print(depth)
        # im = geo_data[0]
        index_y = patch[:,:,0]
        index_x = patch[:,:,1]
        index_y = index_y.flatten()
        index_x = index_x.flatten()
        for im in geo_data:
            interpolation_results = bilinear_interpolate(im, index_y, index_x)
            debug_print(interpolation_results)



        index += 1
        


def bilinear_interpolate(im, y, x):
    """
    Function: Interpolation algorithmn
    Parameters: image data, index_y, index_x
    Returns: list of results after interpolation
    """
    x0 = numpy.floor(x).astype(int)
    x1 = x0 + 1
    y0 = numpy.floor(y).astype(int)
    y1 = y0 + 1

    x0 = numpy.clip(x0, 0, im.shape[1]-1)
    x1 = numpy.clip(x1, 0, im.shape[1]-1)
    y0 = numpy.clip(y0, 0, im.shape[0]-1)
    y1 = numpy.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]


    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T


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
    tag = numpy.random.randint(0,20,20)
    tag[10:12] = 0
    cmap = plt.cm.ocean
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.5,.5,.5,1.0)
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = numpy.linspace(-1000,70000,21)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    geo_data = geo_data_loader()
    band = numpy.copy(geo_data[90])
    # points_index_file = "points_index.info.20181229_063205"
    points_index_file = "points_index.info.20190128_081340"
    points_index = load_points_index(points_index_file)
    _, depths = load_depth_data_parser(DEPTH_DATA_PATH)
    margin = 4
    index = 0
    duplicate = 0
    unique_points_index = list()
    depth_list = list()
    for point in points_index:
        if point:
            if point in unique_points_index:
                pass
                duplicate += 1
                depth = float(depths[index])
                depth_list.append(depth)
            else:
                unique_points_index.append(point)
                index_y = point[0]
                index_x = point[1]
                depth = float(depths[index])
                depth_list.append(depth)
                band[index_y -margin:index_y+margin, index_x -margin : index_x + margin] = depth * 2500
        else:
            print("Cannot locate point:", index)
            pass
        index += 1

    depth_array = numpy.array(depth_list)
    max_depth = numpy.max(depth_array)
    min_depth = numpy.min(depth_array)
    # print(max_depth, min_depth)
    # sys.exit(0)

    print('duplicate', duplicate)
    plt.figure('')
    gci = plt.imshow(band, cmap=cmap, norm=norm)
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
    # mark_labels()
    # points_index_file = "../out/points_index.info.20190128_151218"
    points_index_file = "../out/points_index_patch_x2.info.20190218_053622"
    depth_list_file = "../out/filterd_depth.info.20190217_125210"
    extract_pixel_value(points_index_file)
    # test_interpolation(points_index_file, depth_list_file)

