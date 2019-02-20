
"""
Submodule to do the visualization.
"""

import os
import sys
import pickle
import numpy as np
from global_envs import *
from osgeo import gdal
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from coordinate_transform import gdal_reader, load_depth_data_parser, obj_dump
from extract_features import load_pickle_obj

global GEO_TIFF_FILE 
global DEPTH_DATA_PATH
global OUTPUT_PATH


def check_result():

    labels_file = op.join(OUTPUT_PATH, 'label.info.20190220_084456')
    predict_file = op.join(OUTPUT_PATH, 'predict.info.20190220_084456')
    labels_list = load_pickle_obj(labels_file)
    predict_list = load_pickle_obj(predict_file)
    plt.figure(figsize=(10,5))
    temp_len = 200
    x = range(len(predict_list[0:temp_len]))
    print(labels_list[0:50])
    print(predict_list[0:50])
    # plt.plot(x, predict_list[0:temp_len], color='g')
    # plt.plot(x, labels_list[0:temp_len], color='r')
    # plt.show()


def flaten_img(img, min_margin = -40, max_margin = 70, gap = 10):
    flatened_img = np.where(((img>= min_margin) & (img < max_margin)), img, min_margin)
    cnts = int(( max_margin - min_margin ) / gap)
    for i in range(1, cnts + 1 ):
	
        down_border = min_margin + (i -1)* gap
        up_border = min_margin + i* gap
        middle = (down_border + up_border) / 2
        flatened_img = np.where(((img >= down_border) & (img < up_border)), middle, flatened_img)
    return flatened_img

def load_data(data_file):
    """
    Function: load the index of each point
    parameters: dumped file
    """
    with open(data_file, 'rb') as data_file_handle:
        data = pickle.load(data_file_handle)
        return data

def show(data_file=None):


    tag = np.random.randint(0,20,20)
    tag[10:12] = 0
    cmap = plt.cm.ocean
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.5,.5,.5,1.0)
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(-1000,70000,21)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    
    # # plt.figure('%s' %data_file)
    depth_data = np.load(data_file)
    # plt.figure('0')
    # gci_0 = plt.imshow(depth_data)

    corrected_depth_data = (depth_data ) * 2000
    river_extraction_label = np.load('river_extraction.np.npy')
    corrected_depth_data = np.multiply(river_extraction_label, corrected_depth_data)

    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    geo_data = geo_data_loader()
    band = np.copy(geo_data[90])
    compound_img = band + 4000 + corrected_depth_data
    compound_img = np.where(((compound_img>= -10000) & (compound_img < 70000)), compound_img, 0)

    plt.figure('1')
    # gci = plt.imshow(band)
    gci = plt.imshow(compound_img, cmap=cmap, norm=norm)
    cbar = plt.colorbar(gci) 
    cbar.set_label('$Depth(m)$')  
    cbar.set_ticks(np.linspace(3000, 70000, 8))  
    cbar.set_ticklabels( ('0', '10', '20', '30', '40',  '50',  '60',  '70'))


    print('Plot %s' %data_file)
    prediction = load_data(data_file)
    flatened_img = prediction

    plt.figure('%s' %data_file)
    plt.imshow(flatened_img, plt.cm.gray)
    plt.imshow(flatened_img, cmap=plt.cm.BuPu_r)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # show('depth.img.npy')
    check_result()

