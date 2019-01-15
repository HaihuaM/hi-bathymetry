
"""
Submodule to do the visualization.
"""

import os
import sys
import pickle
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import matplotlib.pyplot as plt


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

def show(data_file):

    print('Plot %s' %data_file)
    prediction = load_data(data_file)
    # print(prediction.shape)
    # sys.exit(0)
    flatened_img = flaten_img(prediction)

    plt.figure('%s' %data_file)
    plt.imshow(flatened_img, plt.cm.gray)
    # plt.imshow(flatened_img, cmap=plt.cm.BuPu_r)
    plt.imshow(flatened_img, cmap=plt.cm.tab10)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    show('../out/model_predict.data.20190115_160145')

