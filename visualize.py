
"""
Submodule to do the visualization.
"""

import os
import sys
import pickle
import numpy as np
from global_envs import *
from utility import debug_print
from osgeo import gdal
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from coordinate_transform import gdal_reader, load_depth_data_parser, obj_dump
from extract_features import load_pickle_obj
from skimage import data, color
from skimage.morphology import disk
import skimage.filters.rank as sfr
import skimage.morphology as sm
import skimage.filters.rank as sfr

global GEO_TIFF_FILE 
global DEPTH_DATA_PATH
global OUTPUT_PATH


def check_result():

    labels_file = op.join(OUTPUT_PATH, 'label.info.20190225_115340')
    predict_file = op.join(OUTPUT_PATH, 'predict.info.20190225_115340')
    labels_list = load_pickle_obj(labels_file)
    predict_list = load_pickle_obj(predict_file)

    depth_file_path = "../orig_data/data.depth/depth_serial"
    depth_file_list = glob(op.join(depth_file_path, "*.txt"))
    file_cnts = len(depth_file_list)
    len_list = list()
    for i in range(1, file_cnts+1):
        # debug_print(i)
        _depth_file = op.join(depth_file_path, str(i)+".txt")
        file_handle = open(_depth_file)
        sample_cnt = len(file_handle.readlines())
        len_list.append(sample_cnt)

    prev_len = 0
    index = 0
    for item in tqdm(len_list):
        index += 1
        pic_name = "fracture_surface_" + str(index) + '.png'
        pic_path = op.join('../out/predict/pictures/', pic_name)
        plt.figure(pic_name)

        predict = predict_list[prev_len:prev_len+item]
        labels = labels_list[prev_len:prev_len+item]
        predict = runningMeanFast(predict, 10)
        labels = runningMeanFast(labels, 10)
        # errors = list(map(lambda x: x[0] - x[1], zip(predict, labels)))
        errors = predict - labels
        errors = runningMeanFast(errors, 10)
        x = range(len(predict))
        
        fig, ax = plt.subplots()
        plt.ylabel('Height (m)')
        plt.grid(True)
        plt.title('The %sth fracture surface'%index)

        ax.plot(x, predict,  label='Prediction Value', color='g')
        ax.plot(x, labels,   label='Ground Truth', color='r')
        ax.plot(x, errors,   label='Errors', color='y')
        # legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
        legend = ax.legend(loc='upper right', shadow=True)
        legend.get_frame().set_facecolor('C2')

        prev_len += item
        plt.savefig(pic_path)
        plt.close()
        # plt.show()

def runningMeanFast(x, N):
    # return np.convolve(x, np.ones((N,))/(N*2))[(N-1):]
    return np.convolve(x, np.ones((N,))/N)[(N-1):]
    

def flaten_img(img, min_margin = -40, max_margin = 70, gap = 20):
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

def dialation(img, scope=3):
    return sm.dilation(img, sm.square(scope)) 

def erosion(img, scope=3):
    return sm.erosion(img, sm.square(scope)) 

def opening(img, scope=3):
    return sm.opening(img, sm.disk(scope))

def mean_filter(img, scope=3):
    # mean_kernal = tf.constant(np.ones(3, 3, 1, 1))
    # return tf.nn.conv2d(img, mean_kernal, strides=[1, 1, 1, 1], padding='SAME')
    return sfr.mean(img, disk(scope)) 


def _show(data_file=None):

    prediction = np.load(data_file)
    flatened_img = flaten_img(prediction, 0, 60, 20)


    river_extraction_label = np.load('river_extraction.np.npy')
    # result = np.multiply(river_extraction_label, img)
    n_max = np.max(flatened_img)
    n_min = np.min(flatened_img)
    normal_img = (flatened_img - n_min)/(n_max - n_min)
    filtered_img = mean_filter(normal_img, 6)
    img = (filtered_img / 256)*(n_max - n_min) + n_min
    result = np.multiply(river_extraction_label, img)

    Z = result
    from matplotlib import cm
    # norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
    # cmap = cm.gist_ncar
    # cmap = cm.hsv
    # cmap = cm.gist_rainbow
    # cmap = cm.gnuplot
    cmap = cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.5,.5,.5,1.0)
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(0,60,25)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    # axs = _axs.flatten()

    levels = range(0, 60, 2)
    cset1 = axs.contourf(Z, levels, origin='upper', norm=norm, cmap=cm.get_cmap(cmap, len(levels) - 1))
    fig.colorbar(cset1, ax=axs)
    plt.show()







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
    depth_data = flaten_img(np.load(data_file))

    corrected_depth_data = (depth_data ) * 2000
    river_extraction_label = np.load('river_extraction.np.npy')
    corrected_depth_data = np.multiply(river_extraction_label, corrected_depth_data)

    xsize, ysize, raster_count, geo_data_loader = gdal_reader(GEO_TIFF_FILE)
    geo_data = geo_data_loader()
    band = np.copy(geo_data[90])
    compound_img = band + 4000 + corrected_depth_data
    compound_img = np.where(((compound_img>= -10000) & (compound_img < 70000)), compound_img, 0)

    gci = plt.imshow(compound_img, cmap=cmap, norm=norm)
    cbar = plt.colorbar(gci) 
    cbar.set_label('$Depth(m)$')  
    cbar.set_ticks(np.linspace(3000, 70000, 8))  
    cbar.set_ticklabels( ('0', '10', '20', '30', '40',  '50',  '60',  '70'))


    print('Plot %s' %data_file)
    prediction = load_data(data_file)
    flatened_img = prediction

    plt.figure('%s' %data_file)
    # plt.imshow(flatened_img, plt.cm.gray)
    plt.imshow(flatened_img, cmap=plt.cm.BuPu_r)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # show('model_predict.data.20190220_210542')
    # _show('model_predict.data.20190220_210542')
    # _show('model_predict.data.20190222_082124')
    check_result()

