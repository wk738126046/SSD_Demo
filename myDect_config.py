# -- coding: utf-8 --

import numpy as np
import mxnet as mx

class_names = ['papercup']
num_class = len(class_names)
data_shape = (3,512,512)
batch_size = 1
std = np.array([61.04467501, 60.03631381, 60.7750983 ])
rgb_mean = np.array([ 130.063048,  129.967301,  124.410760])
#anchors (reference data_analysis file)
# sizes_list = [[0.17720574, 0.23724939], [0.30426919, 0.40458742], [.37, .619],[.71, .79], [.88, .961]]
sizes_list = [[0.09556162,0.13453832],[0.17720574, 0.23724939], [0.30426919, 0.40458742], [.37, .619],[.71, .79]]# 0.09556162
ratios_list = [[1,2,.5]]*len(sizes_list)
ctx = mx.gpu(0)
resize = (512,512)
im2rec_path = '/home/wk/anaconda3/lib/python3.6/site-packages/mxnet/tools/im2rec.py'

