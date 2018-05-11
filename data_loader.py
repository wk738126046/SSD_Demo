# --coding: utf-8 --

from mxnet import image
import numpy as np
import mxnet as mx
from utils import show_images

data_shape = (3,512,512)
batch_size = 8
std = np.array([61.04467501, 60.03631381, 60.7750983 ])
rgb_mean = np.array([ 130.063048,  129.967301,  124.410760])
# rgb_mean = np.array([130.063048, 129.967301, 124.410760])
ctx = mx.gpu(0)
resize = (512,512)

def get_iterators(rec_prefix, data_shape, batch_size):
    class_names = ['papercup']
    num_class = len(class_names)
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=data_shape,
        path_imgrec=rec_prefix + '_train.rec',
        path_imgidx=rec_prefix + '_train.idx',
        aug_list=None,
        shuffle=True,
        mean=True,
        std=True,
        rand_crop=1,
        rand_gray=0.2,
        rand_mirror=True,
        rand_pad=0.4,
        pad_val=(rgb_mean[0], rgb_mean[1], rgb_mean[2]),
        min_object_covered=0.95,
        max_attempts=200,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
        aspect_ratio_range=(0.5, 2),
        pca_noise=0.01,
    )

    valid_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=data_shape,
        path_imgrec=rec_prefix + '_val.rec',
        shuffle=False,
        # mean=True,
        # std=True
    )

    return train_iter, valid_iter, class_names, num_class


if __name__ == '__main__':

    rec_prefix = "./dataset/data/rec/img_" + str(resize[0]) + "_" + str(resize[1])
    train_data, valid_data, class_names, num_class = get_iterators(rec_prefix, data_shape, batch_size)

    train_data.reset()
    batch = train_data.next()
    images = batch.data[0][:]
    labels = batch.label[0][:]

    show_images(images.asnumpy(), labels.asnumpy(), rgb_mean, std, show_text=True, fontsize=6, MN=(2, 4))
    #show_9_images(images.asnumpy(), labels, rgb_mean)
    print(labels.shape)