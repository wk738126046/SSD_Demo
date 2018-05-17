# --coding: utf-8 --
'''
model define
'''

from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon import model_zoo
from mxnet.ndarray.contrib import MultiBoxPrior
from mxnet import ndarray as nd
import mxnet as mx
import myDect_config

def get_alexnet_conv(ctx):
    alexnet = model_zoo.vision.alexnet(pretrained=True,ctx=ctx)
    net = nn.HybridSequential()
    net.add(*(alexnet.features[:8]))
    return net

def get_vgg11bn_conv(ctx):
    vgg11net = model_zoo.vision.vgg11_bn(pretrained=True,ctx=ctx)
    net = nn.HybridSequential()
    net.add(*(vgg11net.features[:28]))
    return net

def get_mobilenet_1_conv(ctx):
    mobilenet = model_zoo.vision.mobilenet1_0(pretrained=True,ctx=ctx)
    net = nn.HybridSequential()
    net.add(*(mobilenet.features[:33]))
    # net.add(*(mobilenet.features[:72]))#72/81
    # net.initialize(ctx=ctx)
    return net

def get_resnet18_conv(ctx):
    # if pretrained is false ,you must load param manually
    resnet18net = model_zoo.vision.resnet18_v1(pretrained = True,ctx = ctx,prefix='ssd_')
    # resnet18net.load_params('/home/wk/.mxnet/models/resnet18_v1-38d6d423.params')
    net = nn.HybridSequential()
    # net.initialize()
    net.add(*(resnet18net.features[:8]))
    return net

# maxpool down sample
def down_sample(num_filter):
    out = nn.HybridSequential()
    # extract features which used to output as SSD
    for _ in range(2):
        out.add(nn.Conv2D(num_filter,kernel_size=3,strides=1,padding=1))
        out.add(nn.BatchNorm())
        out.add(nn.Activation('relu'))
    out.add(nn.MaxPool2D(pool_size=2))
    return out

#classify: anchors*(num_classes+1)
def class_predictor(num_class,num_anchors):
    return nn.Conv2D(num_anchors*(num_class+1),kernel_size=3,strides=1,padding=1)

#regression: anchors*4
def box_predictor(num_anchors):
    return nn.Conv2D(num_anchors*4,kernel_size=3,strides=1,padding=1)

#anchor box ()
# sizes_list = [[0.17720574, 0.23724939], [0.30426919, 0.40458742], [.37, .619],
#               [.71, .79], [.88, .961]]
sizes_list = myDect_config.sizes_list

# ratios_list = [[1,2,.5]]*len(sizes_list)
ratios_list = myDect_config.ratios_list*len(sizes_list)

class SSD(nn.HybridBlock):
    def __init__(self,num_class,sizes_list=sizes_list,ratios_list=ratios_list,ctx=mx.gpu(0),
                 verbose=False,**kwargs):
        super(SSD,self).__init__(**kwargs)
        self.num_class = num_class
        self.sizes_list = sizes_list
        self.ratios_list = ratios_list
        #anchors' numbers only adapt to
        self.num_anchors = num_anchors = (len(sizes_list[0])+len(ratios_list[0])-1)
        self.verbose = verbose
        # net = vgg11bn + down_sample(*3) + classify/regression(*5)
        with self.name_scope():
            #part1
            # self.body = get_vgg11bn_conv(ctx)
            # self.body = get_mobilenet_1_conv(ctx)
            self.body = get_resnet18_conv(ctx)
            #part2
            self.down_sample = nn.HybridSequential()
            for _ in range(len(sizes_list)-2):
                self.down_sample.add(down_sample(128))
            #part3
            self.class_predictors = nn.HybridSequential()
            self.box_predictors = nn.HybridSequential()
            for _ in range(len(sizes_list)):
                self.class_predictors.add(class_predictor(num_class,num_anchors))
                self.box_predictors.add(box_predictor(num_anchors))

            self.down_sample.initialize(ctx=ctx)
            self.class_predictors.initialize(ctx=ctx)
            self.box_predictors.initialize(ctx=ctx)

    def hybrid_forward(self, F, x):
        x = self.body(x)
        cls_preds =[]
        box_preds =[]
        anchors = []
        for i in range(len(self.sizes_list)):
            cls_preds.append((self.class_predictors[i](x)).transpose((0, 2, 3, 1)).flatten())
            box_preds.append((self.box_predictors[i](x)).transpose((0,2,3,1)).flatten())
            anchors.append(MultiBoxPrior(x,sizes=self.sizes_list[i],ratios=self.ratios_list[i]))

            if self.verbose:
                print('predict scale ',i,x.shape,' with ',anchors[-1].shape,' anchors')
            if i < len(self.sizes_list)-2:
                x = self.down_sample[i](x)
            elif i == len(self.sizes_list)-2:
                x = F.Pooling(x,global_pool=True,pool_type='max',kernel=(x.shape[2],x.shape[3]))

        cls_preds = nd.concat(*cls_preds,dim=1).reshape((0,-1,self.num_class+1))
        box_preds = nd.concat(*box_preds,dim=1)
        anchors = nd.concat(*anchors,dim=1)
        return anchors,box_preds,cls_preds

