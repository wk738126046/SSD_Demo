# --conding: utf-8 --
import numpy as np
import mxnet as mx
from data_loader import get_iterators
from utils import *
from model import SSD,sizes_list,ratios_list
import time
from mxnet.ndarray.contrib import MultiBoxTarget, MultiBoxPrior
from mxnet import ndarray as nd
from mxnet import gluon


data_shape = (3,512,512)
batch_size = 4
std = np.array([61.04467501, 60.03631381, 60.7750983 ])
rgb_mean = np.array([ 130.063048,  129.967301,  124.410760])
ctx = mx.gpu(0)
resize = data_shape[1:]
rec_prefix = './dataset/data/rec/img_'+str(resize[0])+'_'+str(resize[1])
# num_class = 1
'''
loss define
'''
class FocalLoss(gluon.loss.Loss):
    def __init__(self,axis=-1,alpha=0.25,gamma=2,batch_axis=0,**kwargs):
        super(FocalLoss,self).__init__(None,batch_axis,**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.axis = axis
        self.batch_axis = batch_axis

    def hybrid_forward(self, F, y,label):
        y=F.softmax(y)
        py = y.pick(label,axis=self.axis,keepdims=True)
        loss = -(self.alpha *((1-py)**self.gamma))*nd.log(py)
        return nd.mean(loss,axis=self.batch_axis,exclude=True)

class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self,batch_axis=0,**kwargs):
        super(SmoothL1Loss,self).__init__(None,batch_axis,**kwargs)
        self.batch_axis = batch_axis

    def hybrid_forward(self, F, y,label,mask):
        loss = F.smooth_l1((y-label)*mask,scalar=1.0)
        return nd.mean(loss,axis=self.batch_axis,exclude=True)

lossdoc='''
使用AP分数作为分类评价的标准。
由于在模型检测问题中，反例占据了绝大多数，即使把所有的边框全部预测为反例已然会具有不错的精度。
因此不能直接使用分类精度作为评价标准。
 AP曲线考虑在预测为正例的标签中真正为正例的概率（查准率， precise）
 以及在全部正例中预测为正例的概率（召回率， recall），更能反映模型的正确性。
 
 使用MAE（平均绝对值误差）作为回归评价的标准。
'''
from mxnet import metric
from mxnet import autograd
from mxnet.ndarray.contrib import MultiBoxDetection
import numpy as np
'''
trian net
'''
def evaluate_acc(net,data_iter,ctx):
    data_iter.reset()
    box_metric = metric.MAE()
    outs,labels = None,None
    for i, batch in enumerate(data_iter):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        anchors,box_preds,cls_preds = net(data)
        #MultiBoxTraget 作用是将生成的anchors与哪些ground truth对应，提取出anchors的偏移和对应的类型
        #预测的误差是每次网络输出的预测框g与anchors的差分别/anchor[xywh]，然后作为smoothL1（label-g）解算，g才是预测
        # 正负样本比例1：3
        box_offset,box_mask,cls_labels=MultiBoxTarget(anchors,label,cls_preds.transpose((0,2,1)),
                                                      negative_mining_ratio=3.0)
        box_metric.update([box_offset],[box_preds*box_mask])
        cls_probs = nd.SoftmaxActivation(cls_preds.transpose((0,2,1)),mode='channel')
        #对输出的bbox通过NMS极大值抑制算法筛选检测框
        out = MultiBoxDetection(cls_probs,box_preds,anchors,force_suppress=True, clip=False, nms_threshold=0.45)
        if out is None:
            outs = out
            labels = label
        else:
            outs = nd.concat(outs,out,dim=0)
            labels = nd.concat(labels,label,dim=0)
    AP = evaluate_MAP(outs,labels)
    return AP,box_metric


info = {"train_ap": [], "valid_ap": [], "loss": []}

def mytrain(net,train_data,start_epoch, end_epoch, cls_loss,box_loss,trainer=None):
    if trainer is None:
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5, 'wd': 5e-4})
    box_metric = metric.MAE()

    for e in range(start_epoch, end_epoch):
        train_data.reset()
        box_metric.reset()
        tic = time.time()
        _loss = [0, 0]
        if e == 100 or e == 150 or e == 180:
            trainer.set_learning_rate(trainer.learning_rate * 0.5)

        outs, labels = None, None
        for i, batch in enumerate(train_data):
            data = batch.data[0].as_in_context(ctx)
            label = batch.label[0].as_in_context(ctx)

            with autograd.record():
                anchors, box_preds, cls_preds = net(data)
                # print(anchors, box_preds, cls_preds)
                # negative_mining_ratio，在生成的mask中增加*3的反例参加loss的计算。
                box_offset, box_mask, cls_labels = MultiBoxTarget(anchors, label, cls_preds.transpose(axes=(0, 2, 1)),
                                                                  negative_mining_ratio=3.0)  # , overlap_threshold=0.75)

                loss1 = cls_loss(cls_preds, cls_labels)
                loss2 = box_loss(box_preds, box_offset, box_mask)
                loss = loss1 + loss2
                # print(loss)
            loss.backward()
            trainer.step(data.shape[0])
            _loss[0] += nd.mean(loss1).asscalar()
            _loss[1] += nd.mean(loss2).asscalar()

            cls_probs = nd.SoftmaxActivation(cls_preds.transpose((0, 2, 1)), mode='channel')
            out = MultiBoxDetection(cls_probs, box_preds, anchors, force_suppress=True, clip=False, nms_threshold=0.45)
            if outs is None:
                outs = out
                labels = label
            else:
                outs = nd.concat(outs, out, dim=0)
                labels = nd.concat(labels, label, dim=0)

            box_metric.update([box_offset], [box_preds * box_mask])

        train_AP = evaluate_MAP(outs, labels)
        valid_AP, val_box_metric = evaluate_acc(valid_data, ctx)
        info["train_ap"].append(train_AP)
        info["valid_ap"].append(valid_AP)
        info["loss"].append(_loss)

        if (e + 1) % 10 == 0:
            print("epoch: %d time: %.2f loss: %.4f, %.4f lr: %.4f" % (
            e, time.time() - tic, _loss[0], _loss[1], trainer.learning_rate))
            print("train mae: %.4f AP: %.4f" % (box_metric.get()[1], train_AP))
            print("valid mae: %.4f AP: %.4f" % (val_box_metric.get()[1], valid_AP))


if __name__ == '__main__':
    #1. get dataset and show
    train_data,valid_data,class_names,num_classes = get_iterators(rec_prefix,data_shape,batch_size)
    ##label数量需要大于等于3
    if train_data.next().label[0][0].shape[0] < 3:
        train_data.reshape(label_shape=(3, 5))
    valid_data.sync_label_shape(train_data)

    train_data.reset()
    batch = train_data.next()
    images = batch.data[0][:].as_in_context(mx.gpu(0))
    labels = batch.label[0][:].as_in_context(mx.gpu(0))
    show_images(images.asnumpy(),labels.asnumpy(),rgb_mean,std,show_text=True,fontsize=6,MN=(2,4))
    print(labels.shape)

    #2. net initialize
    net = SSD(1,verbose=True,prefix='ssd_')
    # print(net)
    tic = time.time()
    anchors,box_preds,cls_preds = net(images)
    print(time.time()-tic)
    print(net)
    #MultiBoxTraget 作用是将生成的anchors与哪些ground truth对应，提取出anchors的偏移和对应的类型
    #预测的误差是每次网络的预测框g与anchors的差分别/anchor[xywh]，然后作为smoothL1（label-g）解算，g才是预测
    # box_offset,box_mask,cls_labels = MultiBoxTarget(anchors,batch.label[0],cls_preds)
    # box_offset, box_mask, cls_labels = MultiBoxTarget(anchors, batch.label[0].as_in_context(mx.gpu(0)),
    #                                                   cls_preds.transpose((0, 2, 1)))

    #3. loss define
    # cls_loss = FocalLoss()
    # box_loss = SmoothL1Loss()

    #4. train
    # mytrain(net, train_data, 0, 200, cls_loss, box_loss)
    # net.save_params("./model/papercupDetect.param")