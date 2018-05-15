# --coding: utf-8 --
import cv2
import matplotlib.pyplot as plt
from utils import show_det_result
import numpy as np
import mxnet as mx
from model import SSD,ratios_list
import time
from mxnet import ndarray as nd
from mxnet import image
from mxnet.ndarray.contrib import MultiBoxDetection
import myDect_config
import os

def load_weight():
    sizes_list = myDect_config.sizes_list
    num_class = myDect_config.num_class
    ratios_list = myDect_config.ratios_list
    ctx = myDect_config.ctx
    net = SSD(num_class,sizes_list,ratios_list,ctx,prefix='ssd_')
    # net.load_params('./Model/mobilenet1.0_papercupDetect.param',ctx=ctx)
    net.load_params('./Model/resnet18_papercupDetect.param',ctx=ctx)
    # net.load_params('./Model/vgg11bn29_512x512_data_sizes.param')
    return net

def predict(img_nd,net):
    #predict
    tic = time.time()
    anchors,box_preds,cls_preds = net(img_nd)
    #process result
    cls_probs = nd.SoftmaxActivation(cls_preds.transpose((0,2,1)),mode='channel')
    out = MultiBoxDetection(cls_probs,box_preds,anchors,force_suppress=True,clip=False,nms_threshold=0.1)
    out = out.asnumpy()
    print(out.shape)
    print('detect time:',time.time()-tic)
    return out

def detector(net, img_paths, threshold=0.3):
    img_nds = None
    print(img_paths)
    tic = time.time()
    sizes = []
    for img_path in img_paths:
        # read img
        img = plt.imread(img_path)
        sizes.append(img.shape[:2])
        # test gray img
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # grb <-> bgr
        img = cv2.resize(img, myDect_config.resize)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = (img - myDect_config.rgb_mean) / myDect_config.std
        # img = (cv2.resize(img, myDect_config.resize) - myDect_config.rgb_mean) / myDect_config.std
        img_nd = nd.array(img,ctx=myDect_config.ctx)
        img_nd = img_nd.expand_dims(0).transpose((0,3,1,2))
        if img_nds is None:
            img_nds = img_nd
        else:
            img_nds = nd.concat(img_nds,img_nd,dim=0)
        print('complete once calc')
    print('IO time:',time.time()-tic)
    outs = predict(img_nds,net)

    all_results = []
    for i, out in enumerate(outs):
        img_w = sizes[i][1]
        img_h = sizes[i][0]
        results = []
        colom_mask = (out[:,1] > threshold) * (out[:,0] != -1)
        out = out[colom_mask, :]
        for item in out:
            class_name = myDect_config.class_names[int(item[0])]
            prob = float(item[1])
            cx = float((item[2]+item[4])/2)*img_w
            cy = float((item[3]+item[5])/2)*img_h
            w = float((item[4]-item[2]))*img_w
            h = float((item[5]-item[3]))*img_h
            result = [class_name,prob,[cx,cy,w,h]]
            results.append(result)
        all_results.append(results)
    return  all_results


if __name__ == '__main__':
    # print(cv2.__version__)
    colors = ['red', 'blue', 'yellow', 'green']
    # CPU上处理时间随batch线性增长（1s/图），gpu（TITAN X）上可同时算8张(约2.5s)。
    img_paths = ['test.jpg','test2.jpg']
    #
    # img_paths =[]
    # img_path = os.walk('./detectimage/')
    # for root,dir,files in img_path:
    #     print(root)
    #     for file in files:
    #         print(root+file)
    #         img_paths.append(root+file)
    #
    net = load_weight()
    outs = detector(net, img_paths,threshold=0.3)
    print(outs)
    for i, out in enumerate(outs):
        _, figs = plt.subplots()
        img = plt.imread(img_paths[i])

        # # 灰度测试
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        figs.imshow(img)
        # plt.gca()
        # tmp = [img.shape[1], img.shape[0]] * 2
        for j,item in enumerate(out):
            box = np.array(item[2])
            rect = plt.Rectangle((box[0] - box[2] / 2, box[1] - box[3] / 2), box[2], box[3], fill=False, color=colors[j % 4] )
            figs.add_patch(rect)
            figs.text(box[0] - box[2] / 2, box[1] - box[3] / 2, item[0] + ' ' + '%4f' % (item[1]), color = colors[j % 4])
        # plt.imshow(img)
        plt.savefig('results_%d.png'%(i))
        plt.show()
    # plt.savefig('results.png')
    # print(outs)

