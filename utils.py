# coding=utf-8
import os

"""
1. some generally tool function
"""
def mkdir_if_not_exist(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
  

"""
2. dataset transform
"""
from PIL import Image
import os
import numpy as np
import xml.dom
def resize_imageset(image_root, out_dir, resize, resample=Image.BILINEAR):
    for imgname in os.listdir(image_root):
        imgpath = image_root + "/" + imgname
        img = Image.open(imgpath)
        img = img.resize(resize, resample)
        img.save(out_dir + "/" + imgname)

def cal_mean(path_root):
    mean = np.array([0., 0., 0.])
    for imgpath in os.listdir(path_root):
        img = Image.open(os.path.join(path_root, imgpath))
        img = np.array(img)
        mean += np.mean(img, axis=(0, 1))
    mean /= len(os.listdir(path_root))
    return mean

def cal_mean_std(path_root):
    imgs = None
    for imgpath in os.listdir(path_root):
        img = Image.open(os.path.join(path_root, imgpath))
        img = np.array(img).reshape((-1, 3))
        if imgs is None:
            imgs = img
        else:
            imgs = np.concatenate((imgs, img), axis=0)
    return np.mean(imgs, axis=0), np.std(imgs, axis=0)


def turn_SDL_to_SDL2(anno_root, image_root, out_anno_root, ext=".jpg"):
    """
    turn SDL format annotation to SDL2 annotation
    SDL format every line is:
        x1 y1 x2 y2 x3 y3 x4 y4 angle label
    SDL2 format every line is:
        xmin \t ymin \t xmax \t ymax \t angle \t label
    """
    mkdir_if_not_exist(out_anno_root)
    for anno_name in os.listdir(anno_root):
        anno_path = os.path.join(anno_root, anno_name)
        image_path = os.path.join(image_root, anno_name[:-4] + ext)
        info = get_info_from_annotaions_SDL(anno_path, image_path, normalize=True)
        boxes, angles, labels = info["boxes"], info["angles"], info["labels"]
        
        f = open(out_anno_root + "/" + anno_name, 'w')
        for i in range(len(boxes)):
            box, angle, label = boxes[i], angles[i], labels[i]
            box = [str(b) for b in box]
            line = "\t".join(box) + "\t" + str(angle) + "\t" + str(label) + "\n"
            f.write(line)
        f.close()


def parse_voc_xml(path):
    '''
    解析voc文件
    :param path:
    :return:
    '''
    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement

    filename = collection.getElementsByTagName("filename")[0].childNodes[0].data
    width = int(collection.getElementsByTagName("width")[0].childNodes[0].data)
    height = int(collection.getElementsByTagName("height")[0].childNodes[0].data)


    objects = collection.getElementsByTagName("object")
    bndboxs = []
    names = []
    for object in objects:
        name = object.getElementsByTagName("name")[0].childNodes[0].data
        bndbox = object.getElementsByTagName("bndbox")[0]
        bndbox = [
            bndbox.getElementsByTagName("xmin")[0].childNodes[0].data,
            bndbox.getElementsByTagName("ymin")[0].childNodes[0].data,
            bndbox.getElementsByTagName("xmax")[0].childNodes[0].data,
            bndbox.getElementsByTagName("ymax")[0].childNodes[0].data,

        ]
        bndbox = list(map(int, bndbox))
        bndbox[0] /= width
        bndbox[2] /= width
        bndbox[1] /= height
        bndbox[3] /= height

        bndboxs.append(bndbox)

        names.append(name)
    return bndboxs, names, filename
        
"""
3. data prepared
"""
import os
import shutil
import xml.dom.minidom as d
from PIL import Image

__label_dict = {}
def get_info_from_annotaions_VOC(annopath, normalize=True):
    """
    args:
        annopath: VOC format annotaion file path
        normalzie: whether boxes is normalzie to [0, 1] (/w, /h)
    
    return:
        boxes: ground truth boxes for detection.
        labels: ground truth boxes's class id, same length with boxes.
        w: image width recored in annoaton file.
        h: image height recored in annoaton file.
    """
    def get_label_id(name):
        if not __label_dict.has_key(name):
            __label_dict[name] = len(__label_dict)
        return __label_dict[name]
    
    dom = d.parse(annopath)
    root = dom.documentElement
    
    size = root.getElementsByTagName("size")[0]
    w = float(size.getElementsByTagName('width')[0].childNodes[0].data)
    h = float(size.getElementsByTagName('height')[0].childNodes[0].data)
    
    boxes = []
    labels = []
    for o in root.getElementsByTagName("object"):
        l = get_label_id(o.getElementsByTagName('name')[0].childNodes[0].data)
        bd = o.getElementsByTagName('bndbox')[0];
        x0 = bd.getElementsByTagName('xmin')[0].childNodes[0].data
        y0 = bd.getElementsByTagName('ymin')[0].childNodes[0].data
        x1 = bd.getElementsByTagName('xmax')[0].childNodes[0].data
        y1 = bd.getElementsByTagName('ymax')[0].childNodes[0].data
        
        if normalize:
            box = [float(x0)/w, float(y0)/ h, float(x1)/w, float(y1)/h]
        else:
            box = [int(x0), int(y0), int(x1), int(y1)]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
            labels.append(l)
        else:
            print( str(box) + " is not valid box in " + annopath + ", just ignore.")
    info = {"boxes": boxes, "labels": labels, "w": w, "h": h}
    return info

def get_info_from_annotaions_SDL(annopath, imgpath=None, normalize=True):
    """
    args:
        annopath: VOC format annotaion file path.
        imgpath: the image path to annotaion file.
        normalzie: whether boxes is normalzie to [0, 1] (/w, /h)
    
    return:
        boxes: ground truth boxes for detection.
        labels: ground truth boxes's class id.
        w: image width recored in annoaton file.
        h: image height recored in annoaton file.
    """
    def get_image_wh(imgpath):
        img = Image.open(imgpath)
        width = img.width
        height = img.height
        return width, height
    
    def split_label(label):
        label = label.strip(' ').strip('\r\n').strip('\n')
        if len(label.split('\t')) >= 9: return label.split('\t')  # not seperate with '\t'
        label = label.split(' ')
        new_label = []
        for l in label:
            if len(l) > 0:
                new_label.append(l)
        return new_label
    
    if imgpath is None and normalize:
        raise ValueError("when set normalize to True, imgpath must be specified")
    
    if imgpath is not None:
        w, h = get_image_wh(imgpath)
    else:
        w, h = None, None
    
    windows = open(annopath).readlines()
    boxes = []
    labels = []
    angles = []
    for label in windows:
        label = split_label(label)
        label = list(map(lambda x: float(x), label[:]))
        print(label)
        x1, y1, x2, y2, x3,y3, x4, y4 = label[:8]
        xmin = min([x1, x2, x3, x4])
        ymin = min([y1, y2, y3, y4])
        xmax = max([x1, x2, x3, x4])
        ymax = max([y1, y2, y3, y4])
        if normalize:
            box = (xmin/w, ymin/h, xmax/w, ymax/h)
        else:
            box = (xmin, ymin, xmax, ymax)
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
            labels.append(label[-1])
            angles.append(label[-2])
        else:
            print( str(box) + " is not valid box in " + annopath + ", just ignore.")
    info = {"boxes": boxes, "labels": labels, "w": w, "h": h, 'angles': angles}
    return info

def get_info_from_annotaions(annopath, fmt, normalize=True, **kwargs):
    if fmt.lower() == "voc":
        return get_info_from_annotaions_VOC(annopath, normalize)
    elif fmt.lower() == "sdl":
        return get_info_from_annotaions_SDL(annopath, kwargs['imgpath'], normalize)
    else:
        raise ValueError("annotation format is not support, annotaion fomrat must be on of [VOC, SDL]")

def list_image_det(lst_file, annotations_root, out_lst_file=None, fmt="VOC", path_root=None, resize=None, resize_out_dir=None):
    """
    lst_file: .lst file generate by im2rec.py
    out_lst_file: output .lst file after modyfied
    annotations_root: annaotaion file's root dir
    fmt: could be 'SDL' or 'VOC'
    path_root: if fmt is 'SDL', then path_root need to spcified, it is direcroty prefix of image path in .lst file
    resize: (w, h) pair. use to set (w, h) in lst file, will not really do resize if resize_out_dir is not specified.
    resie_out_dir: will do resize to imageset, and path_root must be specified.
    """
    # some check
    fmt = fmt.lower()
    if fmt == 'voc':
        ext = ".xml"
    elif fmt == 'sdl':
        if path_root  is None:
            raise ValueError("when data is SDL formt, 'path_root' must specified, it will use to open image.")
        ext = '.txt'
    else:
        raise ValueError("annotation format is not support, 'fmt' must be on of [VOC, SDL]")
    if (resize_out_dir is not None) and path_root is None:
        raise ValueError("path_root must be specified to the dir of images when resize_out_dir has been specified.")
    
    def boxes_to_str(boxes, labels):
        s = ""
        for i, box in enumerate(boxes):
            box = [str(it) for it in box]
            s += str(labels[i]) + "\t" + "\t".join(box) + "\t"
        return s[:-1]
    
    lst_f = open(lst_file)
    if out_lst_file is None:
        o_lst_f = open("tmp", 'w')
    else:
        o_lst_f = open(out_lst_file, 'w')
        
    for line in lst_f.readlines():
        items = line.split('\t')
        imgname = items[-1].split("/")[-1]
        imgname = imgname.split(".")[0]
        
        new_line = items[0] + "\t"
        
        annopath = os.path.join(annotations_root, imgname+ext)
        if fmt == 'voc':
            info = get_info_from_annotaions(annopath, fmt)
        if fmt == 'sdl':
            info = get_info_from_annotaions(annopath, fmt, 
                                                           imgpath=path_root+"/"+items[-1].strip('\n'))
        boxes, labels, w, h = info['boxes'], info['labels'], info['w'], info['h']
        new_line += "4\t" + str(5) + "\t" 
        if resize is not None:
            new_line += str(resize[0]) + '\t' + str(resize[1]) + '\t'
        else:
            new_line += str(w) + '\t' + str(h) + '\t'
        new_line += boxes_to_str(boxes, labels) + "\t"
        
        new_line += items[-1]
        o_lst_f.write(new_line)
    lst_f.close()
    o_lst_f.close()
    
    if out_lst_file is None:
        shutil.move('tmp', lst_file)
        
    if resize_out_dir is not None:
        resize_imageset(path_root, resize_out_dir, resize)
        
    return __label_dict


"""
4. data visualize
""" 
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
def show_image_SDL_annotation(imgpath, annopath=None, color=(0, 1, 0)):
    """
    show single image with sdl format annotation
    """
    def draw_rect(x0, y0, x1, y1, color):
        plt.plot([x0, x1],[y0, y0], color=color)
        plt.plot([x1, x1],[y0, y1], color=color)
        plt.plot([x0, x1],[y1, y1], color=color)
        plt.plot([x0, x0],[y0, y1], color=color)
    fig = plt.figure(figsize=(16, 16), dpi=72)
    image = Image.open(imgpath)
    plt.imshow(np.asarray(image))
    if annopath is not None:
        info = get_info_from_annotaions_SDL(annopath, imgpath, False)
        for box in info["boxes"]:
            draw_rect(box[0], box[1], box[2], box[3], color)
    plt.show()

def try_asnumpy(data):
    try:
        data = data.asnumpy() # if is <class 'mxnet.ndarray.ndarray.NDArray'>
    except BaseException:
        pass
    return data

def box_to_rect(box, color, linewidth=1):
    """convert an anchor box to a matplotlib rectangle"""
    return plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                  fill=False, edgecolor=color, linewidth=linewidth)

def show_images(images, labels=None, rgb_mean=np.array([0, 0, 0]), std=np.array([1, 1, 1]),
                MN=None, color=(0, 1, 0), linewidth=1, figsize=(8, 4), show_text=False, fontsize=5):
    """
    advise to set dpi to 120
        import matplotlib as mpl
        mpl.rcParams['figure.dpi'] = 120
    
    images: numpy images type, shape is (n, 3, h, w), or (n, 2, h, w)
    labels: boxes, shape is (n, m, 5), m is number of box, 5 means every box is [label_id, xmin, ymin, xmax, ymax]
    rgb_mean: if images has sub rgb_mean, shuold specified.
    MN: is subplot's row and col, defalut is (-1, 5), -1 mean row is adaptive, and col is 5
    """
    images = try_asnumpy(images)
    labels = try_asnumpy(labels)
    
    if MN is None:
        M, N = (images.shape[0] + 4) / 5, 5
    else:
        M, N = MN
    _, figs = plt.subplots(M, N, figsize=figsize)
    
    images = (images.transpose((0, 2, 3, 1)) * std) + rgb_mean
    h, w = images.shape[1], images.shape[2]
    for i in range(M):
        for j in range(N):
            if N * i + j < images.shape[0]:
                image = (images[N * i + j] / 255).clip(0, 1)
                figs[i][j].imshow(image)
                
                figs[i][j].axes.get_xaxis().set_visible(False)
                figs[i][j].axes.get_yaxis().set_visible(False)
                
                if labels is not None:
                    label = labels[N * i + j]
                    for l in label:
                        if l[0] < 0: continue
                        l[1], l[2], l[3], l[4] = l[1] * w, l[2] * h, l[3] * w, l[4] * h
                        rect = box_to_rect(l[1:5], color, linewidth)
                        figs[i][j].add_patch(rect)
                        if show_text:
                            figs[i][j].text(l[1], l[2], str(int(l[0])), 
                                            bbox=dict(facecolor=(1, 1, 1), alpha=0.5), fontsize=fontsize, color=(0, 0, 0))
            else:
                figs[i][j].set_visible(False)
    plt.show()

def show_9_images(images, labels=None, rgb_mean=np.array([0, 0, 0]), color=(0, 1, 0), linewidth=1, **kwargs):
    """
    invoke show_images with MN=(3, 3)
    """
    show_images(images, labels, rgb_mean, (3, 3), color, linewidth, figsize=(6, 6), **kwargs)
    
    
def show_det_result(im, out, threshold=0.5, class_names=None, colors = ['blue', 'green', 'red', 'black', 'magenta']):
    """
    im: image data, numpy.array or ndarray
    out: detection result, numpy.array or ndarray
    theshold: score threshold
    class_name: class or labels name
    """
    im = try_asnumpy(im)
    out = try_asnumpy(out)
    
    plt.imshow((im / 255).clip(0, 1))
    for row in out:
        class_id, score = int(row[0]), row[1]
        if class_id < 0 or score < threshold:  # class_id < 0 is background rect
            continue
        color = colors[class_id%len(colors)]
        box = row[2:6] * np.array([im.shape[0],im.shape[1]]*2)
        rect = box_to_rect(box, color, 2)
        plt.gca().add_patch(rect)

        text = class_names[class_id] if class_names else "class " + str(class_id)
        plt.gca().text(box[0], box[1],
                       '{:s} {:.2f}'.format(text, score),
                       bbox=dict(facecolor=color, alpha=0.5),
                       fontsize=10, color='white')
    plt.show()

def show_det_results(images, outs, threshold=0.5, class_names=None, 
                     colors = ['blue', 'green', 'red', 'black', 'magenta'], MN=None, figsize=(8, 4),
                     linewidth=1, show_text=True, fontsize=5):
    """
    im: image data, numpy.array or ndarray
    out: detection result, numpy.array or ndarray
    theshold: score threshold
    class_name: class or labels name
    MN: sub figure's row and col number
    """
    images = try_asnumpy(images)
    outs = try_asnumpy(outs)
    
    if MN is None:
        M, N = (images.shape[0] + 4) / 5, 5
    else:
        M, N = MN
    _, figs = plt.subplots(M, N, figsize=figsize)
    
    for i in range(M):
        for j in range(N):
            if N * i + j < images.shape[0]:
                image = (images[N * i + j] / 255).clip(0, 1)
                figs[i][j].imshow(image)
                figs[i][j].axes.get_xaxis().set_visible(False)
                figs[i][j].axes.get_yaxis().set_visible(False)
                
                if outs is None: continue
                out = outs[N * i + j]
                for row in out:
                    class_id, score = int(row[0]), row[1]
                    if class_id < 0 or score < threshold:  # class_id < 0 is background rect
                        continue
                    color = colors[class_id%len(colors)]
                    box = row[2:6] * np.array([image.shape[0],image.shape[1]]*2)
                    rect = box_to_rect(box, color, linewidth)
                    figs[i][j].add_patch(rect)
                    if show_text:
                        text = class_names[class_id] if class_names else "class " + str(class_id)
                        figs[i][j].text(box[0], box[1],
                                       '{:s} {:.2f}'.format(text, score),
                                       bbox=dict(facecolor=color, alpha=0.5),
                                       fontsize=10, color='white')
                    
            else:
                figs[i][j].set_visible(False)
    plt.show()
    
"""
5. data analysis
"""
def get_all_boxes_from_annotations_SDL2(anno_root):
    boxes = []
    for anno_name in os.listdir(anno_root):
        anno_path = os.path.join(anno_root, anno_name)
        for line in open(anno_path).readlines():
            items = [float(item) for item in line.split('\t')]
            boxes.append(items[:4])
    return boxes

"""
6. evaluate
"""
import numpy as np
"""
attention: 
    1. use numpy, note type, v(int) = v(float), will cilp value
    2. use numpy, note copy, use copy when need deep copy to avoid shallow copy
"""
def IOU_1v1(box1, box2):
    """
    box=[xmin, ymin, xmax, ymax]
    """
    box = box1.copy()
    box[0] = max([box1[0], box2[0]])
    box[1] = max([box1[1], box2[1]])
    box[2] = min([box1[2], box2[2]])
    box[3] = min([box1[3], box2[3]])
    if box[0] >= box[2] or box[1] >= box[3]:
        return 0.
    area = (box[2] - box[0]) * (box[3] -  box[1])
    area1 = (box1[2] - box1[0]) * (box1[3] -  box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] -  box2[1])
    return float(area) / (area1 + area2 - area)

def IOU_NvN(boxes1, boxes2):
    """
        boxes1: numpy array, shape=(N, 4)
        boxes2: numpy array, shape=(N, 4)
    return 
        IOU: numpy array, shape=(N,)
    """
    def Area(boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    boxes1 = boxes1.astype('float64')
    boxes2 = boxes2.astype('float64')
    boxes = np.zeros(shape=boxes1.shape)
    boxes[:, 0] = np.max([boxes1[:, 0], boxes2[:, 0]], axis=0)
    boxes[:, 1] = np.max([boxes1[:, 1], boxes2[:, 1]], axis=0)
    boxes[:, 2] = np.min([boxes1[:, 2], boxes2[:, 2]], axis=0)
    boxes[:, 3] = np.min([boxes1[:, 3], boxes2[:, 3]], axis=0)
    area = Area(boxes)
    area[boxes[:, 0] >= boxes[:, 2]] = 0
    area[boxes[:, 1] >= boxes[:, 3]] = 0
    area1 = Area(boxes1)
    area2 = Area(boxes2)
    return area / (area1 + area2 - area)

def IOU_1vN(box1, boxes2):
    """
    box=[xmin, ymin, xmax, ymax]
    """
    boxes1 = np.tile(box1, (boxes2.shape[0], 1))
    return IOU_NvN(boxes1, boxes2)

def IOU(box1, box2):
    assert box1.shape[-1] == 4 and box2.shape[-1] == 4
    assert len(box1.shape) <= 2 and len(box2.shape) <= 2
    if len(box1.shape) == 1:
        if len(box2.shape) == 1:  # 1 v 1
            return IOU_1v1(box1, box2)
        else:                     # 1 v N
            return IOU_1vN(box1, box2)
    else:
        if len(box2.shape) == 1:  # N v 1
            return IOU_1vN(box2, box1)
        else:                     # N v N
            return IOU_NvN(box1, box2)


# def cal_pred_scores(outs, labels, overlap_threshold=0.01):
#     # compute tp_score, num_pred, num_gt
#     tp_scores, fp_scores, num_pred, num_gt = [],[], 0., 0.
#     for n in xrange(outs.shape[0]): # every image
#         out = outs[n]
#         out = out[out[:, 0] >= 0]
#         out = np.array(sorted(out, key=lambda row: -row[1])) # sort out by score
#         tmp_label = labels[n]
#         label = tmp_label[tmp_label[:, 0] >= 0]

#         num_pred += out.shape[0]
#         num_gt += label.shape[0]

#         gt_flags = np.array([True] * label.shape[0])
#         for i, pred_box in enumerate(out):       # every pred box
#             if np.sum(gt_flags) <= 0: break
#             overlaps = IOU(pred_box[2:], label[:, 1:])
#             maxi = np.argmax(overlaps)
#             max_overlap = overlaps[maxi]
#             if max_overlap >= overlap_threshold and gt_flags[maxi]:
#                 gt_flags[maxi] = False
#                 tp_scores.append(pred_box[1])
#             else:
#                 fp_scores.append(pred_box[1])
#         fp_scores.extend(list(out[i:, 1]))
#     tp_scores = np.array(tp_scores)
#     fp_scores = np.array(fp_scores)
#     return tp_scores, fp_scores, num_pred, num_gt

def cal_pred_scores_pair(outs, labels, overlap_threshold=0.01):
    # compute tp_score, num_pred, num_gt
    scores, is_tp, num_pred, num_gt = [], [], 0., 0.
    for n in range(outs.shape[0]): # every image
        out = outs[n]
        out = out[out[:, 0] >= 0]
        out = np.array(sorted(out, key=lambda row: -row[1])) # sort out by score
        tmp_label = labels[n]
        label = tmp_label[tmp_label[:, 0] >= 0]

        num_pred += out.shape[0]
        num_gt += label.shape[0]

        gt_flags = np.array([True] * label.shape[0])
        i = 0
        for i, pred_box in enumerate(out):       # every pred box
            if np.sum(gt_flags) <= 0: break
            overlaps = IOU(pred_box[2:], label[:, 1:])
            maxi = np.argmax(overlaps)
            max_overlap = overlaps[maxi]
            if max_overlap >= overlap_threshold and gt_flags[maxi]:
                gt_flags[maxi] = False
                scores.append(pred_box[1])
                is_tp.append(True)
            else:
                scores.append(pred_box[1])
                is_tp.append(False)
        if i < out.shape[0]:
            scores.extend(list(out[i:, 1]))
        is_tp.extend([False] * out[i:].shape[0])
        
    scores = np.array(scores)
    is_tp = np.array(is_tp)
    return scores, is_tp, num_pred, num_gt

def cal_scores_recall_prec(outs, labels, overlap_threshold=0.01, verbose=False):
    """
        scores 是升序排列的所有box的score集合
        tp[i] 表示使用score阈值为scores[i]时的true positive的数量
        fp[i] 表示使用score阈值为scores[i]时的fp positive的数量
    """
    scores, is_tp, num_pred, num_gt = cal_pred_scores_pair(outs, labels, overlap_threshold)
    if verbose:
        print (len(scores), len(is_tp), int(num_pred), int(num_gt))
    
    idx = np.argsort(scores)
    scores = scores[idx]
    is_tp = is_tp[idx]
    
    tp = np.zeros(shape=is_tp.shape)
    fp = np.zeros(shape=is_tp.shape)
    N = is_tp.shape[0]
    tp[N-1] = is_tp[N-1]
    fp[N-1] = (not is_tp[N-1])
    for i in range(N-2, -1, -1):
        # score_th = scores[i]
        tp[i] = tp[i+1]
        fp[i] = fp[i+1]
        if is_tp[i]:
            tp[i] += 1
        else:
            fp[i] += 1
            
    prec = tp / (tp + fp)
    recall = tp / num_gt
    return scores, recall, prec


EPS = 1e-10
def evaluate_MAP(outs, labels, overlap_threshold=0.01, ap_version="11points", verbose=False):
    outs = outs.asnumpy()   # share memory with outs's ndarray
    labels = labels.asnumpy()
    
    # tp_scores, fp_scores, num_pred, num_gt = cal_pred_scores(outs, labels, overlap_threshold)
        # compute 11 point AP
#     AP = 0.
#     for i in range(11): # 0-1.0
#         score_th = i / 10.0
#         tp = np.sum(tp_scores > score_th)
#         fp = np.sum(fp_scores > score_th)
#         print tp, fp
#         recall = tp / (tp + fp + EPS)
#         prec = tp / num_pred
#         AP += prec
#     AP /= 11
#     if verbose:
#         print tp_scores.shape[0], fp_scores.shape[0], int(num_pred), int(num_gt)
#     return AP

    scores, recall, prec = cal_scores_recall_prec(outs, labels, overlap_threshold, verbose)
    
    if ap_version == "11points":
        start_idx = 0
        AP = 0.
        max_prec = 0
        recall_th = 1.0
        for i in range(recall.shape[0]):
            if recall[i] < recall_th:
                AP += max_prec
                recall_th -= 0.1
                if recall_th < 0: break
            if max_prec < prec[i]:
                max_prec = prec[i]
                    
#         for j in range(10, -1, -1):
#             for i in range(start_idx, N):
#                 if recall[i] >= j / 10.0:
#                     if max_prec < prec[i]:
#                         max_prec = prec[i]
#                 else:
#                     AP += max_prec
#                     start_idx = i
#                     break
        return AP / 11
#         for i in range(N-1, -1, -1):
#             if recall[i] > recll[]
    elif ap_version.lower() == "integral":  # recall 算出来是降序的，因为scores升序，score越大，recall越小
        delta_recall = recall[:-1] - recall[1:]
        print (prec[0], recall[0])
        return np.sum(delta_recall * prec[1:] + prec[-1] * recall[-1])

def draw_ROC(outs, labels, overlap_threshold=0.01, verbose=False, show=True, color='r', label_suffix=""):
    outs = outs.asnumpy()   # share memory with outs's ndarray
    labels = labels.asnumpy()
    scores, recall, prec = cal_scores_recall_prec(outs, labels, overlap_threshold)
    plt.plot(recall,prec, '-', label="recall prec"+label_suffix, color=color)
    plt.plot(recall, scores, '--', label="recall score"+label_suffix, color=color)
    plt.legend(loc="upper right")
    if show:
        plt.show()
    
def find_best_score_th(outs, labels, overlap_threshold=0.01):
    scores, recall, prec = cal_scores_recall_prec(outs, labels, overlap_threshold)
    max_area = 0
    max_i = -1
    for i in range(recall.shape[0]):
        if max_area < recall[i] * prec[i]:
            max_area = recall[i] * prec[i]
            max_i = i
    return scores[i]