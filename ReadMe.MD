# My Detection
本工程使用SSD实现了一个物体检测器，基本上具备了从数据采集、数据预处理、模型定义与训练、模型
预测等一系列物体检测的功能，可以轻松应用到其他物体分类问题。

本工程使用Python3.6+mxnet gluon (1.2.0)。

## 组织结构
### 1. 图像标注

使用了图像标注软件[LabelImg](https://github.com/tzutalin/labelImg)进行标注。
标注后会为每张图片生成\[filename\].xml文件，将所有图片文件放到data/img下，将所有
xml文件放入data/label下。

### 2.  im2rec
将图片转成rec文件格式。
参考工程[im2rec tutorial](https://github.com/leocvml/mxnet-im2rec_tutorial)
需要注意：mxnet从rec中读取的图片格式是RGB，而OpenCV读取的是BGR。转换方法如下：
```cython
img = cv2.imread(imgpath)
b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])
```