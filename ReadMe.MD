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

### 3. 载入数据
读取rec文件的代码位于[data_loader.py]。读取时使用了数据增强的方法，这样每次
读取batch时会对图片随机加入一些扰动，增加模型的泛化能力。

### 4. 模型定义

模型定义于文件[model.py]。可通过修改该程序的66行将SSD的bodynet更改为其他模型。
默认使用的是mobilenet1_0（预训练文件18M大小）。
tip:使用预训练网络的参数时pertrained=true，默认为false指只使用网络，参数需要initialize。
需要注意将模型前缀定义一致: prefix="name_"
获取参数方法：
```cython
#使用net.collect_param().keys()查看参数名（不需要name_scope前缀）

param = net.collect_param().get('conv0_weight').data()
```

### 5. 模型训练
训练文件见 train.py 
另外，在[data_analysis.ipynb]中统计了数据集size的分布。

### 6. 模型预测

预测部分位于[predict.py]中。修改其中的图片路径与模型路径，直接执行即可。
基于resnet18_v1训练模型(链接: https://pan.baidu.com/s/1cvVe8Mj4HU-MKG7q7qagGQ 密码: kub9)

### 7. 模型检测
检测详见predict.py文件，修改文件路径即可（img_paths）

### 8. 检测结果
采用resnet18训练的结果，GPU限制效果一般。batch太小，过拟合比较严重。后期在服务器上尝试炼丹
！[result](https://github.com/wk738126046/SSD_Demo/blob/master/results_0.png)
