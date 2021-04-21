import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO
import torch.nn as nn
yolo = YOLO()
#-------------------------------------#
#   调用摄像头
# PATH=r"D:\picture\3.jpg"

# PATH=R"D:\dataset\yolo_uav\JPEGImages\00424.jpg"
# PATH=R"D:\dataset\yolo_uav\JPEGImages\00832.jpg"
# PATH=R"D:\picture\1\1.jpg"
#
#
PATH=R"D:\picture\VCO07\000059.jpg"
# PATH=R"D:\picture\VCO07\000084.jpg"
# PATH=R"D:\picture\VCO07\000004.jpg"
# PATH=R"D:\picture\VCO07\000152.jpg"
PATH=R"D:\picture\VCO07\000155.jpg"
PATH=R"D:\picture\VCO07\000069.jpg"
PATH=R"D:\picture\1\1.jpg"
PATH = r"D:\picture\3.jpg"
# PATH = R"D:\dataset\yolo_uav\JPEGImages\00119.jpg"
# PATH = R"D:\dataset\yolo_uav\JPEGImages\00168.jpg"
# PATH=R"D:\picture\000300.jpg"
# PATH=R"D:\picture\JPEGImages\00180.jpg"
# PATH=R"D:\picture\JPEGImages\01054.jpg"
# PATH=R"D:\picture\VCO07\001884.jpg"
# PATH=R"D:\picture\VCO07\000103.jpg"
# PATH=R"D:\picture\VCO07\000111.jpg"
# PATH=R"D:\picture\VCO07\000038.jpg"
# PATH=R"D:\picture\VCO07\000166.jpg"
# PATH=R"D:\picture\VCO07\000097.jpg"
# PATH=R"D:\picture\000155.jpg"
# PATH=r"D:\picture\1.mp4"
# PATH=r"D:\dataset3\RSOD\mix\JPEGImages\aircraft_1006.jpg"
# PATH=r"D:\视频\scu2.mp4"
# PATH=r"D:\picture\uav.jpg"
# PATH=r"D:\picture\2.jpg"
# PATH=R"D:\picture\000293.jpg"
# PATH=r"D:\picture\000291.jpg"
###########PATH=r"D:\picture\000318.jpg"
# PATH=r"D:\picture\000155.jpg"
# PATH=r"D:\picture\000166.jpg"
# PATH=R"D:\picture\JPEGImages\000143.jpg"
# PATH=R"D:\picture\JPEGImages\00185.jpg"
# PATH=R"D:\picture\JPEGImages\000229.jpg"
# PATH=R"D:\picture\JPEGImages\00962.jpg"
# PATH=R"D:\picture\000330.jpg"
# PATH=R"D:\picture\000153.jpg"
# PATH=R"D:\picture\000173.jpg"
# PATH=R"D:\picture\2.jpg"
####PATH=R"D:\picture\JPEGImages\00091.jpg"
# PATH=R"D:\picture\JPEGImages\01110.jpg"

# PATH=R"D:\picture\t1.jpg"
SAVE_video=1

import cv2
import math
import random
import numpy as np
import os


def imshowAtt(beta, img=None):
    cv2.namedWindow("img")
    cv2.namedWindow("img1")
    if img is None:
        img = cv2.imread(
            os.path.join("VOCdevkit\VOC2007\JPEGImages/000001.jpg"), 1
        )  # the same input image
    # img=img.numpy()
    img=np.array(img)
    h, w, c = img.shape
    img1 = img.copy()
    img = np.float32(img) / 255

    pool_c = nn.AdaptiveAvgPool2d((1, 1))
    t1=pool_c(beta).squeeze(0).squeeze(1).squeeze(1)

    weight=torch.softmax(t1,dim=0)
    # ####tm=torch.mean(t,axis=1,keepdim=True)

    batch_cams = (weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * beta).sum(dim=1)
    beta=batch_cams

    (height, width) = beta.shape[1:]
    h1 = int(math.sqrt(height))
    w1 = int(math.sqrt(width))

    for i in range(height):
        img_show = img1.copy()
        h2 = int(i / w1)
        w2 = int(i % h1)

        mask = np.zeros((h1, w1), dtype=np.float32)
        mask[h2, w2] = 1
        mask = cv2.resize(mask, (w, h))
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img_show * mask
        color = (random.random(), random.random(), random.random())
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img_show = img_show + 0.8 * clmsk - 0.8 * mskd

        cam = beta[0, i, :]
        cam = cam.view(h1, w1).data.cpu().numpy()
        cam = cv2.resize(cam, (w, h))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        # cam = 1 / (1 + np.exp(-cam))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * (cam))
        cv2.imwrite("att.jpg", cam)
        cv2.imwrite("img.jpg", np.uint8(img_show))
        # cv2.imshow("img", cam)
        # cv2.imshow("img1", np.uint8(img_show))
        k = cv2.waitKey(0)
        if k & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit(0)

# print(args.save_video,'args.save_video')

#-------------------------------------#
# capture=cv2.VideoCapture(0)
fps = 0.0
i=1

# coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from nets.yolo4_tiny import YoloBody
from utils.cfg import NET
def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)

    # 获取模型输出的feature/score
    model.eval()
    features = model.features(img)
    output = model.classifier(features)

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘

while(True):
    i+=1
    t1 = time.time()
    # 读取某一帧
    frame1=cv2.imread(PATH)
    print(type(frame1))
    # 格式转变，BGRtoRGB
    # frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame1))
    # 进行检测
    if NET!=4:
        frame_box ,beta= np.array(yolo.detect_image(frame))
    if NET==4:
        frame_box, beta = np.array(yolo.detect_image(frame))
    print(type(frame_box))
    frame_box=np.array(frame_box)
    # RGBtoBGR满足opencv显示格式
    # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    # beta=frame
    # imshowAtt(beta,frame1)
    # model = YoloBody(3, 20).eval()
    # draw_CAM(model,PATH,'detection_result.jpg')
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    # frame_box = cv2.putText(frame_box, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame_box)
    if SAVE_video:
        cv2.imwrite('detection_result.jpg', frame_box)
        cv2.waitKey(0)
    break


