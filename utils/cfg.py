# -*- coding: utf-8 -*-
import os

# SAVE_PATH='2021年3月6日unite loss系数1忽略小于0.01的'
SAVE_PATH='3-19UNITEd LOSS-alpha=0.1 无人机'
SAVE_PATH='4-2uav原始+focalloss+PAN+CA'
SAVE_PATH='4-2voc12+密集链接'
SAVE_PATH='4-16UAV原始A'
SAVE_PATH='4-16VOC+PAN'
# SAVE_PATH='0222uav11类+eiou'
# SAVE_PATH='0221（cls+conf+eiou）+LF1+label=0.1'



MODEL_PATH=r'F:\毕业\代码2\yolov4-tiny-pytorch-master\model_data\yolov4_tiny_weights_voc.pth'
# MODEL_PATH=r"F:\毕业\代码2\Epoch62-Total_Loss5.6128-Val_Loss5.7007.pth"

# MODEL_PATH=r"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-29voc12+CA注意力train从0\Epoch40-Total_Loss6.1768-Val_Loss6.2330.pth"
# MODEL_PATH=r"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-23 voc eiou 余弦学习率\Epoch36-Total_Loss8.4656-Val_Loss8.5597.pth"
# MODEL_PATH=r"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-21加Panet\Epoch58-Total_Loss6.5375-Val_Loss6.8611.pth"
# MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-19UNITEd LOSS-alpha=0.01\Epoch52-Total_Loss7.1294-Val_Loss7.5578.pth"
# MODEL_PATH=r"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-17uav+注意力上层通道\Epoch48-Total_Loss0.5295-Val_Loss0.4515.pth"
# MODEL_PATH=r"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-15注意力两个通道\Epoch49-Total_Loss5.8593-Val_Loss6.7246.pth"
# MODEL_PATH=r"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-14zhuyili\Epoch50-Total_Loss6.5117-Val_Loss6.9832.pth"
# MODEL_PATH=r"F:\毕业\代码\代码产生的数据\YOLOV4\Epoch21-Total_Loss4.5107-Val_Loss4.8854.pth"
# MODEL_PATH=r"F:\毕业\代码\代码产生的数据\YOLOV4\unite实验2小系数\Epoch7-Total_Loss4.1480-Val_Loss4.1712.pth"
# MODEL_PATH=r"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-22 voc eiou1\Epoch45-Total_Loss7.9934-Val_Loss8.6076.pth"
SAVE_PATH= r'F:\毕业\代码\代码产生的数据\YOLOV4n1\\'+SAVE_PATH


###################学习率
Init_learn_rate=0.003
GAMMA=0.95
lr_STEP=1
PreTrain=1
Freeze_Epoch=200
USE_ATTEN=0
Focal_LOSS=0
###############
Load_backbone_only=1# 注意改学习率为0.003，下降幅度0.95

Train_backbone_only=0#只训练主干网
Train_changed_net_only=0#只训练和预训练权重不一样的部分，用于新改了的模型
#####网络选取#########
NET=4 #0原始
            #1按照某核心论文加，改卷积层数
            #3加空洞卷积
            #4 PANet
            #6 轻量级密集连接 yolobody5
Shufflenet=0#backbone的选取
######trick
SMOOTH_LABEL=0.01
MOSAIC=False
######
#####数据集选取#########
USE_yolo_uav=0

DATASET=4#0 uav;
            # 1 voc
            #2 UAV+9voc
            #3 遥感
            #4 voc2012+2007
            #5 vco2007train(5000张）
            #6  yolo uav
########损失函数选取,二者一个为0一个为1
USE_CIOU=1
USE_EIOU=0

LOSS_UNITE=0
######固定
count=0
save_flag=1#决定了使用loss值存储
dataset=['UAV','voc','UAV+9voc','遥感','voc12+07','voc07train','yoloUAV']
net=['原始','核心论文','','加空洞卷积','密集链接融合特征','注意力可视化','密集链接轻量']
##################
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
f = open(SAVE_PATH + "/实验参数.txt", 'w',encoding='utf-8')
f.write('smoooth_label=%r' % SMOOTH_LABEL)
f.write('\nMosaic=%r' % MOSAIC)
f.write('\nPreTrain=%r' % PreTrain)
f.write('\nTrain_backbone_only=%r' % Train_backbone_only)
f.write('\nDATASET=%r' % dataset[DATASET])
f.write('\nNET=%r' % net[NET])
f.write('\nUSE_CIOU=%r' % USE_CIOU)
f.write('\nUSE_EIOU=%r' % USE_EIOU)
f.write('\n学习率下降系数=%r' % GAMMA)
f.write('\nMODEL_PATH=%r' % MODEL_PATH)
f.write('\nLOSS_UNITE=%r' % LOSS_UNITE)
f.write('\n只加载主干网=%r' % Load_backbone_only)
f.write('\nUSE_ATTEN=%r' % USE_ATTEN)

f.close()
# print('count',count)


