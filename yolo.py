#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import colorsys
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from nets.yolo4_tiny import YoloBody,YoloBody2,YoloBody5
from utils.util import (DecodeBox, bbox_iou, letterbox_image,
                         non_max_suppression, yolo_correct_boxes)

from utils.cfg import DATASET,NET,USE_ATTEN
#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class YOLO(object):
    # MODEL_PATH = r"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-28voc12+注意力train从0\Epoch36-Total_Loss6.2760-Val_Loss6.1499.pth"
    # MODEL_PATH=r"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-28voc12+CA注意力train从0\Epoch42-Total_Loss6.0775-Val_Loss6.0535.pth"
    # MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-28voc12+CA注意力train从0\Epoch43-Total_Loss6.0961-Val_Loss6.0441.pth"
    #
    # MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-30PAN+yolo_uav从0训练\Epoch76-Total_Loss0.7516-Val_Loss0.4983.pth"
    # MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-29voc12+CA注意力train从0\Epoch40-Total_Loss6.1768-Val_Loss6.2330.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-30PAN+yolo_uav从0训练\Epoch62-Total_Loss0.7352-Val_Loss0.4645.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-1voc12原始\Epoch51-Total_Loss6.3956-Val_Loss6.2855.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-29voc12+panet\Epoch47-Total_Loss6.5437-Val_Loss6.2948.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-2voc12+密集链接\Epoch58-Total_Loss6.3768-Val_Loss6.2264.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-2uav原始\Epoch70-Total_Loss2.3435-Val_Loss1.9377.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-1voc12原始\Epoch47-Total_Loss6.4453-Val_Loss6.2745.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-31voc12CA+FOCAL+PAN\Epoch55-Total_Loss2.2844-Val_Loss2.2558.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-14voc12+CA1CA2\Epoch12-Total_Loss7.3433-Val_Loss7.0294.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-16UAV原始\Epoch64-Total_Loss1.8493-Val_Loss0.7704.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-16UAV原始A\Epoch183-Total_Loss0.7971-Val_Loss0.3715.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-16VOC+密集连接\Epoch57-Total_Loss6.3256-Val_Loss6.2872.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-14voc12+pan\Epoch50-Total_Loss6.4514-Val_Loss6.2655.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-16VOC+PAN\Epoch53-Total_Loss6.4170-Val_Loss6.2519.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-16VOC+PAN\Epoch61-Total_Loss6.3024-Val_Loss6.2645.pth"
    MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-16VOC+PAN\Epoch67-Total_Loss6.3629-Val_Loss6.2761.pth"
    # MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-31voc12CA+FOCAL+PAN\Epoch51-Total_Loss2.3078-Val_Loss2.2570.pth"
    # MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-2uav原始+focalloss+PAN+CA\Epoch63-Total_Loss0.6729-Val_Loss0.5571.pth"
    # MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-3voc12+原始余弦学习率\Epoch53-Total_Loss6.1594-Val_Loss6.0889.pth"
    # MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-3voc12+原始余弦学习率\Epoch52-Total_Loss6.1885-Val_Loss6.1078.pth"
    # MODEL_PATH=R"F:\毕业\代码\代码产生的数据\YOLOV4n1\4-3voc12+ca右边+pan+focal\Epoch52-Total_Loss2.3079-Val_Loss2.2354.pth"
    #r"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-19UNITEd LOSS-alpha=0.01\Epoch39-Total_Loss7.2192-Val_Loss7.5852.pth"

    ANCHOR_PATH= r'model_data/voc07/yolo_anchors.txt'
    CLASS_PATH=r'model_data/voc_classes.txt'
    if DATASET==0:
        ANCHOR_PATH = r'F:\毕业\代码2\yolov4-tiny-pytorch-master\model_data\uav\uav_anchors.txt'
        CLASS_PATH = r"F:\毕业\代码2\yolov4-tiny-pytorch-master\model_data\uav_classes .txt"
    if DATASET == 4:
        ANCHOR_PATH = r'F:\毕业\代码2\yolov4-tiny-pytorch-master\model_data\voc12+07\voc12_anchors.txt'
    if DATASET == 6:
        ANCHOR_PATH = r'F:\毕业\代码2\yolov4-tiny-pytorch-master\model_data\yolo_uav\uav1_anchors.txt'
        CLASS_PATH = r"F:\毕业\代码2\yolov4-tiny-pytorch-master\model_data\uav_classes .txt"
        # r"F:\毕业\代码\代码产生的数据\YOLOV4n1\3-14原来\Epoch38-Total_Loss6.6758-Val_Loss6.7692.pth"
    _defaults = {

        "model_path"    :   MODEL_PATH    ,
        "anchors_path"      : ANCHOR_PATH,
        "classes_path"      : CLASS_PATH,
        "model_image_size"  : (416, 416, 3),
        "confidence"        : 0.4,
        "iou"               : 0.3,
        "cuda"              : True,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   建立yolov4_tiny模型
        #---------------------------------------------------#
        if NET==4:
            self.net = YoloBody2(len(self.anchors[0]), len(self.class_names)).eval()
        elif NET==6:
            self.net = YoloBody5(len(self.anchors[0]), len(self.class_names)).eval()
        else:
            self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        #---------------------------------------------------#
        #   载入yolov4_tiny模型的权重
        #---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        print('Finished!')
        
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    
        #---------------------------------------------------#
        #   建立特征层解码用的工具
        #---------------------------------------------------#
        self.yolo_decodes = []
        self.anchors_mask = [[3,4,5],[1,2,3]]
        for i in range(2):
            self.yolo_decodes.append(DecodeBox(np.reshape(self.anchors,[-1,2])[self.anchors_mask[i]], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img,dtype = np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            if USE_ATTEN:
                atten=outputs[-1]
                outputs=outputs[:-1]
            output_list = []
            for i in range(2):
                output_list.append(self.yolo_decodes[i](outputs[i]))

            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)
        
            #---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            #---------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                if USE_ATTEN:
                    return image, atten
                return image,outputs[-1]
            
            #---------------------------------------------------------#
            #   对预测框进行得分筛选
            #---------------------------------------------------------#
            top_index = batch_detections[:,4] * batch_detections[:,5] > self.confidence
            top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
            top_label = np.array(batch_detections[top_index,-1],np.int32)
            top_bboxes = np.array(batch_detections[top_index,:4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

            #-----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            #-----------------------------------------------------------------#
            if self.letterbox_image:
                boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
            else:
                top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        if USE_ATTEN:
            return image,atten
        return image,outputs[-1]

        # return image

