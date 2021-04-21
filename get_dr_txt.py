#-------------------------------------#
#       mAP所需文件计算代码
#       具体教程请查看Bilibili
#       Bubbliiiing
#-------------------------------------#
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from yolo import YOLO
from nets.yolo4_tiny import YoloBody
from PIL import Image,ImageFont, ImageDraw
from utils.util import non_max_suppression, bbox_iou, DecodeBox,letterbox_image,yolo_correct_boxes
from tqdm import tqdm
from utils.cfg import DATASET

class mAP_Yolo(YOLO):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.01
        self.iou = 0.5
        result_path="./input/detection-results/"
        if DATASET==0:
            result_path="./input/uav/detection-results/"
        if DATASET==4:
            result_path="./input/voc12/detection-results/"
        if DATASET==6:
            result_path="./input/yolo_uav/detection-results/"
        f = open(result_path+image_id+".txt","w")


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
        
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return 
            
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

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

yolo = mAP_Yolo()
image_ids = open('F:\毕业\数据集\VOC2007/ImageSets/Main/val.txt').read().strip().split()
if DATASET==0:
    image_ids = open(r"F:\毕业\数据集\MIXuav\yolo_uav\val.txt").read().strip().split()
if DATASET==4:
    image_ids = open(r"F:\毕业\数据集\VOC2007test\ImageSets/Main/val.txt").read().strip().split()
if DATASET==6:
    image_ids = open(r"D:\dataset\yolo_uav\val.txt").read().strip().split()
if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

if not os.path.exists("./input/uav"):
    os.makedirs("./input/uav")
if not os.path.exists("./input/uav/detection-results"):
    os.makedirs("./input/uav/detection-results")
if not os.path.exists("./input/uav/images-optional"):
    os.makedirs("./input/uav/images-optional")

if not os.path.exists("./input/voc12"):
    os.makedirs("./input/voc12")
if not os.path.exists("./input/voc12/detection-results"):
    os.makedirs("./input/voc12/detection-results")
if not os.path.exists("./input/voc12/images-optional"):
    os.makedirs("./input/voc12/images-optional")

if not os.path.exists("./input/yolo_uav"):
    os.makedirs("./input/yolo_uav")
if not os.path.exists("./input/yolo_uav/detection-results"):
    os.makedirs("./input/yolo_uav/detection-results")
if not os.path.exists("./input/yolo_uav/images-optional"):
    os.makedirs("./input/yolo_uav/images-optional")
for image_id in tqdm(image_ids):
    image_path = "F:\毕业\数据集\VOC2007/JPEGImages/"+image_id+".jpg"
    if DATASET==0:
        image_path=r'F:\毕业\数据集\MIXuav\yolo_uav\JPEGImages/'+image_id+".jpg"
    if DATASET==4:
        image_path = r'F:\毕业\数据集\VOC2007test\JPEGImages/' + image_id + ".jpg"
    if DATASET==6:
        image_path = r'D:\dataset\yolo_uav\JPEGImages/' + image_id + ".jpg"
    image = Image.open(image_path)
    # 开启后在之后计算mAP可以可视化
    # image.save("./input/images-optional/"+image_id+".jpg")
    yolo.detect_image(image_id,image)
    

print("Conversion completed!")
