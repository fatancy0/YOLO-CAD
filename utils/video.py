#-------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
#-------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

yolo = YOLO()
#-------------------------------------#
#   调用摄像头
# PATH=r"G:\datasheet\无人机数据集\source video\MaybeGood\smallSkyMove.mp4"
PATH=r"F:\视频\scu2.mp4"
capture=cv2.VideoCapture(PATH)
SAVE_video=1
vid = capture
video_frame_cnt = int(vid.get(7))
print(video_frame_cnt,'video_frame_cnt')
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))
# print(args.save_video,'args.save_video')
if SAVE_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))
    print('1........')
#-------------------------------------#
# capture=cv2.VideoCapture(0)
fps = 0.0
i=1
while(True):
    i+=1
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    frame = np.array(yolo.detect_image(frame))
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)
    if SAVE_video:
        videoWriter.write(frame)
        print('write')
    # if i==350:
    #     break
    c= cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        break
print(i)