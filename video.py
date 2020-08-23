# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:configs.py
# software: PyCharm


from yolo import YOLO
from PIL import Image
import numpy as np
import cv2

'''
    video detection
    the real time detection
'''
yolo = YOLO()

# capture=cv2.VideoCapture('test.mp4')

# get capture
capture = cv2.VideoCapture(0)

while True:
    # get frame
    ref, frame = capture.read()
    # BGR2RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # array2Image
    frame = Image.fromarray(np.uint8(frame))

    # start detecting
    frame = np.array(yolo.detect_image(frame))
    # RGBtoBGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('video', frame)
    # if press esc, stop.
    if (cv2.waitKey(35) & 0xff) == 27:
        capture.release()
        break

yolo.close_session()
