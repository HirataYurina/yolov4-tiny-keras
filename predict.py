# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:my_queue.py
# software: PyCharm

from yolo import YOLO
from PIL import Image

yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        result = yolo.detect_image(image)
        result.show()
        result.save('./img/result_kite.jpg')
yolo.close_session()
