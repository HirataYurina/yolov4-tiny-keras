# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:mish.py
# software: PyCharm

import os
import xml.etree.ElementTree as ET

# ----------------------------------------------------
# get ground-truth of test data
# ----------------------------------------------------

image_ids = open('VOCdevkit/VOC2088/ImageSets/Main/test.txt', encoding='utf-8').read().strip().split()
image_ids = [image_id.split('/')[1] for image_id in image_ids]

if not os.path.exists("./mAP"):
    os.makedirs("./mAP")
if not os.path.exists("./mAP/ground-truth"):
    os.makedirs("./mAP/ground-truth")

for image_id in image_ids:
    with open("./mAP/ground-truth/" + image_id + ".txt", "w") as new_f:
        root = ET.parse("VOCdevkit/VOC2088/Annotations/" + image_id + ".xml").getroot()
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            obj_name = obj_name.replace(' ', '_')
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

print("Conversion completed!")
