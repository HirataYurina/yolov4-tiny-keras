# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:cosine_anneal.py
# software: PyCharm

from yolo import YOLO
from PIL import Image
from keras.layers import Input
from keras import backend as K
from utils.utils import letterbox_image
from nets.yolo4 import yolo_body, yolo_eval
import colorsys
import numpy as np
import os
from keras.models import load_model

from config.configs import CONFIG


class MapYolo(YOLO):
    """
    detect test data and log results in txt file
    """

    def __init__(self, **kwargs):
        super(MapYolo, self).__init__(**kwargs)
        self.scores = CONFIG.DETECT.SCORE
        self.iou = CONFIG.DETECT.IOU
        self.resolution = CONFIG.DETECT.RESOLUTION

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                   'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # draw bounding boxes
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # random color
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           num_classes, self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image_id, image):

        f = open("./mAP/detection-results/" + image_id + ".txt", "w")

        # use letter box to resize the original img
        new_image_size = self.resulition
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # detect img
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                # learning_phase = 0
                # so, BN uses historical mean and std
                K.learning_phase(): 0
            })

        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            predicted_class = predicted_class.replace(' ', '_')
            score = str(out_scores[i])

            top, left, bottom, right = out_boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6],
                                             str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return


yolo = MapYolo()

image_ids = open('VOCdevkit/VOC2088/ImageSets/Main/test.txt', encoding='utf-8').read().strip().split()

if not os.path.exists("./mAP"):
    os.makedirs("./mAP")
if not os.path.exists("./mAP/detection-results"):
    os.makedirs("./mAP/detection-results")
if not os.path.exists("./mAP/images-optional"):
    os.makedirs("./mAP/images-optional")

for image_id in image_ids:
    image_path = "./VOCdevkit/VOC2088/JPEGImages/" + image_id + ".jpg"
    image = Image.open(image_path)
    image_id = image_id.split('/')[1]
    yolo.detect_image(image_id, image)
    print(image_id, " done!")

print("Conversion completed!")
