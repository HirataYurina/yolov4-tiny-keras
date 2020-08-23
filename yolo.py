import os
import numpy as np
import copy
import colorsys
from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from nets.yolo4 import yolo_eval
from nets.yolo4_tiny import yolo4_tiny
from utils.utils import letterbox_image
from config.configs import CONFIG
import tensorflow as tf

# remember add this code to avoid some bugs
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)


class YOLO(object):

    _defaults = {
        "model_path": CONFIG.PREDICT.WEIGHTS,
        "anchors_path": CONFIG.PREDICT.ANCHOR_PATH,
        "classes_path": CONFIG.PREDICT.CLASS_PATH,
        "score": CONFIG.PREDICT.SCORE,
        "iou": CONFIG.PREDICT.IOU,
        "model_image_size": CONFIG.PREDICT.RESOLUTION,
        "max_boxes": CONFIG.PREDICT.MAX_BOXES
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        # update dict of YOLO class
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            # TODO: num_anchors // 3 --> num_anchors // 2
            self.yolo_model = yolo4_tiny(Input(shape=(None, None, 3)), num_anchors // 2, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                   'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # draw rectangles
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                               self.colors))

        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))

        boxes, scores, classes = yolo_eval(self.yolo_model.output,
                                           self.anchors,
                                           num_classes,
                                           self.input_image_shape,
                                           max_boxes=self.max_boxes,
                                           score_threshold=self.score,
                                           iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):

        start = timer()

        # convert img_size to input_size
        new_image_size = (self.model_image_size[0], self.model_image_size[1])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        # sess.run
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # print(out_scores)
        # print(out_boxes)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # starting draw bounding boxes
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness of bounding box and this thickness is changing according to img_size
        thickness = (image.size[0] + image.size[1]) // 500

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i],
                               outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                           fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print('detect time:', end - start)
        return image

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':

    yolo = YOLO()

    image = Image.open('./img/kite.jpg')
    img = yolo.detect_image(image)
    img.show()
