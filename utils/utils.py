# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:configs.py
# software: PyCharm

from functools import reduce
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from config.configs import CONFIG


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def rand(a=0.0, b=1.0):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line,
                    input_shape,
                    max_boxes=CONFIG.AUG.MAX_BOXES,
                    jitter=.3,
                    hue=.1,
                    sat=1.5,
                    val=1.5):
    """online data augment

    Args:
        annotation_line: 'img_path x1, y1, x2, y2 class'
        input_shape:     [h, w]
        max_boxes:       max boxes that can be trained
        jitter:          random aspect ratio
        hue:             random hue transformation
        sat:             random sat transformation
        val:             random val transformation

    Returns:
        image_data
        box_data

    """
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip:
            box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def print_answer(argmax):
    with open("./model_data/index_word.txt", "r", encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]

    return synset[argmax]


# TODO:test this code
def get_random_mosaic_data(annotations,
                           input_shape,
                           max_boxes=CONFIG.AUG.MAX_BOXES,
                           hue=.1,
                           sat=1.5,
                           val=1.5,
                           jitter=0.3):
    """mosaic augment V1
    mosaic augment V1 can make every image be cropped to be suitable for four regions.
    But this lose variety of image scale because every image need to suitable for their region.
    Args:
        annotations: ['img_path x1, y1, x2, y2 class', 'img_path x1, y1, x2, y2 class', ...]
                     merge four images one time
        input_shape: (416, 416) or others
        max_boxes:   50 or bigger
        hue:         random hue transformation
        sat:         random sat transformation
        val:         random val transformation
        jitter
    Returns:
        augment data with mosaic
    """
    h, w = input_shape
    min_x = 0.4
    min_y = 0.4
    scale_min = 1 - min(min_x, min_y)
    # scale_max = scale_min + 0.2
    scale_max = scale_min + 0.6

    place_x = [0, int(min_x * w), 0, int(min_x * w)]
    place_y = [0, 0, int(min_y * h), int(min_y * h)]

    imgs = []
    boxes_data = []
    index = 0

    for line in annotations:
        contents = line.split()
        img = Image.open(contents[0])
        img = img.convert('RGB')

        iw, ih = img.size

        boxes = np.array([np.array(list(map(int, box.split(',')))) for box in contents[1:]])

        # 1.resize
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(scale_min, scale_max)
        if new_ar < 1:
            new_h = int(scale * h)
            new_w = int(new_h * new_ar)
        else:
            new_w = int(scale * w)
            new_h = int(new_w / new_ar)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # 2.flip
        flip = rand() < 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # don't translate boxes coordination here, because len(boxes) may be equal zero.
            # boxes[..., [0, 2]] = new_w - boxes[..., [0, 2]]

        # 3.hsv transform
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(img) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        img = hsv_to_rgb(x)
        img = Image.fromarray((img * 255).astype(np.uint8))

        # 4.place img
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        dx = place_x[index]
        dy = place_y[index]
        new_img.paste(img, (dx, dy))
        new_img = np.array(new_img) / 255
        imgs.append(new_img)
        index += 1

        # 5.correct box
        if len(boxes) > 0:
            # correct resize
            boxes[..., [0, 2]] = boxes[..., [0, 2]] * new_w / iw
            boxes[..., [1, 3]] = boxes[..., [1, 3]] * new_h / ih
            # correct flip
            if flip:
                boxes[..., [2, 0]] = new_w - boxes[..., [0, 2]]
            # correct place
            boxes[..., [0, 2]] = boxes[..., [0, 2]] + dx
            boxes[..., [1, 3]] = boxes[..., [1, 3]] + dy
            # pick valid boxes
            boxes[..., [0, 1]][boxes[..., [0, 1]] < 0] = 0
            boxes[..., 2][boxes[..., 2] > w] = w
            boxes[..., 3][boxes[..., 3] > h] = h
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_y = boxes[..., 3] - boxes[..., 1]
            boxes = boxes[np.logical_and(boxes_w > 1, boxes_y > 1)]

            boxes_data.append(boxes)
        else:
            # TODO: ######################################################################################
            # TODO: have fixed bug:
            # TODO: if len(boxes) <= 0, boxes not appended into boxes_data,
            # TODO: so, when cropping the boxes, it will encounter errors because i use i in for loop
            # TODO: ######################################################################################
            boxes_data.append([])

    # 6.crop imgs
    cropx = np.random.randint(int(w * min_x), int(w * (1 - min_x)))
    cropy = np.random.randint(int(h * min_y), int(h * (1 - min_y)))
    merge_img = np.zeros((h, w, 3))
    merge_img[:cropy, :cropx, :] = imgs[0][:cropy, :cropx, :]
    merge_img[:cropy, cropx:, :] = imgs[1][:cropy, cropx:, :]
    merge_img[cropy:, :cropx, :] = imgs[2][cropy:, :cropx, :]
    merge_img[cropy:, cropx:, :] = imgs[3][cropy:, cropx:, :]

    boxes = np.zeros(shape=(max_boxes, 5))

    new_boxes = crop_boxes(boxes_data, cropx, cropy)
    num_boxes = len(new_boxes)
    if num_boxes <= max_boxes and num_boxes > 0:
        boxes[0:num_boxes] = new_boxes
    elif num_boxes > max_boxes:
        boxes = new_boxes[:max_boxes]

    return merge_img, boxes


def crop_boxes(boxes, cropx, cropy):
    """crop boxes produced by 'get_random_mosaic_data' function"""

    cropped_boxes = []

    for i, boxes in enumerate(boxes):
        # TODO: ##############################################################
        # TODO: have fixed bug that if boxes=[], there will have some errors
        # TODO: ##############################################################
        if i == 0 and len(boxes) > 0:
            boxes[..., [0, 2]] = np.minimum(cropx, boxes[..., [0, 2]])
            boxes[..., [1, 3]] = np.minimum(cropy, boxes[..., [1, 3]])
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_h = boxes[..., 3] - boxes[..., 1]
            valid_boxes = boxes[np.logical_and(boxes_w >= 5, boxes_h >= 5)]
            cropped_boxes.extend(valid_boxes)

        if i == 1 and len(boxes) > 0:
            boxes[..., [0, 2]] = np.maximum(cropx, boxes[..., [0, 2]])
            boxes[..., [1, 3]] = np.minimum(cropy, boxes[..., [1, 3]])
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_h = boxes[..., 3] - boxes[..., 1]
            valid_boxes = boxes[np.logical_and(boxes_w >= 5, boxes_h >= 5)]
            cropped_boxes.extend(valid_boxes)

        if i == 2 and len(boxes) > 0:
            boxes[..., [0, 2]] = np.minimum(cropx, boxes[..., [0, 2]])
            boxes[..., [1, 3]] = np.maximum(cropy, boxes[..., [1, 3]])
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_h = boxes[..., 3] - boxes[..., 1]
            valid_boxes = boxes[np.logical_and(boxes_w >= 5, boxes_h >= 5)]
            cropped_boxes.extend(valid_boxes)

        if i == 3 and len(boxes) > 0:
            boxes[..., [0, 2]] = np.maximum(cropx, boxes[..., [0, 2]])
            boxes[..., [1, 3]] = np.maximum(cropy, boxes[..., [1, 3]])
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_h = boxes[..., 3] - boxes[..., 1]
            valid_boxes = boxes[np.logical_and(boxes_w >= 5, boxes_h >= 5)]
            cropped_boxes.extend(valid_boxes)

    return np.array(cropped_boxes)

def get_random_mosaic_data_v2(annotations,
                              input_shape,
                              max_boxes=CONFIG.AUG.MAX_BOXES,
                              jitter=0.3,
                              scale_min=0.5,
                              scale_max=2,
                              hue=.1,
                              sat=1.5,
                              val=1.5,
                              min_x=0.3,
                              min_y=0.3):
    """mosaic augment V2
    V2 can keep the variety of size in data augment
    But, V2 may produce some gray region that don't have background or foreground.

    Args:
        annotations: ['img_path x1, y1, x2, y2 class', ...]
        input_shape: h, w
        max_boxes:   a scalar
        jitter:      aspect ratio
        scale_min:   min scale ratio
        scale_max:   max scale ratio
        hue:         a float
        sat:         a float
        val:         a float
        min_x:       random cut ratio of x is random(min_x, 1-min_x)
        min_y:       random cut ratio of y is random(min_y, 1-min_y)

    Returns:
        merge_img
        boxes

    """
    min_x = min_x
    min_y = min_y
    h, w = input_shape

    imgs = []
    boxes_data = []

    for line in annotations:
        line = line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 1.resize image
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(scale_min, scale_max)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 2.generate new image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 3.flip image
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 4.hsv transformation
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

        # 5.correct box
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box_data[:len(box)] = box

        boxes_data.append(box_data)
        imgs.append(image_data)

    # 6.crop image
    merge_img = np.zeros(shape=(h, w, 3))
    cut_x = int(w * rand(a=min_x, b=1 - min_x))
    cut_y = int(h * rand(a=min_y, b=1 - min_y))
    # left-top
    y1 = int(rand(0.0, h - cut_y))
    x1 = int(rand(0.0, h - cut_x))
    merge_img[:cut_y, :cut_x, :] = imgs[0][y1:y1+cut_y, x1:x1+cut_x, :]
    # right-top
    x2 = int(rand(0.0, cut_x))
    y2 = int(rand(0.0, h - cut_y))
    merge_img[:cut_y, cut_x:, :] = imgs[1][y2:y2+cut_y, x2:x2+w-cut_x, :]
    # left-bottom
    x3 = int(rand(0.0, h - cut_x))
    y3 = int(rand(0.0, cut_y))
    merge_img[cut_y:, :cut_x, :] = imgs[2][y3:y3+h-cut_y, x3:x3+cut_x, :]
    # right-bottom
    x4 = int(rand(0.0, cut_x))
    y4 = int(rand(0.0, cut_y))
    merge_img[cut_y:, cut_x:, :] = imgs[3][y4:y4+h-cut_y, x4:x4+w-cut_x]

    # 7.correct boxes
    valid_box = []
    box_data_1 = boxes_data[0]
    if len(box_data_1) > 0:
        point_11 = (0, 0)
        point_12 = (x1, y1)
        point_13 = (x1+cut_x, y1+cut_y)
        valid_box_1 = crop_boxes_v2(box_data_1, point_11, point_12, point_13)
        valid_box.append(valid_box_1)

    box_data_2 = boxes_data[1]
    if len(box_data_2) > 0:
        point_21 = (cut_x, 0)
        point_22 = (x2, y2)
        point_23 = (x2+w-cut_x, y2+cut_y)
        valid_box_2 = crop_boxes_v2(box_data_2, point_21, point_22, point_23)
        valid_box.append(valid_box_2)

    box_data_3 = boxes_data[2]
    if len(box_data_3) > 0:
        point_31 = (0, cut_y)
        point_32 = (x3, y3)
        point_33 = (x3+cut_x, y3+h-cut_y)
        valid_box_3 = crop_boxes_v2(box_data_3, point_31, point_32, point_33)
        valid_box.append(valid_box_3)

    box_data_4 = boxes_data[3]
    if len(box_data_4) > 0:
        point_41 = (cut_x, cut_y)
        point_42 = (x4, y4)
        point_43 = (x4+w-cut_x, y4+h-cut_y)
        valid_box_4 = crop_boxes_v2(box_data_4, point_41, point_42, point_43)
        valid_box.append(valid_box_4)

    valid_box = np.concatenate(valid_box, axis=0)

    boxes = np.zeros(shape=(max_boxes, 5), dtype='float32')
    num_boxes = len(valid_box)
    if num_boxes <= max_boxes:
        boxes[0:num_boxes] = valid_box
    else:
        boxes = valid_box[:max_boxes]

    return merge_img, boxes


def crop_boxes_v2(boxes, point1, point2, point3):
    """crop and shift boxes after boxes merging

    Args:
        boxes: the boxes need to be corrected   [n, 5]
        point1: left-top point of merged image  [2,]
        point2: left-top point of cropped box   [2,]
        point3: right-bottom point of cropped box   [2,]

    Returns:
        valid_boxes: valid boxes that have been cropped and shifted [num_valid, 5]

    """
    point1 = np.array(point1, dtype='float32')
    point2 = np.array(point2, dtype='float32')
    point3 = np.array(point3, dtype='float32')

    boxes_min = boxes[..., :2]
    boxes_max = boxes[..., 2:4]
    crop_min = np.maximum(boxes_min, point2)  # [max_boxes, 2]
    crop_max = np.minimum(boxes_max, point3)
    crop_w = crop_max[..., 0] - crop_min[..., 0]  # [max_boxes,]
    crop_h = crop_max[..., 1] - crop_min[..., 1]  # [max_boxes,]
    valid_mask = np.logical_and(crop_w > 10, crop_h > 10)
    boxes[..., 0:2] = crop_min
    boxes[..., 2:4] = crop_max
    valid_boxes = boxes[valid_mask]

    # shift valid boxes
    if len(valid_boxes) > 0:
        shift_xy = point1 - point2
        valid_boxes[..., 0:2] = valid_boxes[..., 0:2] + shift_xy
        valid_boxes[..., 2:4] = valid_boxes[..., 2:4] + shift_xy

    return valid_boxes


if __name__ == '__main__':
    # test mosaic augment
    with open('../2088_trainval.txt') as f:
        annotations = f.readlines()
        annotations = [anno.strip() for anno in annotations]
        # print(annotations)

    merge_img, boxes = get_random_mosaic_data(annotations[4:8], input_shape=(416, 416))
    merge_img = merge_img

    import cv2

    boxes_wh = boxes[:, 2:4] - boxes[:, 0:2]
    valid_mask = boxes_wh[:, 0] > 0
    valid_boxes = boxes[valid_mask]
    for box in valid_boxes:
        cv2.rectangle(merge_img,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      color=(0, 255, 0),
                      thickness=2)
    cv2.imshow('img', merge_img)
    cv2.waitKey(0)
