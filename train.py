# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:configs.py
# software: PyCharm

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from nets.yolo4_tiny import yolo4_tiny
from nets.yolo4_loss import yolo_loss
from keras.backend.tensorflow_backend import set_session
from utils.utils import get_random_data, get_random_mosaic_data, get_random_mosaic_data_v2
from my_queue import GeneratorEnqueuer
import time
import math
from cosine_anneal import WarmUpCosineDecayScheduler
from config.configs import CONFIG


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()

    # use list expression to make your code more concise
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()

    # use list expression to make your code more concise
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def data_generator(annotation_lines,
                   batch_size,
                   input_shape,
                   anchors,
                   num_classes):
    """data generator for fit_generator
    the assignment strategy:
        one gt ---> one anchor
        1.find which anchor(9 anchors) gt belongs to
        2.find which grid gt belongs to

    Args:
        annotation_lines: a list [anno1, anno2, ...]
        batch_size:       batch size
        input_shape:      resolution [h, w]
        anchors:          anchor boxes
        num_classes:      the number of class
        max_boxes:        box_data: [max_boxes, 5]
                          when have a lot of gt to predict, must set max_boxes bigger.

    Returns:
        batch data:       [image_data, *y_true], np.zeros(batch_size)

    """
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                # shuffle dataset at begin of epoch
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        # get true_boxes
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        # use yield to get generator
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_mosaic_iou_thres(annotation_lines,
                                    batch_size,
                                    input_shape,
                                    anchors,
                                    num_classes):
    """data generator for fit_generator
    the assignment strategy:
        one gt ---> more anchor(iou > iou_threshold)

    Args:
        annotation_lines: a list [anno1, anno2, ...]
        batch_size:       batch size
        input_shape:      resolution [h, w]
        anchors:          anchor boxes
        num_classes:      the number of class
        max_boxes:        box_data: [max_boxes, 5]
                          when have a lot of gt to predict, must set max_boxes bigger.
        iou_threshold:    if iou > iou_threshold, the anchor is responsible for this gt.

    Returns:
        batch data:       [image_data, *y_true], np.zeros(batch_size)

    """
    n = len(annotation_lines)
    shuffle_num = n // 4
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                # shuffle dataset at begin of epoch
                np.random.shuffle(annotation_lines)
            image, box = get_random_mosaic_data(annotation_lines[i:i + 4], input_shape)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % shuffle_num
        image_data = np.array(image_data)
        box_data = np.array(box_data)

        y_true = preprocess_true_boxes_iou_thres(box_data, input_shape, anchors, num_classes,
                                                 iou_threshold=CONFIG.TRAIN.IOU_THRESHOLD)
        # use yield to get generator
        yield [image_data, *y_true], np.zeros(batch_size)


def preprocess_true_boxes(true_boxes,
                          input_shape,
                          anchors,
                          num_classes):

    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3

    anchor_mask = [[0, 1, 2], [3, 4, 5]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')  # 416,416

    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    true_boxes[..., 0:2] = boxes_xy / input_shape[:]
    true_boxes[..., 2:4] = boxes_wh / input_shape[:]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 16, 1: 32}[l] for l in range(num_layers)]

    # [(m, 52, 52, 3, 85), (m, 26, 26, 3, 85)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]
    # (1, 9, 2)
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # filter invalid boxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # [n, 1, 2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # get iou
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:

                    # assign gt to one grid
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    # assign gt to one anchor
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    # score = 1 and get one hot class label
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def preprocess_true_boxes_iou_thres(true_boxes,
                                    input_shape,
                                    anchors,
                                    num_classes,
                                    iou_threshold=0.3):
    """get true boxes with iou threshold"""

    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'

    num_layers = len(anchors) // 3
    anchor_mask = [[0, 1, 2], [3, 4, 5]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')  # 416,416

    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    true_boxes[..., 0:2] = boxes_xy / input_shape[:]
    true_boxes[..., 2:4] = boxes_wh / input_shape[:]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 16, 1: 32}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]
    # [1, 9, 2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    # filter invalid boxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # [n, 1, 2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # 1.iou > iou_threshold
        positive = iou > iou_threshold  # [num_true_boxes, num_anchors]
        for t, n in enumerate(positive):
            n = np.array(n, dtype=np.int32)
            pos_index = np.argwhere(n == 1)
            if len(pos_index):
                continue
            for id in pos_index:
                id = id[0]
                for l in range(num_layers):
                    if id in anchor_mask[l]:
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(id)
                        c = true_boxes[b, t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]

                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1

        # 2.if no positive anchor, just choose the best one to be the positive.
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]

                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def get_batch(num_workers,
              max_queue_size=32,
              use_mosaic_iout_generator=CONFIG.DATASET.MOSAIC_AUG,
              multiprocessing=CONFIG.DATASET.MULTIPROCESS,
              **kwargs):
    """

    Args:
        num_workers:               number of workers
        max_queue_size:            max queue size
        multiprocessing:           true in linux and false in windows
        use_mosaic_iout_generator: use mosaic_iou_thres_generator or not
        **kwargs:                  args used in data generator

    """

    enqueuer = None

    try:
        if use_mosaic_iout_generator:
            enqueuer = GeneratorEnqueuer(data_generator_mosaic_iou_thres(**kwargs),
                                         use_multiprocessing=multiprocessing)
        else:
            enqueuer = GeneratorEnqueuer(data_generator(**kwargs),
                                         use_multiprocessing=multiprocessing)
        enqueuer.start(max_queue_size=max_queue_size, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


config = tf.ConfigProto()
# A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

if __name__ == "__main__":

    annotation_path = CONFIG.TRAIN.ANNO_PATH
    valid_anno_path = CONFIG.TRAIN.VALID_PATH

    classes_path = CONFIG.TRAIN.CLASS_PATH
    anchors_path = CONFIG.TRAIN.ANCHOR_PATH
    # pretrained model path
    weights_path = CONFIG.TRAIN.PRE_TRAINED_MODEL

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # checkpoint path
    log_dir = CONFIG.TRAIN.SAVE_PATH
    # resolution
    input_shape = CONFIG.TRAIN.RESOLUTION

    # clear previous tf graph
    K.clear_session()

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape

    # create model
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # TODO: num_anchors // 3 --> num_anchors // 2
    model_body = yolo4_tiny(image_input, num_anchors // 2, num_classes)

    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    y_true = [Input(shape=(h // {0: 16, 1: 32}[l], w // {0: 16, 1: 32}[l],
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    loss_input = [*model_body.output, *y_true]

    # merge custom loss layer into model
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors,
                                   'num_classes': num_classes,
                                   'ignore_thresh': CONFIG.TRAIN.IGNORE_THRES,
                                   'use_focal_confidence_loss': CONFIG.TRAIN.CONFIDENCE_FOCAL,
                                   'use_focal_class_loss': CONFIG.TRAIN.CLASS_FOCAL,
                                   'use_diou': CONFIG.TRAIN.DIOU,
                                   'use_ciou': CONFIG.TRAIN.CIOU,
                                   'print_loss': False})(loss_input)

    # create model_loss
    model = Model([model_body.input, *y_true], model_loss)

    # freeze_layers = 249
    freeze_layers = CONFIG.TRAIN.FREEZE_LAYERS
    for i in range(freeze_layers):
        model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    # checkpoint
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}.h5',
                                 monitor='loss',
                                 save_weights_only=True,
                                 save_best_only=False, period=CONFIG.TRAIN.SAVE_PERIOD)
    # reduce lr on plateau
    # this lr decay is worse than cosine anneal
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
    # i don't use early stopping frequently because it is not orthogonal.
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

    # get training annotations
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    num_train = len(lines)
    # get validating annotations
    with open(valid_anno_path) as f:
        valid_lines = f.readlines()
    np.random.shuffle(valid_lines)
    num_val = len(valid_lines)
    np.random.seed(None)

    # one stage training
    if CONFIG.TRAIN.TRANSFER:
        model.compile(optimizer=Adam(lr=CONFIG.TRAIN.LR_STAGE1),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = CONFIG.TRAIN.BATCH1
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(get_batch(num_workers=CONFIG.DATASET.WORKERS,
                                      max_queue_size=CONFIG.DATASET.MAX_QUEUE,
                                      annotation_lines=lines, batch_size=batch_size,
                                      input_shape=input_shape, anchors=anchors,
                                      num_classes=num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            # validation_data=get_batch(1, annotation_lines=valid_lines, batch_size=batch_size,
                            #                           input_shape=input_shape, anchors=anchors,
                            #                           num_classes=num_classes),
                            # validation steps: at the end of epoch, generate validation_steps * batch data
                            # validation_steps=max(1, num_val // batch_size),
                            epochs=CONFIG.TRAIN.EPOCH1,
                            initial_epoch=0,
                            callbacks=[checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # unfreeze in second stage
    for i in range(freeze_layers):
        model_body.layers[i].trainable = True
    print('layers have been unfrozen!!')

    # training in second stage
    # fine tune
    if True:
        model.compile(optimizer=Adam(lr=CONFIG.TRAIN.LR_STAGE2),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = CONFIG.TRAIN.BATCH2

        # cosine anneal
        total_epoch = CONFIG.TRAIN.EPOCH2 - CONFIG.TRAIN.EPOCH1
        cosine_anneal = WarmUpCosineDecayScheduler(learning_rate_base=CONFIG.TRAIN.LR_STAGE2,
                                                   total_steps=total_epoch * math.ceil(num_train / batch_size),
                                                   interval_epoch=CONFIG.TRAIN.COS_INTERVAL)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(get_batch(num_workers=CONFIG.DATASET.WORKERS,
                                      max_queue_size=CONFIG.DATASET.MAX_QUEUE,
                                      annotation_lines=lines, batch_size=batch_size,
                                      input_shape=input_shape, anchors=anchors,
                                      num_classes=num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            # validation_data=get_batch(annotation_lines=valid_lines, batch_size=batch_size,
                            #                           input_shape=input_shape, anchors=anchors,
                            #                           num_classes=num_classes),
                            # validation_steps=max(1, num_val // batch_size),
                            epochs=CONFIG.TRAIN.EPOCH2,
                            initial_epoch=CONFIG.TRAIN.EPOCH1,
                            callbacks=[checkpoint, cosine_anneal])
        model.save_weights(log_dir + 'last1.h5')
