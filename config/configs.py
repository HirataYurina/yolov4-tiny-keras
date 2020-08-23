# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:configs.py
# software: PyCharm

import easydict

CONFIG = easydict.EasyDict()

# mAP evaluation
CONFIG.DETECT = easydict.EasyDict()

CONFIG.DETECT.SCORE = 0.3
CONFIG.DETECT.IOU = 0.5
CONFIG.DETECT.RESOLUTION = (416, 416)
CONFIG.DETECT.mAP_THRES = 0.5


# prediction
CONFIG.PREDICT = easydict.EasyDict()
CONFIG.PREDICT.WEIGHTS = 'logs/yolo4_tiny_weight.h5'
CONFIG.PREDICT.ANCHOR_PATH = 'model_data/yolo4_tiny_anchors.txt'
CONFIG.PREDICT.CLASS_PATH = 'model_data/coco_classes.txt'
CONFIG.PREDICT.SCORE = 0.2
CONFIG.PREDICT.IOU = 0.3
CONFIG.PREDICT.RESOLUTION = (416, 416)
CONFIG.PREDICT.MAX_BOXES = 40

# train
CONFIG.TRAIN = easydict.EasyDict()

CONFIG.TRAIN.LR_STAGE1 = 0.001
CONFIG.TRAIN.LR_STAGE2 = 0.0001  # it is better to smaller than lr_stage1
# CONFIG.TRAIN.BATCH1 = 32
CONFIG.TRAIN.BATCH1 = 32  # when i use mosaic aug, i am used to make it smaller.
CONFIG.TRAIN.BATCH2 = 4  # it is depending on you GPU memory
CONFIG.TRAIN.EPOCH1 = 50  # it is enough for transfer training in stage 1
CONFIG.TRAIN.EPOCH2 = 250  # fine tuning needs more epochs
CONFIG.TRAIN.TRANSFER = False
CONFIG.TRAIN.IOU_THRESHOLD = 0.3

CONFIG.TRAIN.COS_INTERVAL = [0.05, 0.15, 0.30, 0.50]  # cosine anneal

CONFIG.TRAIN.ANNO_PATH = '2088_trainval.txt'
CONFIG.TRAIN.VALID_PATH = '2088_test.txt'
CONFIG.TRAIN.TEST_PATH = ''
CONFIG.TRAIN.CLASS_PATH = 'model_data/danger_source_classes.txt'
CONFIG.TRAIN.ANCHOR_PATH = 'model_data/yolo4_tiny_anchors_v2.txt'
CONFIG.TRAIN.PRE_TRAINED_MODEL = 'logs/yolo4_tiny_weight.h5'
CONFIG.TRAIN.SAVE_PATH = 'logs/first/'
CONFIG.TRAIN.SAVE_PERIOD = 10

CONFIG.TRAIN.RESOLUTION = (416, 416)
CONFIG.TRAIN.IGNORE_THRES = 0.7  # 0.7 is using in source code
CONFIG.TRAIN.CONFIDENCE_FOCAL = False
CONFIG.TRAIN.CLASS_FOCAL = False
CONFIG.TRAIN.DIOU = False
CONFIG.TRAIN.CIOU = True

# use scale xy to eliminate grid sensitivity
# CONFIG.TRAIN.SCALE_XY = [1.05, 1.1, 1.2]
CONFIG.TRAIN.SCALE_XY = [1.05, 1.05]

# CONFIG.TRAIN.FREEZE_LAYERS = 367
CONFIG.TRAIN.FREEZE_LAYERS = 74

# Augment
CONFIG.AUG = easydict.EasyDict()
CONFIG.AUG.MAX_BOXES = 50

# dataset
CONFIG.DATASET = easydict.EasyDict()
CONFIG.DATASET.MULTIPROCESS = False  # windows can not support multiprocessing in python
CONFIG.DATASET.MOSAIC_AUG = True

CONFIG.DATASET.WORKERS = 1
CONFIG.DATASET.MAX_QUEUE = 32



