# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:cosine_anneal.py
# software: PyCharm

import numpy as np
import keras as keras
import keras.backend as K


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,
                             min_learn_rate=0,
                             ):
    """SGDR: Stochastic Gradient Descent with Warm Restarts
    https://arxiv.org/abs/1608.03983v5

    Args:
        global_step:          current steps
        learning_rate_base:   the base learning rate
                              In warming up, learning rate increases from [warmup_learning_rate] to [learning_rate_base]
        total_steps:          epoch * num_train_examples / batch_size
        warmup_learning_rate: initial learning rate in warming up period
        warmup_steps:         if warmup steps > 0, start warmup when training model
        hold_base_rate_steps: the steps that hold learning_rate_base after warming up
        min_learn_rate:       the minimum learning rate

    Returns:
        learning_rate:        learning_rate = max(learning_rate, min_learn_rate)

    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')

    # cosine anneal
    # minimum lr in this equation is 0
    # TODO: if you don't want the minimum lr to be zero, you can change this equation.
    learning_rate = 0.5 * learning_rate_base * \
        (1 + np.cos(np.pi * (global_step - warmup_steps - hold_base_rate_steps)
                    / float(total_steps - warmup_steps - hold_base_rate_steps)))

    # hold base rate when warmup_steps < global_step < warmup_steps + hold_base_rate_step
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        # linear increase
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate

        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)

    learning_rate = max(learning_rate, min_learn_rate)
    return learning_rate


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 min_learn_rate=0,
                 # restart at interval_epoch
                 interval_epoch=None,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()

        if interval_epoch is None:
            interval_epoch = [0.05, 0.15, 0.30, 0.50]
        self.learning_rate_base = learning_rate_base
        self.warmup_learning_rate = warmup_learning_rate

        # 0 = silent
        self.verbose = verbose
        self.min_learn_rate = min_learn_rate
        # store learning rate
        self.learning_rates = []

        self.interval_epoch = interval_epoch
        # the initial step that is useful when  you continue training model
        self.global_step_for_interval = global_step_init
        self.warmup_steps_for_interval = warmup_steps
        self.hold_steps_for_interval = hold_base_rate_steps
        self.total_steps_for_interval = total_steps

        self.interval_index = 0
        # total steps in every interval epoch
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch) - 1):
            self.interval_reset.append(self.interval_epoch[i + 1] - self.interval_epoch[i])
        self.interval_reset.append(1 - self.interval_epoch[-1])

    # on_batch_end
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        self.global_step_for_interval = self.global_step_for_interval + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    # on_batch_begin
    def on_batch_begin(self, batch, logs=None):

        # restart lr
        if self.global_step_for_interval in [0] + [int(i * self.total_steps_for_interval) for i in self.interval_epoch]:
            self.total_steps = self.total_steps_for_interval * self.interval_reset[self.interval_index]
            self.warmup_steps = self.warmup_steps_for_interval * self.interval_reset[self.interval_index]
            self.hold_base_rate_steps = self.hold_steps_for_interval * self.interval_reset[self.interval_index]
            self.global_step = 0
            self.interval_index += 1

        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps,
                                      min_learn_rate=self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))
