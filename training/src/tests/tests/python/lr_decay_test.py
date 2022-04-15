# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import tensorflow as tf

tf.compat.v1.enable_eager_execution()


def _learning_rate_decay(global_step):
    tacotron_initial_learning_rate = 1e-2
    tacotron_start_decay = 200
    decay_steps = 1000
    decay_rate = 0.5
    tacotron_final_learning_rate = 1e-4
    warmup_enable = True
    #################################################################
    # Narrow Exponential Decay:

    # Phase 1: lr = 1e-3
    # We only start learning rate decay after 50k steps

    # Phase 2: lr in ]1e-5, 1e-3[
    # decay reach minimal value at step 310k

    # Phase 3: lr = 1e-5
    # clip by minimal learning rate value (step > 310k)
    #################################################################

    # Compute natural exponential decay
    lr = tf.train.exponential_decay(
        tacotron_initial_learning_rate,
        global_step - tacotron_start_decay,  # lr = 1e-3 at step 50k
        decay_steps,
        decay_rate,  # lr = 1e-5 around step 310k
        name="lr_exponential_decay",
    )

    # clip learning rate by max and min values (initial and final values)
    lr = tf.minimum(
        tf.maximum(lr, tacotron_final_learning_rate), tacotron_initial_learning_rate
    )

    if self.warmup_enable:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(hp.warmup_num_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = tacotron_initial_learning_rate * warmup_percent_done

        lr = tf.minimum(lr, warmup_learning_rate)
    return lr


steps = [
    0,
    10,
    20,
    100,
    1000,
    2000,
    10000,
    100000,
    300000,
    300001,
    300010,
    300100,
    301000,
]
for step in steps:
    print(
        "step =", step, "lr =", _learning_rate_decay(tf.Variable(step, dtype=tf.int32))
    )
