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

tf.random.set_seed(0)
opt = tf.keras.optimizers.RMSprop(
    learning_rate=0.1,
    rho=0.9,
    momentum=0.1,
    epsilon=0.1,
    centered=False,
)
var1 = tf.Variable(tf.random.normal([1, 2, 3, 4]))
loss = lambda: (var1 ** 2) / 2.0  # grad = var1
print("Parameter: ", var1)
step_count = opt.minimize(loss, [var1])
print("Parameter (after first step): ", var1)
step_count = opt.minimize(loss, [var1])
print("Parameter (after second step): ", var1)

opt = tf.keras.optimizers.RMSprop(
    learning_rate=0.1,
    rho=0.9,
    momentum=0.1,
    epsilon=0.1,
    centered=True,
)
print("Parameter: ", var1)
step_count = opt.minimize(loss, [var1])
print("Parameter (after first step): ", var1)
step_count = opt.minimize(loss, [var1])
print("Parameter (after second step): ", var1)
