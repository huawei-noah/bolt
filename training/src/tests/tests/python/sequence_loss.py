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

import tensorflow_addons as tfa
import tensorflow as tf

tf.random.set_seed(1)

logits = tf.Variable(
    tf.random.uniform(
        [2, 3, 4], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="logits",
)
targets = tf.Variable(
    tf.random.uniform(
        [2, 3, 4], minval=0, maxval=3, dtype=tf.dtypes.int64, seed=0, name=None
    ),
    name="targets",
)
weights = tf.Variable(
    tf.random.uniform(
        [2, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=0, name=None
    ),
    name="weights",
)

print("logits: ", logits)
print("targets: ", targets)
print("weights: ", weights)

# No reduction
print(tfa.seq2seq.sequence_loss(logits, targets, weights, False, False, False, False))
# All sum possibilities
print(tfa.seq2seq.sequence_loss(logits, targets, weights, False, False, True, False))
print(tfa.seq2seq.sequence_loss(logits, targets, weights, False, False, False, True))
print(tfa.seq2seq.sequence_loss(logits, targets, weights, False, False, True, True))
# All average possibilities
print(tfa.seq2seq.sequence_loss(logits, targets, weights, True, False, False, False))
print(tfa.seq2seq.sequence_loss(logits, targets, weights, False, True, False, False))
print(tfa.seq2seq.sequence_loss(logits, targets, weights, True, True, False, False))
