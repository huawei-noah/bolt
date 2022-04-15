# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#!/usr/bin/env python

import unittest

import torch
import torch.nn as nn


class PaddingLayerTests(unittest.TestCase):
    def setUp(self):
        self.filling_value = 5.0
        fv = self.filling_value
        self.symmetric_padding_result = torch.tensor(
            [
                [
                    [
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv, fv],
                        [fv, fv, fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv, fv],
                        [fv, fv, fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                    ]
                ]
            ]
        )

        self.asymmetric_padding_result = torch.tensor(
            [
                [
                    [
                        [fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv],
                        [fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv],
                        [fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv],
                    ]
                ]
            ]
        )

        self.asymmetric_padding_for_W_result = torch.tensor(
            [
                [
                    [
                        [fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv],
                        [fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv],
                        [fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv],
                    ]
                ]
            ]
        )

        self.delta1_for_constant_pad = torch.tensor(
            [
                [
                    [
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv, fv],
                        [fv, fv, fv, fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv, fv],
                        [fv, fv, fv, fv, 1.0, 1.0, 1.0, 1.0, 1.0, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                    ]
                ]
            ]
        )

        self.constant_pad_backward_pass_result1 = torch.tensor(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ]
                ]
            ]
        )

        self.delta2_for_constant_pad = torch.tensor(
            [
                [
                    [
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                        [fv, fv, fv, fv, 2.0, 2.0, 2.0, 2.0, 2.0, fv, fv, fv],
                        [fv, fv, fv, fv, 2.0, 2.0, 2.0, 2.0, 2.0, fv, fv, fv],
                        [fv, fv, fv, fv, 2.0, 2.0, 2.0, 2.0, 2.0, fv, fv, fv],
                        [fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv, fv],
                    ]
                ]
            ]
        )

        # self.backward_pass_result2 filled by 3.0 value because of torch.autograd package calculates gradient
        # as sum of all results of backward() function calls
        self.constant_pad_backward_pass_result2 = torch.tensor(
            [
                [
                    [
                        [3.0, 3.0, 3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0, 3.0, 3.0],
                    ]
                ]
            ]
        )

        self.symmetric_reflection_padding_result = torch.tensor(
            [
                [
                    [
                        [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.0, 9.0],
                        [6.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 5.0],
                        [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0],
                        [6.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 5.0],
                        [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.0, 9.0],
                        [6.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 5.0],
                        [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0],
                    ]
                ]
            ]
        )

        self.asymmetric_reflection_padding_result = torch.tensor(
            [
                [
                    [
                        [7.0, 6.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 5.0],
                        [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0],
                        [7.0, 6.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 5.0],
                        [11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.0, 9.0],
                    ]
                ]
            ]
        )

        self.reflection_pad_backward_pass_result = torch.tensor(
            [
                [
                    [
                        [1.0, 3.0, 3.0, 3.0, 2.0],
                        [3.0, 9.0, 9.0, 9.0, 6.0],
                        [2.0, 6.0, 6.0, 6.0, 4.0],
                    ]
                ]
            ]
        )

        self.symmetric_replication_padding_result = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                        [4.0, 4.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0],
                        [8.0, 8.0, 8.0, 9.0, 10.0, 11.0, 11.0, 11.0],
                        [8.0, 8.0, 8.0, 9.0, 10.0, 11.0, 11.0, 11.0],
                        [8.0, 8.0, 8.0, 9.0, 10.0, 11.0, 11.0, 11.0],
                    ]
                ]
            ]
        )

        self.asymmetric_replication_padding_result = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                        [4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0],
                        [8.0, 8.0, 8.0, 8.0, 9.0, 10.0, 11.0, 11.0, 11.0],
                    ]
                ]
            ]
        )

        self.replication_pad_backward_pass_result = torch.tensor(
            [
                [
                    [
                        [15.0, 3.0, 3.0, 3.0, 12.0],
                        [5.0, 1.0, 1.0, 1.0, 4.0],
                        [10.0, 2.0, 2.0, 2.0, 8.0],
                    ]
                ]
            ]
        )

    def test_symmetric_padding_for_each_side_of_H_and_W(self):
        common_padding = 3
        input = torch.ones(1, 1, 3, 5, dtype=torch.float)
        model = nn.ConstantPad2d(common_padding, self.filling_value)
        output = model(input)

        added_from_left = common_padding
        added_from_right = common_padding
        added_from_top = common_padding
        added_from_bottom = common_padding
        self.assertEqual(input.size(), torch.Size([1, 1, 3, 5]))
        self.assertEqual(
            output.size(),
            torch.Size(
                [
                    1,
                    1,
                    3 + added_from_top + added_from_bottom,
                    5 + added_from_left + added_from_right,
                ]
            ),
        )
        self.assertTrue(torch.equal(output, self.symmetric_padding_result))

    def test_asymmetric_padding_for_each_side_of_H_and_W(self):
        paddings = [1, 2, 3, 4]
        input = torch.ones(1, 1, 3, 5, dtype=torch.float)
        model = nn.ConstantPad2d(paddings, self.filling_value)
        output = model(input)

        added_from_left = paddings[0]
        added_from_right = paddings[1]
        added_from_top = paddings[2]
        added_from_bottom = paddings[3]
        self.assertEqual(input.size(), torch.Size([1, 1, 3, 5]))
        self.assertEqual(
            output.size(),
            torch.Size(
                [
                    1,
                    1,
                    3 + added_from_top + added_from_bottom,
                    5 + added_from_left + added_from_right,
                ]
            ),
        )
        self.assertTrue(torch.equal(output, self.asymmetric_padding_result))

    def test_assymmetric_padding_only_for_W(self):
        paddings = [1, 2]
        input = torch.ones(1, 1, 3, 5, dtype=torch.float)
        model = nn.ConstantPad2d(paddings, self.filling_value)
        output = model(input)

        added_from_left = paddings[0]
        added_from_right = paddings[1]
        self.assertEqual(input.size(), torch.Size([1, 1, 3, 5]))
        self.assertEqual(
            output.size(), torch.Size([1, 1, 3, 5 + added_from_left + added_from_right])
        )
        self.assertTrue(torch.equal(output, self.asymmetric_padding_for_W_result))

    def test_setting_no_paddings_for_each_side_of_H_and_W(self):
        no_padding = 0
        input = torch.ones(1, 1, 3, 5, dtype=torch.float)
        model = nn.ConstantPad2d(no_padding, 0.0)
        output = model(input)

        self.assertEqual(input.size(), torch.Size([1, 1, 3, 5]))
        self.assertEqual(output.size(), input.size())
        self.assertTrue(torch.equal(output, input))

        no_paddings = [0, 0, 0, 0]
        model = nn.ConstantPad2d(no_paddings, 0.0)
        output = model(input)

        self.assertEqual(input.size(), torch.Size([1, 1, 3, 5]))
        self.assertEqual(output.size(), input.size())
        self.assertTrue(torch.equal(output, input))

    def test_that_cannot_be_set_different_paddings_only_for_left_right_of_W_and_top_of_H(
        self,
    ):
        input = torch.ones(1, 1, 3, 5, dtype=torch.float)
        model = nn.ConstantPad2d([1, 2, 3], 0.0)
        self.assertRaises(AssertionError, model, input)

    def test_that_cannot_be_set_different_paddings_only_for_left_of_W(self):
        input = torch.ones(1, 1, 3, 5, dtype=torch.float)
        model = nn.ConstantPad2d([1], 0.0)
        self.assertRaises(AssertionError, model, input)

    def test_that_cannot_be_set_more_then_4_paddings(self):
        input = torch.ones(1, 1, 3, 5, dtype=torch.float)
        model = nn.ConstantPad2d([1, 2, 3, 4, 5], 0.0)
        self.assertRaises(AssertionError, model, input)

    def test_backward_computations_of_ConstantPad2d(self):
        # since ConstantPad2d is functional layer (without weights)
        # gradient of ConstantPad2d input is result of calculation
        # of function ConstantPad2d_derivative(backward_input)
        # where ConstantPad2d_derivative calculation result is tensor
        # with the same shape as input and
        # grad[n][c][h][w] = backward_input[n][c][h + padding_top][w + padding_left]
        paddings = [4, 3, 2, 1]
        input = torch.ones(1, 1, 3, 5, dtype=torch.float, requires_grad=True)
        model = nn.ConstantPad2d(paddings, self.filling_value)
        output = model(input)

        output.backward(self.delta1_for_constant_pad)
        self.assertEqual(input.grad.size(), input.size())
        self.assertTrue(
            torch.equal(input.grad, self.constant_pad_backward_pass_result1)
        )

        output.backward(self.delta2_for_constant_pad)
        self.assertEqual(input.grad.size(), input.size())
        self.assertTrue(
            torch.equal(input.grad, self.constant_pad_backward_pass_result2)
        )

    def test_symmetric_reflection_padding(self):
        common_padding = 2
        input = torch.arange(12, dtype=torch.float).reshape(1, 1, 3, 4)
        model = nn.ReflectionPad2d(common_padding)
        output = model(input)

        added_from_left = common_padding
        added_from_right = common_padding
        added_from_top = common_padding
        added_from_bottom = common_padding
        self.assertEqual(input.size(), torch.Size([1, 1, 3, 4]))
        self.assertEqual(
            output.size(),
            torch.Size(
                [
                    1,
                    1,
                    3 + added_from_top + added_from_bottom,
                    4 + added_from_left + added_from_right,
                ]
            ),
        )
        self.assertTrue(torch.equal(output, self.symmetric_reflection_padding_result))

    def test_asymmetric_reflection_padding(self):
        paddings = [3, 2, 1, 0]
        input = torch.arange(12, dtype=torch.float).reshape(1, 1, 3, 4)
        model = nn.ReflectionPad2d(paddings)
        output = model(input)

        added_from_left = paddings[0]
        added_from_right = paddings[1]
        added_from_top = paddings[2]
        added_from_bottom = paddings[3]
        self.assertEqual(input.size(), torch.Size([1, 1, 3, 4]))
        self.assertEqual(
            output.size(),
            torch.Size(
                [
                    1,
                    1,
                    3 + added_from_top + added_from_bottom,
                    4 + added_from_left + added_from_right,
                ]
            ),
        )
        self.assertTrue(torch.equal(output, self.asymmetric_reflection_padding_result))

    def test_no_reflection_padding(self):
        input = torch.arange(12, dtype=torch.float).reshape(1, 1, 3, 4)
        model = nn.ReflectionPad2d(0)
        output = model(input)

        self.assertEqual(input.size(), torch.Size([1, 1, 3, 4]))
        self.assertEqual(output.size(), input.size())
        self.assertTrue(torch.equal(output, input))

    def test_that_reflection_padding_cannot_be_greater_then_or_equal_to_the_corresponding_dimension(
        self,
    ):
        input = torch.arange(12, dtype=torch.float).reshape(1, 1, 3, 4)
        model = nn.ReflectionPad2d(3)
        self.assertRaises(RuntimeError, model, input)

        model = nn.ReflectionPad2d([4, 1, 1, 1])
        self.assertRaises(RuntimeError, model, input)

    def test_backward_computations_of_ReflectionPad2d(self):
        paddings = [4, 3, 2, 1]
        input = torch.arange(1 * 1 * 3 * 5, dtype=torch.float).reshape(1, 1, 3, 5)
        input.requires_grad_(True)
        model = nn.ReflectionPad2d(paddings)
        output = model(input)

        delta = torch.ones(1, 1, 6, 12, dtype=torch.float)
        output.backward(delta)
        self.assertEqual(input.grad.size(), input.size())
        self.assertTrue(
            torch.equal(input.grad, self.reflection_pad_backward_pass_result)
        )

    def test_symmetric_replecation_padding(self):
        common_padding = 2
        input = torch.arange(12, dtype=torch.float).reshape(1, 1, 3, 4)
        model = nn.ReplicationPad2d(common_padding)
        output = model(input)

        added_from_left = common_padding
        added_from_right = common_padding
        added_from_top = common_padding
        added_from_bottom = common_padding
        self.assertEqual(input.size(), torch.Size([1, 1, 3, 4]))
        self.assertEqual(
            output.size(),
            torch.Size(
                [
                    1,
                    1,
                    3 + added_from_top + added_from_bottom,
                    4 + added_from_left + added_from_right,
                ]
            ),
        )
        self.assertTrue(torch.equal(output, self.symmetric_replication_padding_result))

    def test_asymmetric_replication_padding(self):
        paddings = [3, 2, 1, 0]
        input = torch.arange(12, dtype=torch.float).reshape(1, 1, 3, 4)
        model = nn.ReplicationPad2d(paddings)
        output = model(input)

        added_from_left = paddings[0]
        added_from_right = paddings[1]
        added_from_top = paddings[2]
        added_from_bottom = paddings[3]
        self.assertEqual(input.size(), torch.Size([1, 1, 3, 4]))
        self.assertEqual(
            output.size(),
            torch.Size(
                [
                    1,
                    1,
                    3 + added_from_top + added_from_bottom,
                    4 + added_from_left + added_from_right,
                ]
            ),
        )
        self.assertTrue(torch.equal(output, self.asymmetric_replication_padding_result))

    def test_no_replication_padding(self):
        input = torch.arange(12, dtype=torch.float).reshape(1, 1, 3, 4)
        model = nn.ReplicationPad2d(0)
        output = model(input)

        self.assertEqual(input.size(), torch.Size([1, 1, 3, 4]))
        self.assertEqual(output.size(), input.size())
        self.assertTrue(torch.equal(output, input))

    def test_backward_computations_of_ReplicationPad2d(self):
        paddings = [4, 3, 2, 1]
        input = torch.arange(1 * 1 * 3 * 5, dtype=torch.float).reshape(1, 1, 3, 5)
        input.requires_grad_(True)
        model = nn.ReplicationPad2d(paddings)
        output = model(input)

        delta = torch.ones(1, 1, 6, 12, dtype=torch.float)
        output.backward(delta)
        self.assertEqual(input.grad.size(), input.size())
        self.assertTrue(
            torch.equal(input.grad, self.replication_pad_backward_pass_result)
        )


if __name__ == "__main__":
    unittest.main()
