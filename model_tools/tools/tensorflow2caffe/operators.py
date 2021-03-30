#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import math

class Operators:
    calculate = False;

    @staticmethod
    def set_calculate(flag):
        Operators.calculate = flag
    
    @staticmethod
    def print_data(x, name, print_flag=True):
        if (not print_flag):
            return None
        shape_str = "("
        for i in x.shape:
            shape_str = shape_str + str(i) + ","
        shape_str = shape_str[0:-1] + ")"
        print("[INFO] tensor name %s shape %s sum %f" % (name, shape_str, x.sum()))
        num = x.size
        x = np.reshape(x, [num])
        threshold = 60
        if (num < threshold):
            threshold = num
        print(x[:threshold])

    @staticmethod
    def zeros(shape, name):
        x = np.zeros(shape)
        if (Operators.calculate):
            Operators.print_data(x, name)
        return x

    @staticmethod
    def weight(x, name):
        if (Operators.calculate):
            Operators.print_data(x, name)
        return x

    @staticmethod
    def nchwc8_nchw(_x, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        assert (len(x.shape) == 5)
        x = x.transpose([0, 2, 3, 1, 4])
        input_shape = x.shape
        shape = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]*input_shape[4]]
        x = x.reshape(shape)
        x = x.transpose([0, 3, 1, 2])
        Operators.print_data(x, name)
        return x

    @staticmethod
    def batch_norm(_x, mean, var, eps, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        if (len(_x.shape) == 3):
            x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
        y = np.zeros([x.shape[0], x.shape[1]//8, x.shape[2], x.shape[3], 8])
        for n in range(y.shape[0]):
            for c in range(y.shape[1]):
                for h in range(y.shape[2]):
                    for w in range(y.shape[3]):
                        for i in range(y.shape[4]):
                            y[n][c][h][w][i] = (x[n][c*8+i][h][w] - mean[c*8+i]) / math.sqrt(eps+var[c*8+i])
        Operators.print_data(y, name)
        return y

    @staticmethod
    def scale(_x, gamma, beta, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        if (len(_x.shape) == 3):
            x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
        y = np.zeros([x.shape[0], x.shape[1]//8, x.shape[2], x.shape[3], 8])
        eps = 1e-6
        for n in range(y.shape[0]):
            for c in range(y.shape[1]):
                for h in range(y.shape[2]):
                    for w in range(y.shape[3]):
                        for i in range(y.shape[4]):
                            if beta is None:
                                y[n][c][h][w][i] = x[n][c*8+i][h][w] * gamma[c*8+i]
                            else:
                                y[n][c][h][w][i] = x[n][c*8+i][h][w] * gamma[c*8+i] + beta[c*8+i]
        Operators.print_data(y, name)
        return y

    @staticmethod
    def group_norm(_x, groups, gamma, beta, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        if (len(_x.shape) == 3):
            x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
        y = np.zeros([x.shape[0], x.shape[1]//8, x.shape[2], x.shape[3], 8])
        mean = [0] * groups
        var = [0] * groups
        for n in range(x.shape[0]):
            for c in range(x.shape[1]):
                value = 0
                for h in range(x.shape[2]):
                    for w in range(x.shape[3]):
                        value = value + x[n][c][h][w]
                mean[c % groups] = mean[c % groups] + value
        for n in range(x.shape[0]):
            for c in range(x.shape[1]):
                value = 0
                for h in range(x.shape[2]):
                    for w in range(x.shape[3]):
                        v = x[n][c][h][w] - mean[c % groups]
                        value = value + v * v
                var[c % groups] = var[c % groups] + value
        for n in range(y.shape[0]):
            for c in range(y.shape[1]):
                for h in range(y.shape[2]):
                    for w in range(y.shape[3]):
                        for i in range(y.shape[4]):
                            y[n][c][h][w][i] = (x[n][c*8+i][h][w] - mean[(c*8+i) % groups]) / math.sqrt(eps+var[(c*8+i) % groups]) \
                                * gamma[c*8+i] + beta[c*8+i]
        Operators.print_data(y, name)
        return y

    @staticmethod
    def l2_norm(_x, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        if (len(x.shape) == 2):
            for i in range(x.shape[0]):
                tmp = 0
                for j in range(x.shape[1]):
                    tmp += x[i][j] * x[i][j]
                tmp = tmp ** 0.5
                for j in range(x.shape[1]):
                    x[i][j] = x[i][j]/tmp

        Operators.print_data(x, name)
        return x

    @staticmethod
    def convolution(x, kernels, bias,
                    num_output, kernel_size, strides, paddings,
                    dilation, groups, name):
        if (not Operators.calculate):
            return None
        assert len(kernel_size) == 2
        assert len(strides) == 2
        assert len(paddings) == 4
        _x_shape = x.shape
        if (len(_x_shape) == 3):
            x_shape = [1, _x_shape[0], _x_shape[1], _x_shape[2]]
        else:
            x_shape = _x_shape
        fh_d = (kernel_size[0] - 1) * dilation + 1;
        fw_d = (kernel_size[1] - 1) * dilation + 1;
        h = (x_shape[2] + paddings[0] + paddings[1] - fh_d) // strides[0] + 1
        w = (x_shape[3] + paddings[2] + paddings[3] - fw_d) // strides[1] + 1
        x = x.reshape(x_shape)

        y = np.zeros([x_shape[0], num_output, h, w])
        x_pad = np.pad(x, [[0, 0], [0, 0], [paddings[0], paddings[1]], [paddings[2], paddings[3]]], mode='constant')
        for f in range(y.shape[1]):
            for i in range(y.shape[2]):
                for j in range(y.shape[3]):
                    y[:, f, i, j] = np.sum(x_pad[:, :, i*strides[0]:i*strides[0]+fh_d, j*strides[1]:j*strides[1]+fw_d] * kernels[f, :, :, :], axis=(1, 2, 3))
            if bias is not None:
                y[:, f, :, :] += bias[f]
        y = y.reshape(x_shape[0], num_output//8, 8, h, w)
        y = y.transpose(0, 1, 3, 4, 2)
        Operators.print_data(y, name)
        return y

    @staticmethod
    def layer_norm(_x, s, b, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        x_shape = x.shape
        if (len(_x.shape) == 2):
            x = x.reshape([1, x.shape[0], x.shape[1]])
        if (len(_x.shape) == 4):
            x = x.reshape([x.shape[0], -1, x.shape[3]])
        eps = 1e-12
        for n in range(len(x)):
            for i in range(len(x[0])):
                sum = 0
                for j in range(len(x[0][0])):
                    sum = sum + x[n][i][j]
                mean = sum / len(x[0][0])
                var = 0
                for j in range(len(x[0][0])):
                    v = x[n][i][j] - mean
                    var = var + v * v
                var = var / len(x[0][0])
                for j in range(len(x[0][0])):
                    x[n][i][j] = (x[n][i][j] - mean) / math.sqrt(var + eps)
                    x[n][i][j] = s[j] * x[n][i][j] + b[j]
        x = x.reshape(x_shape)
        Operators.print_data(x, name)
        return x

    @staticmethod
    def matmul(_a, transpose_a, _b, transpose_b, name):
        if (not Operators.calculate):
            return None;
        a = _a.copy()
        b = _b.copy()
        if (transpose_a):
            num = len(a.shape)
            shape = [i for i in range(num)]
            tmp = shape[-1]
            shape[-1] = shape[-2]
            shape[-2] = tmp
            a = a.transpose(shape)
        if (transpose_b):
            num = len(b.shape)
            shape = [i for i in range(num)]
            tmp = shape[-1]
            shape[-1] = shape[-2]
            shape[-2] = tmp
            b = b.transpose(shape)
        x = np.matmul(a, b)
        Operators.print_data(x, name)
        return x

    @staticmethod
    def fully_connect(_x, w, b, name):
        if (not Operators.calculate):
            return None;
        x = np.matmul(_x, w)
        if (b is not None):
            x = x + b
        Operators.print_data(x, name)
        return x
 
    @staticmethod
    def reshape(_x, dim, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        x = np.reshape(x, dim)
        Operators.print_data(x, name)
        return x
 
    @staticmethod
    def transpose(_x, dim, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        x = np.transpose(x, dim)
        Operators.print_data(x, name)
        return x
    
    @staticmethod
    def power(_x, scale, shift, power, name):
        if (not Operators.calculate):
            return None;
        x = _x * scale
        x = x + shift
        x = np.power(x, power)
        Operators.print_data(x, name)
        return x
    
    @staticmethod
    def matmultiply(_x, _y, name):
        if (not Operators.calculate):
            return None;
        x = np.multiply(_x, _y)
        Operators.print_data(x, name)
        return x

    @staticmethod
    def divide(_x, _y, name):
        if (not Operators.calculate):
            return None;
        x = np.divide(_x, _y)
        Operators.print_data(x, name)
        return x

    @staticmethod
    def tile(_x, loops, axis, name):
        if (not Operators.calculate):
            return None
        input = _x
        input = np.array(input, dtype=float)
        input_shape_list = list(input.shape)
        length = loops
        for i in input_shape_list:
            length *= i
        temp_shape_list = input_shape_list
        temp_shape_list[axis] = input_shape_list[axis] * loops
        temp_list = []
        if axis == -1 and len(input_shape_list) == 3:
            for i in range(input_shape_list[0]):
                for j in range(input_shape_list[1]):
                    for n in range(loops):
                        temp_list.append(input[i][j][:])
        if axis == -1 and len(input_shape_list) == 2:
            for i in range(input_shape_list[0]):
                for n in range(loops):
                    temp_list.append(input[i][:])
        x = Operators.reshape(temp_list, tuple(temp_shape_list), name)
        return x

    @staticmethod
    def slice(x, axis, slice_points, names, print_flag=True):
        if (not Operators.calculate):
            return None;
        if (len(x) != 1):
            print("[ERROR] batch != 1")
            exit(0)
        result = []
        for i in range(len(slice_points)+1):
            if (i == 0):
                start = 0
            else:
                start = slice_points[i - 1]
            if (i != len(slice_points)):
                end = slice_points[i]
                if (axis == 0):
                    result.append(x[start:end, :])
                elif (axis == 1):
                    result.append(x[:, start:end])
                elif (axis == 2):
                    result.append(x[:, :, start:end])
                elif (axis == 3):
                    result.append(x[:, :, :, start:end,:])
                elif (axis == 4):
                    result.append(x[:, :, :, :, start:end,:])
                else:
                    print("[ERROR] unsopprted slice axis %d" % (axis))
                    exit(1)
            else:
                if (axis == 0):
                    result.append(x[start:, :])
                elif (axis == 1):
                    result.append(x[:, start:])
                elif (axis == 2):
                    result.append(x[:, :, start:])
                elif (axis == 3):
                    result.append(x[:, :, :, start:,:])
                elif (axis == 4):
                    result.append(x[:, :, :, :, start:,:])
                else:
                    print("[ERROR] unsopprted slice axis %d" % (axis))
                    exit(1)
            Operators.print_data(result[i], names[i], print_flag)
        return result

    @staticmethod
    def sum(inputs, name):
        if (not Operators.calculate):
            return None;

        x = inputs[0].copy()
        for i in range(1, len(inputs)):
            x = x + inputs[i]
        Operators.print_data(x, name)
        return x

    @staticmethod
    def attention_mask(_x, attn_trunc_len, same_length, mask_value, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        x_shape = x.shape
        qlen = x_shape[-2]
        mlen = x_shape[-1] - qlen
        if (attn_trunc_len > 0):
            mask = np.ones([qlen, qlen+mlen])
            for i in range(qlen):
                end = mlen + i
                start = max(end - attn_trunc_len, 0)
                loops = end - start + 1
                for j in range(loops):
                    mask[i][start+j] = 0
        elif (attn_trunc_len == 0):
            mask = np.ones([qlen, qlen+mlen])
            for i in range(qlen):
                if (same_length):
                    start = i
                    loops = qlen + 1
                else:
                    start = 0
                    loops = i + qlen + 1
                for j in range(loops):
                    mask[i][start+j] = 0
        else:
            mask = np.zeros([qlen, qlen+mlen])
        x = x.reshape([-1, qlen, qlen+mlen])
        for i in range(x.shape[0]):
            for j in range(qlen):
                for k in range(qlen+mlen):
                    x[i][j][k] = x[i][j][k] * (1 - mask[j][k]) - mask_value * mask[j][k]
        x = x.reshape(x_shape)
        Operators.print_data(x, name)
        return x

    @staticmethod
    def attention(_x, num_attention, from_seq_length, to_seq_length, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        #x = (1 - x) * np.finfo(np.float16).min
        #x = np.expand_dims(x, axis = 0)
        #x = np.repeat(x, num_attention, axis = 0)
        #x = np.expand_dims(x, axis = 0)

        if (len(x) != 1):
            print("[ERROR] batch != 1")
            exit(0)
        x = np.reshape(x, [len(x[0])])
        count = 0
        for i in x:
            count = count + i
        xx = []
        for j in range(min(count,from_seq_length)):
            xx.append(x)
        for j in range(min(count,from_seq_length), from_seq_length):
            xx.append([0]*to_seq_length)

        mask = -10000
        xx = (1 - np.array(xx)) * mask
        xxx = []
        for j in range(num_attention):
            xxx.append(xx)
        xxx = np.array(xxx)
        xxx = np.reshape(xxx, [1, num_attention, from_seq_length, to_seq_length])
        x = xxx
        Operators.print_data(x, name)
        return x

    @staticmethod
    def embedding(_x, _w, transpose, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        if (len(x) != 1):
            print("[ERROR] batch != 1")
            exit(0)
        x = np.reshape(x, [len(x[0])])
        if (len(_w.shape) == 2):
            w = _w
        elif (len(_w.shape) == 3):
            w = _w.reshape([-1, _w.shape[2]])
        else:
            print("[ERROR] can not support more dimension embedding")
            exit(0)
        y = []
        for i in x:
            index = int(i)
            if (transpose):
                y.append(w[:,index])
            else:
                y.append(w[index])
        y = np.array(y)
        if (transpose):
            x = np.reshape(y, [1, len(x), w.shape[-2]])
        else:
            x = np.reshape(y, [1, len(x), w.shape[-1]])
        Operators.print_data(x, name)
        return x;
    
    @staticmethod
    def softmax(_x, axis, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        x_row_max = x.max(axis=axis)
        x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
        x = x - x_row_max
        x_exp = np.exp(x)
        x_exp_row_sum = x_exp.sum(axis=axis).reshape(list(x.shape)[:-1]+[1])
        x = x_exp / x_exp_row_sum
        Operators.print_data(x, name)
        return x
    
    @staticmethod
    def gelu(_x, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        cdf = 0.5 * (1.0 + np.tanh((0.7978845608028654 * (x + 0.044715 * x * x * x))))
        x = x * cdf
        Operators.print_data(x, name)
        return x
    
    @staticmethod
    def relu(_x, name, max_value=float('inf')):
        if (not Operators.calculate):
            return None;
        x = np.maximum(_x, 0, _x)
        x = np.minimum(_x, max_value, _x)
        #x[x < 0] = 0
        Operators.print_data(x, name)
        return x

    @staticmethod
    def exp(_x, base, scale, shift, name):
        if (not Operators.calculate):
            return None;

        assert(base == -1)
        x = np.exp(scale * _x + shift)
        Operators.print_data(x, name)
        return x

    @staticmethod
    def softplus(_x, name):
        if (not Operators.calculate):
            return None;

        x = np.log(1 + np.exp(_x))
        Operators.print_data(x, name)
        return x

    @staticmethod
    def mish(_x, name):
        if (not Operators.calculate):
            return None;
        x = _x * np.tanh(np.log(1 + np.exp(_x)))
        Operators.print_data(x, name)
        return x

    @staticmethod
    def _sigmoid(_x):
        x = 1.0 / (1.0 + np.exp(-_x))
        return x

    @staticmethod
    def sigmoid(_x, name):
        if (not Operators.calculate):
            return None;

        x = Operators._sigmoid(_x)
        Operators.print_data(x, name)
        return x

    @staticmethod
    def tanh(_x, name):
        if (not Operators.calculate):
            return None;
        x = np.tanh(_x)
        Operators.print_data(x, name)
        return x

    @staticmethod
    def swish(_x, beta, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        x = x * Operator.sigmoid(beta * x)
        Operators.print_data(x, name)
        return x
    
    @staticmethod
    def squeeze(_x, axis, name):
        if (not Operators.calculate):
            return None;
        x = np.squeeze(_x, axis)
        Operators.print_data(x, name)
        return x
    
    @staticmethod
    def expand_dims(input, axis, name):
        if (not Operators.calculate):
            return None;
        x = np.expand_dims(input, axis)
        Operators.print_data(x, name)
        return x
    
    @staticmethod
    def reduction(input, mask, operation, axis, name):
        if (not Operators.calculate):
            return None;
        if (mask is not None):
            if (axis < 0):
                axis = axis + len(input.shape)
            left = 1
            for i in range(axis):
                left = left * input.shape[i]
            _input = input.reshape([left, input.shape[axis], -1])
            mask = mask.reshape([-1, mask.shape[-1]])
            s = []
            for k in range(left):
                ss = []
                for i in range(mask.shape[0]):
                    sss = []
                    for j in range(mask.shape[1]):
                        if (mask[i][j] == 1):
                            sss.append(_input[k, j, :])
                    ss.append(sss)
                s.append(ss)
            s = np.array(s)
            _axis = 2
        else:
            s = input
            _axis = axis
        if (operation == 4):
            x = np.mean(s, _axis)
        elif (operation == 1):
            x = np.sum(s, _axis)
        else:
            print("[ERROR] unsupported reduction operation %s" % (operation))
            exit(1)
        if (mask is not None):
            shape = [i for i in input.shape]
            shape[axis] = mask.shape[0]
            x = x.reshape(shape)
        Operators.print_data(x, name)
        return x
    
    @staticmethod
    def lstm(inputs, state, w, b, projection, projection_bias, zoneout_cell, zoneout_output,
        name, state_name, printFlag=True):
        if (not Operators.calculate):
            return None, None
        w = w.transpose([1, 0])
        inputs = np.reshape(inputs, [1, inputs.shape[-1]])
        c_size = w.shape[1] // 4
        #d, h = np.split(state, indices_or_sections = 2, axis = 1)
        #c = np.zeros([1, 2048])
        c, h = np.split(state, (c_size,), axis = 1)
        x = np.concatenate([inputs, h], axis = 1)
        x = np.matmul(x, w)
        if (b is not None):
            x = x + b

        i, j, f, o = np.split(x, indices_or_sections = 4, axis = 1)
        new_c = np.multiply(c, Operators._sigmoid(np.add(f, 1.0))) \
                + np.multiply(Operators._sigmoid(i), np.tanh(j))
        new_h = np.multiply(np.tanh(new_c), Operators._sigmoid(o))
        if (projection is not None):
            projection = projection.transpose([1, 0])
            new_h = np.matmul(new_h, projection)
            if (projection_bias is not None):
                new_h = new_h + projection_bias
        o_c = new_c * (1 - zoneout_cell) + c * zoneout_cell
        o_h = new_h * (1 - zoneout_output) + h * zoneout_output
        new_state = np.concatenate([o_c, o_h], axis = 1)

        if (printFlag):
            Operators.print_data(new_h, name)
            Operators.print_data(new_state, state_name)
        return new_h, new_state

    @staticmethod
    def gru(inputs, state, w, b, name, state_name, printFlag=True):
        if (not Operators.calculate):
            return None, None
        w = w.transpose([1, 0])
        inputs = np.reshape(inputs, [1, inputs.shape[-1]])
        zr_length = w.shape[-1] // 3 * 2
        w_zr = w[:, :zr_length]
        w_h = w[:, zr_length:]
        h = state
        x = np.concatenate([inputs, h], axis = 1)
        x = np.matmul(x, w_zr)
        if (b is not None):
            b_zr = b[:zr_length]
            x = x + b_zr

        z, r = np.split(Operators._sigmoid(x), indices_or_sections = 2, axis = 1)
        h = np.multiply(r, h)
        x = np.concatenate([inputs, h], axis = 1)
        x = np.matmul(x, w_h)
        if (b is not None):
            b_h = b[zr_length:]
            x = x + b_h
        x = np.tanh(x)
        new_h = np.multiply(z, state) + np.multiply((1 - z), x)
        new_state = new_h

        if (printFlag):
            Operators.print_data(new_h, name)
            Operators.print_data(new_state, state_name)
        return new_h, new_state

    @staticmethod
    def gru_lbr(inputs, state, w, b, name, state_name, printFlag=True):
        if (not Operators.calculate):
            return None, None
        w = w.transpose([1, 0])
        x_dim = inputs.shape[-1]
        inputs = np.reshape(inputs, [1, x_dim])
        hidden = w.shape[-1] // 3
        zr_length = hidden * 2
        w_zr = w[:, :zr_length]
        w_h = w[:, zr_length:]
        h = state
        x = np.concatenate([inputs, h], axis = 1)
        x = np.matmul(x, w_zr)
        if (b is not None):
            b_zr = b[:zr_length]
            x = x + b_zr

        z, r = np.split(Operators._sigmoid(x), indices_or_sections = 2, axis = 1)
        w_h_x = w_h[:x_dim, :]
        w_h_h = w_h[x_dim:, :]
        x = np.matmul(inputs, w_h_x)
        h = np.matmul(state, w_h_h)
        if (b is not None):
            b_h_x = b[zr_length:zr_length+hidden]
            x = x + b_h_x
            b_h_h = b[zr_length+hidden:]
            if (b_h_h.size > 0):
                h = h + b_h_h
        x = np.multiply(r, h) + x
        x = np.tanh(x)
        new_h = np.multiply(z, state) + np.multiply((1 - z), x)
        new_state = new_h

        if (printFlag):
            Operators.print_data(new_h, name)
            Operators.print_data(new_state, state_name)
        return new_h, new_state

    @staticmethod
    def rnn(mode, inputs, state, w, b, projection, projection_bias, zoneout_cell, zoneout_output,
        name, state_name, printFlag=True):
        if (mode == "LSTM"):
            result, state = Operators.lstm(inputs, state, w, b, projection, projection_bias,
                zoneout_cell, zoneout_output, None, None, False)
        elif (mode == "GRU"):
            result, state = Operators.gru(inputs, state, w, b, name, state_name, printFlag)
        elif (mode == "GRU_LBR"):
            result, state = Operators.gru_lbr(inputs, state, w, b, name, state_name, printFlag)
        else:
            print("[ERROR] RNN can not support %s" % (mode))
            exit(1)
        return result, state

    @staticmethod
    def fw_rnn(mode, inputs, w, b, projection, projection_bias, zoneout_cell, zoneout_output, name):
        if (not Operators.calculate):
            return None
        inputs = np.reshape(inputs, [-1, inputs.shape[-1]])
        if (mode == "LSTM"):
            gates = 4;
        elif (mode == "GRU" or mode == "GRU_LBR"):
            gates = 3
        state_length = w.shape[0] // gates
        if (mode == "LSTM"):
            if (projection is not None):
                state_length = state_length + projection.shape[0]
            else:
                state_length = 2 * state_length
        state = np.zeros([1, state_length])
        loops = inputs.shape[0]
        results = []
        for i in range(loops):
            result, state = Operators.rnn(mode, inputs[i], state, w, b, projection, projection_bias,
                zoneout_cell, zoneout_output, None, None, False)
            results.append(result)
        results = np.array(results)
        shape = results.shape
        results = results.reshape([1, shape[0], -1])
        Operators.print_data(results, name)
        return results

    @staticmethod
    def bi_rnn(mode, inputs, w, b, projection, projection_bias, zoneout_cell, zoneout_output, name):
        if (not Operators.calculate):
            return None
        fw = w[0]
        fb = b[0]
        bw = w[1]
        bb = b[1]
        if projection is None:
            fp = None
            bp = None
        else:
            fp = projection[0]
            bp = projection[1]
        if projection_bias is None:
            fpb = None
            bpb = None
        else:
            fpb = projection_bias[0]
            bpb = projection_bias[1]
        inputs = np.reshape(inputs, [-1, inputs.shape[-1]])
        fw_state_length = fw.shape[0] // 4
        if (fp is not None):
            fw_state_length += fp.shape[0]
        else:
            fw_state_length += fw.shape[0] // 4
        bw_state_length = bw.shape[0] // 4
        if (bp is not None):
            bw_state_length += bp.shape[0]
        else:
            bw_state_length += bw.shape[0] // 4
        fw_state = np.zeros([1, fw_state_length])
        bw_state = np.zeros([1, bw_state_length])
        loops = inputs.shape[0]
        fw_results = []
        for i in range(loops):
            fw_result, fw_state = Operators.rnn(mode, inputs[i], fw_state, fw, fb, fp, fpb,
                zoneout_cell, zoneout_output, None, None, False)
            fw_results.append(fw_result)
        bw_results = []
        for i in range(loops):
            bw_result, bw_state = Operators.rnn(mode, inputs[loops-1-i], bw_state, bw, bb, bp, bpb,
                zoneout_cell, zoneout_output, None, None, False)
            bw_results.append(bw_result)
        results = []
        for i in range(loops):
            result = np.concatenate([fw_results[i], bw_results[loops-1-i]], 1)
            results.append(result)
        results = np.array(results)
        shape = results.shape
        results = results.reshape([1, shape[0], -1])
        Operators.print_data(results, name)
        return results

    @staticmethod
    def argmax(inputs, axis, name):
        if (not Operators.calculate):
            return None;
        x = np.argmax(inputs, axis = axis)
        Operators.print_data(x, name)
        return np.array([x])

    @staticmethod
    def copy(_src,
             src_batch_stride,
             src_stride,
             src_offset,
             _dst,
             dst_batch_stride,
             dst_stride,
             dst_offset,
             length,
             name,
             src_index=None,
             dst_index=None):
        if (not Operators.calculate):
            return None;
        src = _src.copy()
        dst = _dst.copy()
        batch = len(src)
        src_shape = src.shape
        dst_shape = dst.shape
        src = src.reshape([src_shape[0], src.size//src_shape[0]])
        dst = dst.reshape([dst_shape[0], dst.size//dst_shape[0]])
        if (length < 0):
            length = src.size;
        if (src_batch_stride < 0):
            src_batch_stride = src.shape[1]
        if (src_stride < 0):
            src_stride = src.shape[1]
        if (dst_batch_stride < 0):
            dst_batch_stride = dst.shape[1]
        if (dst_stride < 0):
            dst_stride = dst.shape[1]
        for i in range(batch):
            src_j = 0
            if src_index is not None:
                src_j = src_index[i][0]
            dst_j = 0
            if dst_index is not None:
                dst_j = dst_index[i][0]
            src_index = int(src_j * src_stride + src_offset)
            dst_index = int(dst_j * dst_stride + dst_offset)
            for k in range(length):
                dst[i][dst_index+k] = src[i][src_index+k]
        dst = dst.reshape(dst_shape)
        Operators.print_data(dst, name)
        return dst

    @staticmethod
    def check(_x,
              _y,
              condition,
              name):
        if (not Operators.calculate):
            return None;
        x = _x.copy() 
        y = _y.copy() 
        batch = len(x)
        status = np.array([False] * batch)
        num = x.size // batch
        x = x.reshape([x.size])
        y = y.reshape([y.size])
        for i in range(batch):
            flag = False
            count = 0
            for j in range(num):
                if (condition == "equal"):
                    if (x[i*num + j] == y[i*num + j]):
                        count = count + 1
                elif (condition == "great"):
                    if (x[i*num + j] > y[i*num + j]):
                        count = count + 1
                elif (condition == "greatequal"):
                    if (x[i*num + j] >= y[i*num + j]):
                        count = count + 1
                else:
                    print("[ERROR] unsupported check %s" % (condition))
                    exit(0)
            if (count == num):
                flag = True

            status[i] = flag
        Operators.print_data(status, name)
        return status

    @staticmethod
    def concat(inputs, axis, name):
        if (not Operators.calculate):
            return None;
        rel_inputs = []
        for idx in inputs:
            if (idx.size > 0):
                rel_inputs.append(idx)
        x = np.concatenate(rel_inputs, axis)
        Operators.print_data(x, name)
        return x

    @staticmethod
    def relative_position_embedding(_x, weight, axis, name):
        if (not Operators.calculate):
            return None;
        length = _x.shape[axis]
        h = weight.shape[0]
        w = weight.shape[1]
        if (length > h):
            x = np.zeros([length, w])
            x[length-h:, :] = weight
        else:
            x = weight[h-length:, :]
        Operators.print_data(x, name)
        return x

    @staticmethod
    def pad(_x, pad_shapes, pad_values, name):
        if (not Operators.calculate):
            return None;
        if (pad_values is not None):
            assert(np.array(pad_values).sum() == 0)
        x = _x.copy()
        x = np.pad(x, pad_shapes, mode='constant')
        Operators.print_data(x, name)
        return x

    @staticmethod
    def relative_shift(_x, axis, shift_length, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        shapes = [i for i in x.shape]
        pad_shapes = [[0, 0] for i in range(len(shapes))]
        pad_shapes[axis][0] = shift_length
        x = np.pad(x, pad_shapes, mode='constant')
        tmp = shapes[axis-1]
        shapes[axis-1] = shapes[axis] + 1
        shapes[axis] = tmp
        x = x.reshape(shapes)
        xx = Operators.slice(x, axis-1, [shift_length], ["other", "remain"], False)
        x = xx[1]
        x = x.reshape([i for i in _x.shape])
        Operators.print_data(x, name)
        return x

    @staticmethod
    def clip(_x, min_value, max_value, name):
        if (not Operators.calculate):
            return None;
        x = np.clip(_x, min_value, max_value)
        Operators.print_data(x, name)
        return x
