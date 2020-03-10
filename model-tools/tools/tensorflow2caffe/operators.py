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
    def print_data(x, name):
        shape_str = "("
        for i in x.shape:
            shape_str = shape_str + str(i) + ","
        shape_str = shape_str[0:-1] + ")"
        print("%s %s" % (name, shape_str))
        num = x.size
        x = np.reshape(x, [num])
        threshold = 32
        if (num < threshold):
            threshold = num
        print(x[:threshold])

    @staticmethod
    def zeros(shape, name):
        x = np.zeros(shape)
        Operators.print_data(x, name)
        return x

    @staticmethod
    def weight(x, name):
        Operators.print_data(x, name)
        return x

    @staticmethod
    def layer_norm(_x, s, b, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        eps = 1e-6
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
        x = _x.copy()
        x = np.matmul(x, w)
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
    def multiply(_x, scale, bias, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        x = x * scale
        x = x + bias
        Operators.print_data(x, name)
        return x
    
    @staticmethod
    def matmultiply(_x, _y, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        y = _y.copy()
        x = np.multiply(x, y)
        Operators.print_data(x, name)
        return x
    
    @staticmethod
    def slice(_x, axis, slice_points, names):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        if (len(x) != 1):
            print("[ERROR] batch != 1")
            exit(0)
        result = []
        result.append(x[0, 0])
        result.append(x[0, 1:])
        Operators.print_data(result[0], names[0])
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
    def embedding(_x, w, transpose, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        if (len(x) != 1):
            print("[ERROR] batch != 1")
            exit(0)
        x = np.reshape(x, [len(x[0])])
    
        y = []
        for i in x:
            index = int(i)
            if (transpose):
                y.append(w[:,index])
            else:
                y.append(w[index])
        y = np.array(y)
        if (transpose):
            x = np.reshape(y, [1, len(x), len(w)])
        else:
            x = np.reshape(y, [1, len(x), len(w[0])])
        Operators.print_data(x, name)
        return x;
    
    @staticmethod
    def softmax(_x, name):
        if (not Operators.calculate):
            return None;
        x = _x.copy()
        x_row_max = x.max(axis=-1)
        x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
        x = x - x_row_max
        x_exp = np.exp(x)
        x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
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
    def relu(_x, name):
        if (not Operators.calculate):
            return None;
        x = np.maximum(_x, 0, _x)
        #x[x < 0] = 0
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

        x = _sigmoid(_x)
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
    def axis_mean(input, axis, name):
        if (not Operators.calculate):
            return None;
        x = np.mean(input, axis)
        Operators.print_data(x, name)
        return x
    
    @staticmethod
    def lstm(inputs, state, w, b, name, state_name):
        if (not Operators.calculate):
            return None;
        
        c, h = np.split(state, indices_or_sections = 2, axis = 1)
        x = np.concatenate([inputs, h], axis = 1)
        x = np.matmul(x, w)
        if (b is not None):
            x = x + b
        i, j, f, o = np.split(x, indices_or_sections = 4, axis = 1)
        new_c = np.multiply(c, Operators._sigmoid(np.add(f, 1.0))) \
                + np.multiply(Operators._sigmoid(i), np.tanh(j))
        new_h = np.multiply(np.tanh(new_c), Operators._sigmoid(o))
        new_state = np.concatenate([new_c, new_h], axis = 1)

        Operators.print_data(new_h, name)
        Operators.print_data(new_state, state_name)
        return new_h, new_state

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
                else:
                    print("[ERROR] unsupported check %s" % (condition))
                    exit(0)
            if (count == num):
                flag = True

            status[i] = flag
        Operators.print_data(status, name)
        return status
