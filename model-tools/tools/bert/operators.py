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
        print(name)
        num = x.size
        print(num)
        x = np.reshape(x, [num])
        threshold = 16
        if (num > threshold):
            print(x[0 : threshold])
        else:
            print(x)
    

    @staticmethod
    def get_input(name, shape):
        num = 1
        for x in shape:
            num = num * x
        data = None
        if name == "word":
            data = [1] * num
        if name == "position":
            data = [i for i in range(num)]
        if name == "token_type":
            data = [0] * num
        if name == "input_mask":
            data = [1] * num
        if (data is None):
            print("[ERROR] unsupported input")
            exit(1)
        data = np.array(data)
        x = np.reshape(data, [1, len(data)])
        Operators.print_data(x, name)
        return x
    

    @staticmethod
    def layer_norm(x, s, b, name):
        if (not Operators.calculate):
            return None;

        eps = 1e-3
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
    def matmul(x, y, name, print_flag=True):
        if (not Operators.calculate):
            return None;

        x = np.matmul(x, y)
        if print_flag:
            Operators.print_data(x, name)
        return x
    

    @staticmethod
    def fully_connect(x, w, b, name):
        if (not Operators.calculate):
            return None;

        tmp = Operators.matmul(x, w, name, False)
        x = tmp + b
        Operators.print_data(x, name)
        return x
    
    
    @staticmethod
    def reshape(x, dim, name):
        if (not Operators.calculate):
            return None;

        x = np.reshape(x, dim)
        Operators.print_data(x, name)
        return x
    
    
    @staticmethod
    def transpose(x, dim, name):
        if (not Operators.calculate):
            return None;

        x = np.transpose(x, dim)
        Operators.print_data(x, name)
        return x
    
    
    @staticmethod
    def multiply(x, scale, name):
        if (not Operators.calculate):
            return None;

        x = x * scale
        Operators.print_data(x, name)
        return x
    
    
    @staticmethod
    def slice(x, axis, slice_points, names):
        if (not Operators.calculate):
            return None;

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

        x = inputs[0]
        for i in range(1, len(inputs)):
            x = x + inputs[i]
        Operators.print_data(x, name)
        return x
    
    
    @staticmethod
    def attention(x, num_attention, name):
        if (not Operators.calculate):
            return None;

        if (len(x) != 1):
            print("[ERROR] batch != 1")
            exit(0)
        x = np.reshape(x, [len(x[0])])
    
        mask = -10000
        xx = x
        for i in range(len(x)):
            xx[i] = (1 - x[i]) * mask
        xxx = []
        for j in range(len(xx)):
            xxx.append(xx)
        xxxx = []
        for j in range(num_attention):
            xxxx.append(xxx)
        xxxx = np.array(xxxx)
        xxxx = np.reshape(xxxx, [1, num_attention, len(x), len(x)])
        x = xxxx
        Operators.print_data(x, name)
        return x
    

    @staticmethod
    def embedding(x, w, name):
        if (not Operators.calculate):
            return None;

        if (len(x) != 1):
            print("[ERROR] batch != 1")
            exit(0)
        x = np.reshape(x, [len(x[0])])
    
        y = []
        for i in x:
            y.append(w[i])
        y = np.array(y)
        x = np.reshape(y, [1, len(x), len(w[0])])
        Operators.print_data(x, name)
        return x;
    
    
    @staticmethod
    def softmax(x, name):
        if (not Operators.calculate):
            return None;

        x_row_max = x.max(axis=-1)
        x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
        x = x - x_row_max
        x_exp = np.exp(x)
        x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
        x = x_exp / x_exp_row_sum
        Operators.print_data(x, name)
        return x
    
    
    @staticmethod
    def gelu(x, name):
        if (not Operators.calculate):
            return None;

        cdf =0.5 * (1.0 + np.tanh((0.7978845608028654 * (x + 0.044715 * x * x * x))))
        x = x * cdf
        Operators.print_data(x, name)
        return x
    
    
    @staticmethod
    def tanh(x, name):
        if (not Operators.calculate):
            return None;

        x = np.tanh(x)
        Operators.print_data(x, name)
        return x
