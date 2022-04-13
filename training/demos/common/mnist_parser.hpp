// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

template <typename T>
void read_Mnist_Label(std::string filename, std::vector<T> &labels)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        file.read((char *)&number_of_images, sizeof(number_of_images));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        for (int i = 0; i < number_of_images; i++) {
            unsigned char label = 0;
            file.read((char *)&label, sizeof(label));
            labels.push_back((T)label);
        }
    }
}

template <typename T>
void read_Mnist_Images(std::string filename, std::vector<std::vector<T>> &images)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        file.read((char *)&number_of_images, sizeof(number_of_images));
        file.read((char *)&n_rows, sizeof(n_rows));
        file.read((char *)&n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);
        for (int i = 0; i < number_of_images; i++) {
            std::vector<T> tp;
            for (int r = 0; r < n_rows; r++) {
                for (int c = 0; c < n_cols; c++) {
                    unsigned char image = 0;
                    file.read((char *)&image, sizeof(image));
                    tp.push_back(image / 255.0);
                }
            }
            images.push_back(tp);
        }
    }
}

template <typename T1, typename T2>
std::vector<std::vector<T2>> one_hot_encoding(std::vector<T1> labels, const int one_hot_len)
{
    std::vector<std::vector<T2>> res;
    for (int i = 0; i < (int)(labels.size()); i++) {
        std::vector<T2> tmp(one_hot_len, 0);
        tmp[(int)(labels[i])] = (T2)1.0;
        res.emplace_back(tmp);
    }
    return res;
}
