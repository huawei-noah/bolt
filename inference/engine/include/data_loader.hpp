// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_DATA_LOADER
#define _H_DATA_LOADER

#include <string>
#include "parameter_spec.h"
#include "tensor.hpp"

int string_end_with(std::string s, std::string sub);

void get_files(std::string directoryName, std::vector<std::string> &files);

std::vector<Tensor> load_fake_data(std::vector<TensorDesc> dataDesc);

std::vector<Tensor> load_txt(std::string dataPath, std::vector<TensorDesc> dataDesc);

std::vector<Tensor> load_bin(
    std::string dataPath, std::vector<DataType> sourceDataType, std::vector<TensorDesc> dataDesc);

std::vector<std::string> load_data(std::string directoryPath,
    std::vector<TensorDesc> dataDesc,
    std::vector<std::vector<Tensor>> *datas);

#ifdef _BUILD_TEST
std::vector<std::string> load_image_with_scale(std::string directoryPath,
    std::vector<TensorDesc> dataDesc,
    std::vector<std::vector<Tensor>> *datas,
    ImageFormat ImageFormat,
    F32 scaleValue);
#endif
#endif
