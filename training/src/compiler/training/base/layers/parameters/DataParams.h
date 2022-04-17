// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DATALAYER_PARAMS_H
#define DATALAYER_PARAMS_H

#include <string>
#include <vector>

#include "BasicParameters.h"

namespace raul
{
/**
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param d depth size in elements
 * @param h height size in elements
 * @param w width size in elements
 * @param labelsCnt size of the labels in elements
 */
struct DataParams : public BasicParams
{
    DataParams() = delete;
    DataParams(const Names& outputs, size_t d, size_t h, size_t w, size_t labelsCnt = 0);

    size_t depth;
    size_t height;
    size_t width;
    size_t labelsCount;

    void print(std::ostream& stream) const override;
};
} // raul namespace

#endif