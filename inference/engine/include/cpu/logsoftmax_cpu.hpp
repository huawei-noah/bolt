// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _LOGSOFTMAX_CPU_H
#define _LOGSOFTMAX_CPU_H

#include "cpu/softmax_cpu.hpp"

class LogSoftmaxCPU : public SoftmaxCPU {
public:
    LogSoftmaxCPU(DataType dt, SoftmaxParamSpec p) : SoftmaxCPU(dt, p)
    {}

    OperatorType get_type() override
    {
        return OT_LogSoftmax;
    }

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<LogSoftmaxCPU> mem =
            std::shared_ptr<LogSoftmaxCPU>(new LogSoftmaxCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        CHECK_STATUS(
            logsoftmax(inputTensors[0], this->p, this->temp, outputTensors[0], &this->archInfo));
    }
};
#endif  // LOGSOFTMAX_CPU_H
