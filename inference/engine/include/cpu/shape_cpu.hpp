// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _SHAPE_CPU_H
#define _SHAPE_CPU_H

#include "shape.hpp"

class ShapeCPU : public Shape {
public:
    ShapeCPU() : Shape()
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ShapeCPU> mem = std::shared_ptr<ShapeCPU>(new ShapeCPU());
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        UNI_MEMCPY(((CpuMemory *)(outputTensor.get_memory()))->get_ptr(), inputDesc.dims,
            inputDesc.nDims * sizeof(U32));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        TensorDesc inputDesc = inTensors[0]->get_desc();
        TensorDesc outputDesc = tensor1d(DT_U32, inputDesc.nDims);
        outTensors[0]->resize(outputDesc);
        return SUCCESS;
    }
};

#endif  // _SHAPE_CPU_H
