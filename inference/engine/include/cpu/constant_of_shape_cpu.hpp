// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _CONSTATNT_OF_SHAPE_CPU_H
#define _CONSTATNT_OF_SHAPE_CPU_H

#include "constant_of_shape.hpp"

class ConstantOfShapeCPU : public ConstantOfShape {
public:
    ConstantOfShapeCPU(DataType dt, ConstantOfShapeParamSpec p) : ConstantOfShape(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ConstantOfShapeCPU> mem =
            std::shared_ptr<ConstantOfShapeCPU>(new ConstantOfShapeCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        TensorDesc desc = this->outputTensors[0].get_desc();
        UNI_INIT(tensorNumElements(desc), desc.dt, this->p.value,
            ((CpuMemory *)(this->outputTensors[0].get_memory()))->get_ptr());
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        TensorDesc inDesc = inTensors[0]->get_desc();
        TensorDesc outDesc;
        outDesc.dt = this->p.dt;
        outDesc.nDims = inDesc.dims[0];
        outDesc.df = getTensorDefaultDataFormat(outDesc.nDims);
        for (U32 i = 0; i < outDesc.nDims; i++) {
            outDesc.dims[i] = inDesc.dims[inDesc.nDims + inDesc.dims[0] - 1 - i];
        }
        if (tensorIsShape(inDesc) && tensorIsShape(outDesc)) {
            for (U32 i = 0; i < tensorNumElements(outDesc); i++) {
                outDesc.dims[outDesc.nDims + i] = p.value;
            }
        }
        outTensors[0]->resize(outDesc);
        return SUCCESS;
    }
};

#endif  // CONSTATNT_OF_SHAPE_CPU_H
