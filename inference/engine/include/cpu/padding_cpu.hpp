// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _PADDING_CPU_H
#define _PADDING_CPU_H

#include "padding.hpp"

class PaddingCPU : public Padding {
public:
    PaddingCPU(DataType dt, PadParamSpec p) : Padding(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<PaddingCPU> mem =
            std::shared_ptr<PaddingCPU>(new PaddingCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        PadParamSpec ps = p;
        if (ps.top == UNI_RESERVE && inputTensors.size() > 1) {
            ps = get_param(inputTensors[1].get_desc());
        }
        CHECK_STATUS(padding(inputTensor, ps, outputTensor, &this->archInfo));
    }

    PadParamSpec get_param(TensorDesc desc)
    {
        PadParamSpec ps = this->p;
        int num = tensorNumElements(desc);
        switch (num) {
            case 8: {
                ps.front = desc.dims[desc.nDims + 1];
                ps.top = desc.dims[desc.nDims + 2];
                ps.left = desc.dims[desc.nDims + 3];
                ps.back = desc.dims[desc.nDims + num / 2 + 1];
                ps.bottom = desc.dims[desc.nDims + num / 2 + 2];
                ps.right = desc.dims[desc.nDims + num / 2 + 3];
                break;
            }
            case 6: {
                ps.front = desc.dims[desc.nDims + 1];
                ps.top = desc.dims[desc.nDims + 2];
                ps.back = desc.dims[desc.nDims + num / 2 + 1];
                ps.bottom = desc.dims[desc.nDims + num / 2 + 2];
                ps.left = ps.right = 0;
                break;
            }
            case 4: {
                ps.front = desc.dims[desc.nDims + 1];
                ps.back = desc.dims[desc.nDims + num / 2 + 1];
                ps.top = ps.bottom = 0;
                ps.left = ps.right = 0;
                break;
            }
            default:
                UNI_ERROR_LOG("can not process pad's parameter from input.\n");
                break;
        }
        return ps;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        PadParamSpec ps = p;
        if (ps.top == UNI_RESERVE && inTensors.size() > 1) {
            ps = get_param(inTensors[1]->get_desc());
        }
        return padding_infer_output_size(inTensors[0], ps, outTensors[0], &this->archInfo);
    }
};

#endif  // _PADDINGCPU_H
