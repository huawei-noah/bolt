// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _QUANTIZELINEAR_CPU_H
#define _QUANTIZELINEAR_CPU_H

#include "quantizelinear.hpp"

class QuantizeLinearCPU : public QuantizeLinear {
public:
    QuantizeLinearCPU(DataType dt, QuantizeLinearParamSpec p) : QuantizeLinear(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<QuantizeLinearCPU> mem =
            std::shared_ptr<QuantizeLinearCPU>(new QuantizeLinearCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        F32 scale = -1;  // default per layer
        TensorDesc inputDesc = this->inputTensors[0].get_desc();
        TensorDesc outputDesc = this->outputTensors[0].get_desc();
        if (inputDesc.dt == outputDesc.dt) {
            UNI_MEMCPY(get_ptr_from_tensor(this->outputTensors[0], this->archInfo.arch),
                get_ptr_from_tensor(this->inputTensors[0], this->archInfo.arch),
                tensorNumBytes(this->inputTensors[0].get_desc()));
            return;
        }
        if (featureScale.size() > 0 && featureScale[0].size() > 0 && featureScale[0][0] > 0) {
            scale = featureScale[0][0];
        }
        if (scale <= 0) {
            scale = this->inputTensors[0].get_scale();
        }
        CHECK_STATUS(
            quantize(this->inputTensors[0], &this->outputTensors[0], &scale, &this->archInfo));
        this->outputTensors[0].set_scale(scale);
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        TensorDesc outputDesc = inTensors[0]->get_desc();
#ifdef _USE_INT8
        if (isQuantMixDataType(this->dt)) {
            if (IS_X86(this->archInfo.arch)) {
                outputDesc.dt = p.dt;
                // special case, matmul mvm
                if (outputDesc.nDims >= 2 && outputDesc.dims[1] != 1) {
                    outputDesc.dt = get_activation_quant_data_type();
                }
            } else {
                outputDesc.dt = get_activation_quant_data_type();
            }
        }
#endif
        outTensors[0]->resize(outputDesc);
        return SUCCESS;
    }
};

#endif  // _QUANTIZELINEAR_CPU_H
