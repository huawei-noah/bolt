// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _MATMUL_CPU_H
#define _MATMUL_CPU_H

#include "matmul.hpp"

class MatMulCPU : public MatMul {
public:
    MatMulCPU(DataType dt, MatMulParamSpec p) : MatMul(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<MatMulCPU> mem =
            std::shared_ptr<MatMulCPU>(new MatMulCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor inputTensorA = this->inputTensors[0];
        Tensor inputTensorB = this->inputTensors[1];
        Tensor inputTensorC;
        if (this->inputTensors.size() > 2) {
            inputTensorC = this->inputTensors[2];
        }
        Tensor outputTensor = this->outputTensors[0];

#ifdef _USE_INT8
        TensorDesc inputDescA = inputTensorA.get_desc();
        TensorDesc inputDescB = inputTensorB.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        if (3 == featureScale.size() && featureScale[0][0] > 0 && DT_I8 != inputDescA.dt &&
            DT_U8_Q != inputDescA.dt) {
            inputTensorA.set_scale(featureScale[0][0]);
        }
        if (3 == featureScale.size() && featureScale[1][0] > 0 && DT_I8 != inputDescB.dt &&
            DT_U8_Q != inputDescB.dt) {
            inputTensorB.set_scale(featureScale[1][0]);
        }
        if (featureScale.size() > 0) {
            outputTensor.set_scale((featureScale.back())[0]);
        }
#endif
        std::vector<Tensor> tmpTensor(1, this->temp);
        CHECK_STATUS(matmul(inputTensors[0], this->p.transpose_a, inputTensors[1],
            this->p.transpose_b, inputTensorC, tmpTensor, outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        EE ret = matmul_infer_output_size(inTensors[0], this->p.transpose_a, inTensors[1],
            this->p.transpose_b, outTensors[0], &this->archInfo);
        if (isQuantMixDataType(this->dt)) {
            auto outputDesc = outTensors[0]->get_desc();
            if (featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
                outputDesc.dt = noQuantDataType(this->dt);
            } else {
                outputDesc.dt = get_activation_quant_data_type();
            }
            outTensors[0]->resize(outputDesc);
        }
        return ret;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(matmul_infer_forward_tmp_bytes(inputTensors[0], this->p.transpose_a,
            inputTensors[1], this->p.transpose_b, outputTensors[0], &bytes, &this->archInfo));
        return bytes;
    }
};

#endif  // _MATMUL_CPU_H
