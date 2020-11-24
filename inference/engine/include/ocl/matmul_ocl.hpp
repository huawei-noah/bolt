// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _MATMUL_OCL_H
#define _MATMUL_OCL_H

#include "matmul.hpp"

class MatMulOCL : public MatMul {
public:
    MatMulOCL(DataType dt, MatMulParamSpec p) : MatMul(dt, p)
    {
        setMALIArchInfo(&(this->archInfo), &(this->runInfo), &this->needSetKernelVec,
            &this->needSelectKernelLS);
    }

    ~MatMulOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<MatMulOCL> mem =
            std::shared_ptr<MatMulOCL>(new MatMulOCL(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor inputTensorA = this->inputTensors[0];
        Tensor inputTensorB = this->inputTensors[1];
        Tensor outputTensor = this->outputTensors[0];

        CHECK_STATUS(matmul(inputTensorA, this->p.transpose_a, inputTensorB, this->p.transpose_b,
            this->temp, outputTensor, &this->archInfo));
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        OCLContext::getInstance().handle.get()->kernelVec = &this->opKernelVec;
        Tensor matrixATensor = this->inputTensors[0];
        Tensor matrixBTensor = this->inputTensors[1];
        Tensor matrixCTensor = this->outputTensors[0];
        ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
            CONVOLUTION_ALGORITHM_NULL;
        I32 algo[4];
        if (algorithmMap->getAlgorithmInfoFromMap(this->name, algo, 4)) {
            this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
            this->runInfo.best_w[0] = algo[1];
            this->runInfo.best_c[0] = algo[2];
            this->runInfo.best_k[0] = algo[3];
        } else {
            CHECK_STATUS(matmul_infer_forward_algorithm(matrixATensor, this->p.transpose_a,
                matrixBTensor, this->p.transpose_b, matrixCTensor, &this->archInfo));
            algo[0] = this->runInfo.algorithm;
            algo[1] = this->runInfo.best_w[0];
            algo[2] = this->runInfo.best_c[0];
            algo[3] = this->runInfo.best_k[0];
            algorithmMap->setAlgorithmInfoToMap(this->name, algo, 4);
        }
        return SUCCESS;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        CHECK_STATUS(matmul_infer_output_size(inTensors[0], this->p.transpose_a, inTensors[1],
            this->p.transpose_b, outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(matmul_infer_forward_tmp_bytes(this->inputTensors[0], this->p.transpose_a,
            this->inputTensors[1], this->p.transpose_b, &bytes, &this->archInfo));
        return bytes;
    }

    REGISTER_OCL_OPERATOR_RUN

protected:
    ForwardRunInfoMali runInfo;
};

#endif  // _MATMUL_OCL_H
