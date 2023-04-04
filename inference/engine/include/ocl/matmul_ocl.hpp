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
        INIT_GPU_INFO(&this->runInfo)
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
        Tensor inputTensorC = Tensor(OCLMem);
        if (this->inputTensors.size() > 2) {
            inputTensorC = this->inputTensors[2];
        }
        Tensor outputTensor = this->outputTensors[0];
        Tensor tmpTensor = Tensor(OCLMem);
        std::vector<Tensor> tmpTensors(3, tmpTensor);
        tmpTensors[0] = this->temp;
        get_tmp_image(0, bytes + 1, &tmpTensors[1]);
        get_tmp_image(1, bytes + 4, &tmpTensors[2]);

        CHECK_STATUS(matmul(inputTensorA, this->p.transpose_a, inputTensorB, this->p.transpose_b,
            inputTensorC, tmpTensors, outputTensor, &this->archInfo));
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
        std::string name = this->name + std::to_string(get_type());
        EE ret = SUCCESS;
        if (algorithmMap->getAlgorithmInfoFromMap(name, algo, 4)) {
            this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
            this->runInfo.best_h[0] = algo[1];
            this->runInfo.best_c[0] = algo[2];
            this->runInfo.best_k[0] = algo[3];
        } else {
            ret = matmul_infer_forward_algorithm(matrixATensor, this->p.transpose_a, matrixBTensor,
                this->p.transpose_b, matrixCTensor, &this->archInfo);
            algo[0] = this->runInfo.algorithm;
            algo[1] = this->runInfo.best_h[0];
            algo[2] = this->runInfo.best_c[0];
            algo[3] = this->runInfo.best_k[0];
            algorithmMap->setAlgorithmInfoToMap(name, algo, 4);
        }
        return ret;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        CHECK_REQUIREMENT(inTensors.size() == 2);
        EE ret = matmul_infer_output_size(inTensors[0], this->p.transpose_a, inTensors[1],
            this->p.transpose_b, outTensors[0], &this->archInfo);
        if (ret == SUCCESS && check_tensors_image(inTensors)) {
            ret = set_tensors_image(outTensors, inTensors.size());
        }
        return ret;
    }

    U32 infer_tmp_memory_size() override
    {
        for (U32 i = 0; i < 7; i++) {
            bytes[i] = 0;
        }
        CHECK_STATUS(matmul_infer_forward_tmp_bytes(this->inputTensors[0], this->p.transpose_a,
            this->inputTensors[1], this->p.transpose_b, this->outputTensors[0], bytes,
            &this->archInfo));
        add_tmp_image(0, bytes + 1);
        add_tmp_image(1, bytes + 4);
        return bytes[0];
    }

    REGISTER_OCL_OPERATOR_RUN

protected:
    ForwardRunInfoMali runInfo;
    U32 bytes[7];
};

#endif  // _MATMUL_OCL_H
