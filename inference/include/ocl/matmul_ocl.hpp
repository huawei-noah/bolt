// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


/**
 * Project deploy
 */
#ifndef _MATMUL_OCL_H
#define _MATMUL_OCL_H

#include "operator.hpp"
#include "tensor_computing.h"
#include "matmul.hpp"

class MatMulOCL: public MatMul {
public:
    MatMulOCL(DataType dt, bool transposeA, bool transposeB) : MatMul(dt, transposeA, transposeB) {}

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        this->handle->curOpName = this->get_op_name();
        Tensor inputTensorA =  this->inputTensors[0];
        TensorDesc inputDescA = inputTensorA.get_desc();
        Tensor inputTensorB =  this->inputTensors[1];
        TensorDesc inputDescB = inputTensorB.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        U8 *inputA = inputTensorA.get_val();
        U8 *inputB = inputTensorB.get_val();
        U8 *tmp = (U8*)this->temp->get_val();

        CHECK_STATUS(matmul(inputDescA, this->transposeA, inputA,
            inputDescB, this->transposeB, inputB, tmp, this->lenOfTemp,
            outputDesc, outputTensor.get_val(), this->schedule, &this->oclExtInfo));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_forward_algorithm(HashMap<std::string, std::string> &algorithmMap) override
    {
        TensorDesc matrixADesc = (this->inputTensors[0]).get_desc();
        TensorDesc matrixBDesc = (this->inputTensors[1]).get_desc();
        TensorDesc matrixCDesc = this->outputTensors[0].get_desc();
        this->oclExtInfo.maliInfo.forwardRunInfo->algorithm = CONVOLUTION_ALGORITHM_NULL;
        if (algorithmMap.find(this->name) != algorithmMap.end()) {
            I32 algo[4];
            Operator::getAlgorithmInfoFromMap(algorithmMap, this->name, algo, 4);
            this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
            this->runInfo.best_w[0] = algo[1];
            this->runInfo.best_c[0] = algo[2];
            this->runInfo.best_k[0] = algo[3];
        } else {
            CHECK_STATUS(matmul_infer_forward_algorithm(matrixADesc, this->transposeA, matrixBDesc, this->transposeB, matrixCDesc, this->schedule, &this->oclExtInfo));
                I32 algo[4];
                algo[0] = this->runInfo.algorithm;
                algo[1] = this->runInfo.best_w[0];
                algo[2] = this->runInfo.best_c[0];
                algo[3] = this->runInfo.best_k[0];
                Operator::setAlgorithmInfoToMap(algorithmMap, this->name, algo, 4);
        }
        return SUCCESS;
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inDimA = inDims[0];
        TensorDesc inDimB = inDims[1];
        this->oclExtInfo.maliInfo.gclmemInputDesc =  NULL;
        this->oclExtInfo.maliInfo.gclmemOutputDesc = NULL;
        CHECK_STATUS(matmul_infer_output_size(inDimA, this->transposeA, inDimB, this->transposeB, &((*outDims)[0]), this->schedule, &this->oclExtInfo));
        return SUCCESS;
    }

    virtual EE infer_gclmem_desc(Vec<GCLMemDesc>* gclmemInputDesc, Vec<GCLMemDesc>* gclmemOutputDesc) override
    {
        TensorDesc inDimA  = this->inputTensors[0].get_desc();
        TensorDesc inDimB  = this->inputTensors[1].get_desc();
        this->oclExtInfo.maliInfo.gclmemInputDesc =  &((*gclmemInputDesc)[0]);
        this->oclExtInfo.maliInfo.gclmemOutputDesc = &((*gclmemOutputDesc)[0]);
        CHECK_STATUS(matmul_infer_output_size(inDimA, this->transposeA, inDimB, this->transposeB, NULL, this->schedule, &this->oclExtInfo));
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        TensorDesc inputDescA = (this->inputTensors[0]).get_desc();
        TensorDesc inputDescB = (this->inputTensors[1]).get_desc();
        U32 bytes = 0;
        CHECK_STATUS(matmul_infer_forward_tmp_bytes(inputDescA, this->transposeA, inputDescB, this->transposeB, &bytes, this->schedule, &this->oclExtInfo));
        return bytes;
    }
};

#endif //_MATMUL_OCL_H
