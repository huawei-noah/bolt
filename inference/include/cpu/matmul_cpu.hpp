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
#ifndef _MATMUL_CPU_H
#define _MATMUL_CPU_H

#include "operator.hpp"
#include "tensor_computing.h"
#include "matmul.hpp"

class MatMulCPU: public MatMul {
public:
    MatMulCPU(DataType dt, bool transposeA, bool transposeB) : MatMul(dt, transposeA, transposeB) {}

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensorA =  this->inputTensors[0];
        TensorDesc inputDescA = inputTensorA.get_desc();
        Tensor inputTensorB =  this->inputTensors[1];
        TensorDesc inputDescB = inputTensorB.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        U8 *inputA = inputTensorA.get_val();
        U8 *inputB = inputTensorB.get_val();
        U8 *tmp = (U8*)this->temp->get_val();

        if (DT_I8 == inputDescA.dt || DT_I8 == inputDescB.dt) {
#ifdef _USE_INT8
            F32 scaleO = 1;
            if (DT_F16 == inputDescA.dt) {
                F16 *inD = (F16*)inputA;
                INT8 *inQ = (INT8*)tmp;
                F16 scale = -1;
                if (featureScale.size() == 3 && featureScale[0][0] > 0) {
                    scale = featureScale[0][0];
                }
                quantize_tensor(inputDescA, inD, &inputDescA, inQ, &scale);
                scaleO *= scale;
                inputA = (U8*)tmp;
                tmp += tensorNumBytes(inputDescA);
            } else {
                scaleO *= inputTensorA.get_scale();
            }
            if (DT_F16 == inputDescB.dt) {
                F16 *inD = (F16*)inputB;
                INT8 *inQ = (INT8*)tmp;
                F16 scale = -1;
                if (featureScale.size() == 3 && featureScale[1][0] > 0) {
                    scale = featureScale[1][0];
                }
                quantize_tensor(inputDescB, inD, &inputDescB, inQ, &scale);
                scaleO *= scale;
                inputB = (U8*)tmp;
                tmp += tensorNumBytes(inputDescB);
            } else {
                scaleO *= inputTensorB.get_scale();
            }
            outputDesc.dt = DT_I32;
            I32 *result = (I32*)tmp;
            U8 *tmpReal = tmp + tensorNumBytes(outputDesc);
            CHECK_STATUS(matmul(inputDescA, this->transposeA, inputA,
                            inputDescB, this->transposeB, inputB,
                            tmpReal, this->lenOfTemp,
                            outputDesc, result, this->schedule));
            if (DT_I8 == outputTensor.get_desc().dt) {
                CHECK_STATUS(quantize_tensor(outputDesc, result, &outputDesc, outputTensor.get_val(), &scaleO));
                outputTensor.set_scale(scaleO);
            } else {
                    CHECK_REQUIREMENT(DT_F16 == outputTensor.get_desc().dt) {
                    F16 *output = outputTensor.get_val();
                    dequantize_int32_to_fp16(tensorNumElements(outputDesc), result, scaleO, output);
                }
            }
#endif
        } else {
            CHECK_STATUS(matmul(inputDescA, this->transposeA, inputA,
                            inputDescB, this->transposeB, inputB,
                            tmp, this->lenOfTemp,
                            outputDesc, outputTensor.get_val(), this->schedule));
        }
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inDimA = inDims[0];
        TensorDesc inDimB = inDims[1];
        CHECK_STATUS(matmul_infer_output_size(inDimA, this->transposeA, inDimB, this->transposeB, &((*outDims)[0]), this->schedule));
        if (DT_F16_8Q == this->dt && featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
            (*outDims)[0].dt = DT_F16;
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        TensorDesc inputDescA = (this->inputTensors[0]).get_desc();
        TensorDesc inputDescB = (this->inputTensors[1]).get_desc();
        U32 bytes = 0;
        CHECK_STATUS(matmul_infer_forward_tmp_bytes(inputDescA, this->transposeA, inputDescB, this->transposeB, &bytes, this->schedule));
        return bytes;
    }
};

#endif //_MATMUL_CPU_H
