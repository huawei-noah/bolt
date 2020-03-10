// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _LAYER_NORM_H
#define _LAYER_NORM_H

#include "operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "op_type.h"

class LayerNorm: public WeightOperator {
public:
    LayerNorm(DataType dt, U32 weightNum){
        this->dt = dt;
        this->weightNum = weightNum;
        this->hasBias = false;
    }

    OperatorType get_op_type() override
    {
        return OT_LayerNorm;
    }

    EE init_weight_bias_from_model(U8** modelPtr)
    {
        auto curOpWs = this->get_weightspec_ptr();
        if(modelPtr == nullptr){
            this->weightNum = curOpWs.bytes_of_weight / bytesOf(curOpWs.mdt);
        }

        TensorDesc weightDesc = tensor1d(this->dt, this->weightNum);
        TensorDesc biasDesc = tensor1d(this->dt, this->weightNum);
        std::shared_ptr<Tensor> modelWeightTensor(new Tensor());
        std::shared_ptr<Tensor> modelBiasTensor(new Tensor());
        modelWeightTensor->set_desc(weightDesc);
        modelBiasTensor->set_desc(biasDesc);
        U32 weightBytes = tensorNumBytes(weightDesc);
        if(modelPtr != nullptr){
            modelWeightTensor->alloc();
            memcpy((U8*)modelWeightTensor->get_val(), *modelPtr, weightBytes);
            *modelPtr += weightBytes;
        } else {
            modelWeightTensor->set_val(curOpWs.weight);
        }

        U8* biasVal = nullptr;
        if(modelPtr != nullptr){
            if(this->hasBias){
                biasVal = *modelPtr;
                *modelPtr += tensorNumBytes(biasDesc);
            }
        } else {
            if(this->hasBias) biasVal = curOpWs.vec; 
        }

        if(biasVal){
            modelBiasTensor->set_val(biasVal);
        } else {
            modelBiasTensor->alloc();
            memset((U8*)modelBiasTensor->get_val(), 0, tensorNumBytes(biasDesc));
        }

        this->weightTensors.push_back(*modelWeightTensor.get());
        this->biasTensors.push_back(*modelBiasTensor.get());
        return SUCCESS;
    }

    void run() override 
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor weightTensor = this->weightTensors[0];
        Tensor biasTensor = this->biasTensors[0];

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        CHECK_STATUS(layer_normalization(weightTensor.get_val(),
                                         biasTensor.get_val(),
                                         inputDesc, inputTensor.get_val(),
                                         outputDesc, outputTensor.get_val(), this->schedule));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override {
        TensorDesc in_dim = inDims[0];
        CHECK_STATUS(normalization_infer_output_size(in_dim, &((*outDims)[0])));
        return SUCCESS;
    }
private:
    U32 weightNum;
};

#endif //_LAYER_NORM_H
