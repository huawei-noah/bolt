// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SCALE_CPU_H
#define _SCALE_CPU_H

#include <iostream>
#include "weight_operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "op_type.h"
#include "scale.hpp"

class ScaleCPU: public Scale
{
public:
    ScaleCPU(DataType dt, int axis, int numChannels, int numSource):
        Scale(dt, axis, numChannels, numSource)
    {
        this->alpha = nullptr;
        this->beta = nullptr;
    }

    virtual EE init_weight_bias_from_model(U8** modelPtr) override
    {
        auto curOpWs = this->get_weightspec_ptr();
        U32 weightNum = 0;
        if(modelPtr == nullptr){
            weightNum = curOpWs.bytes_of_weight / UNI_MAX(1, bytesOf(curOpWs.mdt));
            if (0 == weightNum) {
                weightNum = curOpWs.bytes_of_vec / UNI_MAX(1, bytesOf(curOpWs.mdt));
            }
        }

        TensorDesc weightDesc = tensor1d(this->dt, weightNum);
        TensorDesc biasDesc   = weightDesc;
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
            modelWeightTensor->set_shared_ptr(std::shared_ptr<U8>(curOpWs.weight));
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

        if (biasVal) {
            modelBiasTensor->set_shared_ptr(std::shared_ptr<U8>(biasVal));
        } else {
            modelBiasTensor->alloc();
            memset((U8*)modelBiasTensor->get_val(), 0, tensorNumBytes(biasDesc));
        }

        this->weightTensors.push_back(*modelWeightTensor.get());
        this->biasTensors.push_back(*modelBiasTensor.get());
        return SUCCESS;
    }

    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        int inputTensorNumber = this->inputTensors.size();
        Tensor inputTensor = this->inputTensors[this->dataID];;
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        U8* inputPtr = inputTensor.get_val();

        if (inputTensorNumber == 1) {
            this->alpha = this->weightTensors[0].get_val();
            this->beta = this->biasTensors[0].get_val();
            CHECK_STATUS(scale(inputDesc, inputPtr, this->axis, this->alpha, this->beta,
                inputTensor.get_desc(), outputTensor.get_val(), this->schedule));
        } else {
            CHECK_STATUS(scale(inputDesc, inputPtr,
                this->axis, this->inputTensors[1-this->dataID].get_val(), nullptr,
                inputTensor.get_desc(), outputTensor.get_val(), this->schedule));
        }
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override 
    {
        I32 tmpAxis = (this->axis + inDims[0].nDims) % inDims[0].nDims;
        tmpAxis = inDims[0].nDims - 1 - tmpAxis;
        CHECK_REQUIREMENT(tmpAxis < (I32)inDims[0].nDims);
        U32 ic = inDims[0].dims[tmpAxis];

        auto curOpWs = this->get_weightspec_ptr();
        this->alpha = curOpWs.weight;
        this->beta = curOpWs.vec;
        U32 numChannels;
        if (0 != curOpWs.bytes_of_weight) {
            numChannels = curOpWs.bytes_of_weight / UNI_MAX(1, bytesOf(curOpWs.mdt));
        } else if (0 != curOpWs.bytes_of_vec) {
            numChannels = curOpWs.bytes_of_vec / UNI_MAX(1, bytesOf(curOpWs.mdt));
        } else {
            numChannels = 0;
        }

        TensorDesc inputDesc;
        if (ic != numChannels && 0 != numChannels) {
            std::cout << "[ERROR] ScaleCPU input channels (IC) do not match. Perhaps some channel padding has been done earlier" << std::endl;
            std::cout << "          IC is now " << ic << " but should be " << numChannels << std::endl;
            CHECK_STATUS(NOT_SUPPORTED);
        } else {
            if (inDims.size() > 1 && tensorNumElements(inDims[1]) > tensorNumElements(inDims[0])) {
                this->dataID = 1;
            }
            inputDesc = inDims[this->dataID];
        }

        CHECK_STATUS(scale_infer_output_size(inputDesc, &((*outDims)[0]), this->schedule));
        return SUCCESS;
    }

#ifdef _USE_FP16
    void set_scale_alpha(F16* alpha) 
    {
        this->alpha = (U8*)alpha;
    }

    F16* get_scale_alpha() 
    {
        return (F16*)(this->alpha);
    }

    void set_scale_beta(F16* beta)
    {
        this->beta = (U8*)beta;
    }

    F16* get_scale_beta()
    {
        return (F16*)(this->beta);
    }
#endif

private:
    U8* alpha;
    U8* beta;
};

#endif //_SCALE_CPU_H
