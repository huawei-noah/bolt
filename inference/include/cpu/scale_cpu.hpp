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
    ScaleCPU(DataType dt, int numChannels, int numSource) :
    Scale(dt, numChannels, numSource)
    {
        this->alpha = nullptr;
        this->beta = nullptr;
    }

    virtual EE init_weight_bias_from_model(U8** modelPtr) override
    {
        auto curOpWs = this->get_weightspec_ptr();
        if(modelPtr == nullptr){
            this->numChannels = curOpWs.bytes_of_weight / UNI_MAX(1, bytesOf(curOpWs.mdt));
        }

        TensorDesc weightDesc = tensor1d(this->dt, this->numChannels);
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

        if (biasVal) {
            modelBiasTensor->set_val(biasVal);
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
        Tensor dataTensor = this->inputTensors[this->dataID];;
        TensorDesc inputDesc = dataTensor.get_desc();
        U8* dataPtr = dataTensor.get_val();

        if (inputTensorNumber == 1) {
            this->alpha = this->weightTensors[0].get_val();
            this->beta = this->biasTensors[0].get_val();
#ifdef _USE_FP16
            if (0 != this->numChannels) {  // Assume some padding has been done to the source tensors
                this->from_nchwc8_to_nchw(&inputDesc, (F16*)dataPtr);
                this->from_nchw_to_nchwc8_pad_removed(&inputDesc, (F16*)dataPtr);
            }
#endif
            CHECK_STATUS(scale(this->alpha, this->beta, inputDesc, dataPtr, inputDesc, NULL, this->schedule));
        } else {
            CHECK_STATUS(scale(this->inputTensors[1-this->dataID].get_val(), nullptr, inputDesc, dataPtr, inputDesc, NULL, this->schedule));  // alpha/beta/inputDesc/data
        }

        Tensor outputTensor = this->outputTensors[0];
        U8* outputPtr = outputTensor.get_val();

        if(dataPtr != outputPtr) {
            memcpy(outputPtr, dataPtr, tensorNumBytes(inputDesc));
        }
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override 
    {
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;    
        CHECK_REQUIREMENT(tensorIs4d(inDims[0]));
        CHECK_STATUS(tensor4dGet(inDims[0], &idt, &idf, &in, &ic, &ih, &iw));

        auto curOpWs = this->get_weightspec_ptr();
        this->alpha = curOpWs.weight;
        this->beta = curOpWs.vec;
        U32 numChannels = curOpWs.bytes_of_weight / UNI_MAX(1, bytesOf(curOpWs.mdt));

        TensorDesc inputDesc;
        if (ic != numChannels && 0 != numChannels) {
            std::cout << "[Warning] ScaleCPU input channels (IC) do not match. Assume some channel padding has been done earlier.\n";
            std::cout << "IC is now " << ic << " but should be " << numChannels << ". \n";
            CHECK_REQUIREMENT(numChannels % 8 == 0);
            this->numChannels = numChannels / numSource;
            std::cout << "Retrieving the starting " << this->numChannels << " channels from " << this->numSource << " equal portions.\n";
            inputDesc = tensor4df(idt, idf, in, numChannels, ih, iw);
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
#ifdef _USE_FP16
    EE from_nchwc8_to_nchw(TensorDesc *desc, F16 *data)
    {
        if (desc == nullptr || data == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }

        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        CHECK_STATUS(tensor4dGet(*desc, &idt, &idf, &in, &ic, &ih, &iw));
        if (idf != DF_NCHWC8) {
            CHECK_STATUS(NOT_MATCH);
        }

        *desc = tensor4df(idt, DF_NCHW, in, ic, ih, iw);

        F16 *tmp = (F16 *)malloc(tensorNumBytes(*desc));
        ic /= 8;
        for (U32 n = 0; n < in; n++) {
            for (U32 c = 0; c < ic; c++) {
                for (U32 hw = 0; hw < ih*iw; hw++) {
                    for (U32 c8 = 0; c8 < 8; c8++) {
                        tmp[n*ic*8*ih*iw + (c*8 + c8)*ih*iw + hw] = data[n*ic*ih*iw*8 + c*ih*iw*8 + hw*8 + c8];
                    }
                }
            }
        }
        memcpy(data, tmp, tensorNumBytes(*desc));
        free(tmp);
        return SUCCESS;
    }

    EE from_nchw_to_nchwc8_pad_removed(TensorDesc *desc, F16 *data)
    {
        if (desc == nullptr || data == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }

        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        CHECK_STATUS(tensor4dGet(*desc, &idt, &idf, &in, &ic, &ih, &iw));
        if (idf != DF_NCHW) {
            CHECK_STATUS(NOT_MATCH);
        }

        U32 padding = 8 - (this->numChannels % 8);
        U32 perPadded = this->numChannels + padding;
        *desc = tensor4df(idt, DF_NCHWC8, in, this->numChannels * this->numSource, ih, iw);

        F16 *tmp = (F16 *)malloc(tensorNumBytes(*desc));
        for (U32 n = 0; n < in; n++) {
            for (U32 c = 0; c < ic; c++) {
                if (c % perPadded >= this->numChannels) {
                    continue;
                }

                U32 actualC = (c / perPadded) * this->numChannels + c % perPadded;
                U32 o = actualC / 8;
                U32 c8 = actualC % 8;
                for (U32 hw = 0; hw < ih*iw; hw++) {
                    tmp[n*ic*ih*iw + o*ih*iw*8 + hw*8 + c8] = data[n*ic*ih*iw + c*ih*iw + hw];
                }
            }
        }
        memcpy(data, tmp, tensorNumBytes(*desc));
        free(tmp);
        return SUCCESS;
    }
#endif
    U8* alpha;
    U8* beta;
};

#endif //_SCALE_CPU_H
