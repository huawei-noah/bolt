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
#ifndef _FCELTWISE_CPU_H
#define _FCELTWISE_CPU_H

#include "weight_operator.hpp"
#include "tensor_computing.h"
#include "fully_connected_eltwise.hpp"

class FullyConnectedEltwiseCPU: public FullyConnectedEltwise {
public:
    FullyConnectedEltwiseCPU(DataType dt, U32 numInput, U32 numOutput, std::optional<EltwiseType> eltwiseType):
    FullyConnectedEltwise(dt, numInput, numOutput, eltwiseType) { }

    virtual EE init_weight_bias_from_model(U8** modelPtr) override
    {
        TensorDesc weightDesc = tensor2df(this->dt, DF_NORMAL, this->numOutput, this->numInput);
        TensorDesc biasDesc = tensor1d(this->dt, this->numOutput);

        std::shared_ptr<Tensor> modelWeightTensor(new Tensor());
        std::shared_ptr<Tensor> modelBiasTensor(new Tensor());
        modelWeightTensor->set_desc(weightDesc);
        modelBiasTensor->set_desc(biasDesc);

        auto curOpWs = this->get_weightspec_ptr();
        if(modelPtr != nullptr){
            modelWeightTensor->alloc();
            memcpy((U8*)modelWeightTensor->get_val(), *modelPtr, tensorNumBytes(weightDesc));
            *modelPtr += tensorNumBytes(weightDesc);
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

    TensorDesc desc_process(TensorDesc inDim) {
        TensorDesc inputDesc;
        DataType dt;
        DataFormat df;
        U32 in, ic, ih, iw;
        switch (inDim.nDims) {
            case 2: {
                CHECK_STATUS(tensor2dGet(inDim, &dt, &in, &(this->numInput)));
                inputDesc = inDim;
                break;
            }
            case 3: {
                CHECK_STATUS(tensor3dGet(inDim, &dt, &df, &in, &ih, &iw));
                this->numInput = iw;
                inputDesc = tensor2df(dt, DF_NORMAL, in*ih, iw);
                break;
            }
            case 4: {
                CHECK_STATUS(tensor4dGet(inDim, &dt, &df, &in, &ic, &ih, &iw));
                this->numInput = ic*ih*iw;
                inputDesc = inDim;
                break;
            }
            default:
                break;
        }
        return inputDesc;
    }

    TensorDesc desc_process_reverse(TensorDesc inDim, TensorDesc outDim) {
        TensorDesc outDesc;
        DataType dt;
        DataFormat df;
        U32 in, ih, iw;
        switch (inDim.nDims) {
            case 2: {
                outDesc = outDim;
                break;
            }
            case 3: {
                CHECK_STATUS(tensor3dGet(inDim, &dt, &df, &in, &ih, &iw));
                outDesc = tensor3df(dt, df, in, ih, this->numOutput);
                break;
            }
            case 4: {
                outDesc = outDim;
                break;
            }
            default:
                break;
        }
        return outDesc;
    }


    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc = desc_process(inputTensor.get_desc());

        Tensor weightTensor = this->weightTensors[0];
        TensorDesc weightDesc = weightTensor.get_desc();

        Tensor biasTensor = this->biasTensors[0];
        TensorDesc biasDesc = biasTensor.get_desc();

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = desc_process(outputTensor.get_desc());

        CHECK_STATUS(fully_connected(inputDesc, inputTensor.get_val(),
                                     weightDesc, weightTensor.get_val(),
                                     this->temp.get(), this->lenOfTemp,
                                     outputDesc, outputTensor.get_val(),
                                     biasDesc, biasTensor.get_val(), this->schedule));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inputDesc = desc_process(inDims[0]);
        TensorDesc weightDesc = tensor2df(this->dt, DF_NORMAL, this->numOutput, this->numInput);
        TensorDesc outputDesc;
        
        CHECK_STATUS(fully_connected_infer_output_size(inputDesc, weightDesc, &outputDesc, this->schedule));
        (*outDims)[0] = desc_process_reverse(inDims[0], outputDesc);
        return SUCCESS;
    }

    virtual U32 infer_tmp_memory_size() override
    {
        TensorDesc inputDesc = desc_process((this->inputTensors[0]).desc);
        TensorDesc filterDesc = (this->weightTensors[0]).desc;
        U32 bytes = 0;
        CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(inputDesc, filterDesc, &bytes, this->schedule));
        return bytes;
    }

    virtual U32 infer_wtm_memory_size() override
    {
        TensorDesc weightDesc = (this->weightTensors[0]).desc;
        U32 bytes = 0;
        CHECK_STATUS(fully_connected_transform_filter_bytes(weightDesc, &bytes, this->schedule));
        return bytes;
    }

    virtual EE transform_filter() override
    {
        this->wtm = std::shared_ptr<Tensor>(new Tensor());
        TensorDesc inputDesc = desc_process((this->inputTensors[0]).desc);

        Tensor weightTensor = this->weightTensors[0];
        TensorDesc weightDesc = weightTensor.get_desc();
        U8* weightPtr = weightTensor.get_val();


        auto wtm_bytes = this->infer_wtm_memory_size();
        std::shared_ptr<U8> wtmPtr((U8*) operator new(wtm_bytes));
        auto cpuMem = new CpuMemory();
        cpuMem->set_shared_ptr_caster(wtmPtr);
        Memory_* mem = (Memory_*)(cpuMem);
        std::shared_ptr<Memory_> memWtmPtr(mem);
        this->set_wtm_memory(wtm_bytes, memWtmPtr);

        TensorDesc wtmDesc;
        CHECK_STATUS(fully_connected_transform_filter(inputDesc, weightDesc, weightPtr, &wtmDesc, this->get_wtm()->get_val(), this->schedule));

        this->get_wtm()->set_desc(wtmDesc);
        this->weightTensors[0] = *this->get_wtm();
        return SUCCESS;
    }
};

#endif //_FCELTWISE_CPU_H
