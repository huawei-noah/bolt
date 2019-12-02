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
#ifndef _FCELTWISE_H
#define _FCELTWISE_H

#include <optional>
#include "weight_operator.hpp"
#include "tensor_computing.h"


template<Arch A>
class FullyConnectedEltwise: public WeightOperator<A> {
public:
    FullyConnectedEltwise(DataType dt, U32 numOutput, std::optional<EltwiseType> eltwiseType)
    {
        this->dt = dt;
        this->numOutput = numOutput;
        this->eltwiseType = eltwiseType;
        this->set_op_type(OT_FC);
        this->hasBias = false;
    }

    FullyConnectedEltwise(U32 numOutput)
    {
        std::optional<EltwiseType> etNull;
        FullyConnectedEltwise(numOutput, etNull);
    }

    EE init_weight_bias_from_model(U8** modelPtr)
    {
        TensorDesc weightDesc = tensor2df(this->dt, DF_NORMAL, this->numOutput, this->fcInputInfo);
        TensorDesc biasDesc = tensor1d(this->dt, this->numOutput);

        U8* modelWeightPtr = nullptr;
        U8* modelBiasPtr = nullptr;
        if (modelPtr != nullptr) {
            modelWeightPtr = (U8*)operator new(tensorNumBytes(weightDesc));
            memcpy(modelWeightPtr, *modelPtr, tensorNumBytes(weightDesc));
            *modelPtr += tensorNumBytes(weightDesc);
            if (this->hasBias) {
                modelBiasPtr = (*modelPtr);
                *modelPtr += tensorNumBytes(biasDesc);
            }
        }
        else {
            auto curOpWs = this->get_weightspec_ptr();
            modelWeightPtr = curOpWs.weight;
            modelBiasPtr = curOpWs.vec;
        }

        std::shared_ptr<U8> weightVal(modelWeightPtr);
        Tensor weightTensor = Tensor(weightDesc, weightVal);
        this->weightTensors.push_back(weightTensor);

        // bias
        std::shared_ptr<U8> biasVal;
        Tensor biasTensor = Tensor(biasDesc, biasVal);
        biasTensor.alloc();
        U8* biasPtr = biasTensor.get_val().get();
        if (this->hasBias) {
            memcpy(biasPtr, modelBiasPtr, tensorNumBytes(biasDesc));
        } else {
            memset(biasPtr, 0, tensorNumBytes(biasDesc));
        }
        this->biasTensors.push_back(biasTensor);

        return SUCCESS;
    }

    TensorDesc desc_process(TensorDesc inDim) {
        TensorDesc inputDesc;
        DataType dt;
        DataFormat df;
        U32 in, ic, ih, iw;
        switch (inDim.nDims) {
            case 2: {
                CHECK_STATUS(tensor2dGet(inDim, &dt, &in, &(this->fcInputInfo)));
                inputDesc = inDim;
                break;
            }
            case 3: {
                CHECK_STATUS(tensor3dGet(inDim, &dt, &df, &in, &ih, &iw));
                this->fcInputInfo = iw;
                inputDesc = tensor2df(dt, DF_NORMAL, in*ih, iw);
                break;
            }
            case 4: {
                CHECK_STATUS(tensor4dGet(inDim, &dt, &df, &in, &ic, &ih, &iw));
                this->fcInputInfo = ic*ih*iw;
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


    void run() override
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

        //NOTE: no clean tmp and output
        CHECK_STATUS(fully_connected(inputDesc, inputTensor.get_val().get(),
                                     weightDesc, weightTensor.get_val().get(),
                                     this->temp.get(), this->lenOfTemp,
                                     outputDesc, outputTensor.get_val().get(),
                                     biasDesc, biasTensor.get_val().get(), A));

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inputDesc = desc_process(inDims[0]);
        TensorDesc weightDesc = tensor2df(this->dt, DF_NORMAL, this->numOutput, this->fcInputInfo);
        TensorDesc outputDesc;
        
        CHECK_STATUS_WITH_RETURN(fully_connected_infer_output_size(inputDesc, weightDesc, &outputDesc));
        (*outDims)[0] = desc_process_reverse(inDims[0], outputDesc);
        return SUCCESS;
    }

    U32 infer_tmp_memory_size()
    {
        TensorDesc inputDesc = desc_process((this->inputTensors[0]).desc);
        TensorDesc filterDesc = (this->weightTensors[0]).desc;
        U32 bytes = 0;
        CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(inputDesc, filterDesc, &bytes, A));
        return bytes;
    }

    //TODO 0823 maybe put into mt one day
    U32 infer_wtm_memory_size() 
    {
        TensorDesc weightDesc = (this->weightTensors[0]).desc;
        U32 bytes = 0;
        CHECK_STATUS(fully_connected_transform_filter_bytes(weightDesc, &bytes));
        return bytes;
    }

    EE transform_filter()
    {
        TensorDesc inputDesc = desc_process((this->inputTensors[0]).desc);

        Tensor weightTensor = this->weightTensors[0];
        TensorDesc weightDesc = weightTensor.get_desc();
        U8* weightPtr = weightTensor.get_val().get();

        auto wtm_bytes = this->infer_wtm_memory_size();
        std::shared_ptr<U8> wtmPtr((U8*) operator new(wtm_bytes));
        this->set_wtm_memory(wtm_bytes, wtmPtr);

        TensorDesc wtmDesc;
        CHECK_STATUS_WITH_RETURN(fully_connected_transform_filter(inputDesc,
                                                                  weightDesc, weightPtr,
                                                                  &wtmDesc, this->get_wtm().get()));
        Tensor wtmTensor = Tensor(wtmDesc, this->get_wtm());
        this->weightTensors[0] = wtmTensor;

        return SUCCESS;
    }


public:
    U32 fcInputInfo;
    U32 numOutput;
    std::optional<EltwiseType> eltwiseType;
};

#endif //_FCELTWISE_H
