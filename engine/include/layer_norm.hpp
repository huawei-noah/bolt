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
#include "model_tools.h"

template <Arch A>
class LayerNorm: public WeightOperator<A> {
public:
    LayerNorm(DataType dt){
        this->dt = dt;
        this->set_op_type(OT_LayerNorm);
        this->hasBias = false;
    }

    EE init_weight_bias_from_model(U8** modelPtr){
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        TensorDesc weightDesc = tensor1d(this->dt, inputDesc.dims[0]);    // attention 1d
        TensorDesc biasDesc = tensor1d(this->dt, inputDesc.dims[0]);

        U8* modelWeightPtr = nullptr;
        U8* modelBiasPtr = nullptr;
        if (modelPtr != nullptr) {
            modelWeightPtr = (U8 *)operator new(tensorNumBytes(weightDesc));
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

    void run() override 
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor weightTensor = this->weightTensors[0];
        Tensor biasTensor = this->biasTensors[0];

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        CHECK_STATUS(layer_normalization(weightTensor.get_val().get(),
                                         biasTensor.get_val().get(),
                                         inputDesc, inputTensor.get_val().get(),
                                         outputDesc, outputTensor.get_val().get(), A));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override {
        TensorDesc in_dim = inDims[0];
        CHECK_STATUS_WITH_RETURN(normalization_infer_output_size(in_dim, &((*outDims)[0])));
        return SUCCESS;
    }
};

#endif //_LAYER_NORM_H
