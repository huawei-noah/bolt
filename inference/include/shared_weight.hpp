// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SHARED_WEIGHT_H
#define _SHARED_WEIGHT_H

#include "weight_operator.hpp"
#include "op_type.h"

class SharedWeight: public WeightOperator
{
public:
    /**
    @param mode
    */
    SharedWeight(DataType dt, TensorDesc desc)
    {
        this->dt = dt;
        this->desc = desc;
    }

    OperatorType get_op_type() override
    {
        return OT_SharedWeight;
    }

    EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        UNUSED(inDims);
        (*outDims)[0] = this->desc;
        return SUCCESS;
    }

    void run() override { }

    EE init_weight_bias_from_model(U8** modelPtr)
    {
        TensorDesc weightDesc = this->desc;
        U32 weightBytes = tensorNumBytes(weightDesc);

        std::shared_ptr<Tensor> modelWeightTensor(new Tensor());
        modelWeightTensor->set_desc(weightDesc);

        if(modelPtr != nullptr){
            modelWeightTensor->alloc();
            memcpy((U8*)modelWeightTensor->get_val(), *modelPtr, weightBytes);
            *modelPtr += weightBytes;
        } else {
            auto curOpWs = this->get_weightspec_ptr();
            modelWeightTensor->set_val(curOpWs.weight);
        }
        this->weightTensors.push_back(*modelWeightTensor.get());
        return SUCCESS;
    }

private:
    TensorDesc desc;
};

#endif //_WEIGHT_H
