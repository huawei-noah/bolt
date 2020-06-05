// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _WEIGHTOPERATOR_H
#define _WEIGHTOPERATOR_H

#include "operator.hpp"
#include "utils.hpp"
#include "model_tools.h"

class WeightOperator: public Operator {
public:
    WeightOperator() {
        hasBias = false;
        lenOfWtm = 0;

        ws.mdt = DT_U8;
        ws.bytes_of_weight = 0;
        ws.weight = nullptr;
        ws.bytes_of_vec = 0;
        ws.vec = nullptr;
    }

    virtual bool is_weight() override { 
        return true; 
    }
    /**
     * @param weightTensors
     */
    U32 get_weight_size()
    {
        U32 ret = 0;
        for(auto tensor : weightTensors) {
            auto dim = tensor.get_desc();
            ret += tensorNumElements(dim) * bytesOf(dim.dt);
        }
        return ret;
    }

    virtual void set_weightTensors(Vec<Tensor> weightTensors)
    {
        this->weightTensors = weightTensors;
    }

    virtual void set_biasTensors(Vec<Tensor> biasTensors)
    {
        this->biasTensors = biasTensors;
    }

    virtual U32 infer_wtm_memory_size()
    {
        this->lenOfWtm = 0;
        this->wtm = std::shared_ptr<Tensor>();
        return 0;
    }
#ifdef _USE_MALI
    virtual GCLMemDesc infer_wtm_memory_size_mali()
    {
        this->lenOfWtm = 0;
        this->wtm = std::shared_ptr<Tensor>();
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        GCLMemDesc tmpdesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        return tmpdesc;
    }
#endif

    virtual void set_wtm_memory(U32 len, std::shared_ptr<Memory_> wtm)
    {
        this->lenOfWtm = len;
        this->wtm->set_memory(wtm);
    }

    virtual U32 get_lenOfWtm()
    {
        return this->lenOfWtm;
    }

    virtual Tensor* get_wtm()
    {
        return this->wtm.get();
    }

    virtual void set_weightspec_ptr(WeightSpec ws)
    {
        this->ws = ws;   // struct can not directly assigned to another?
    }

    virtual WeightSpec get_weightspec_ptr()
    {
        return ws;
    }

    virtual void set_hasBias(bool hasBiasOrNot) {
        this->hasBias = hasBiasOrNot;
    }

public:
    Vec<Tensor> weightTensors;
    Vec<Tensor> biasTensors;
    bool hasBias;

    U32 lenOfWtm;
    std::shared_ptr<Tensor> wtm;
    WeightSpec ws;
};

#endif //_WEIGHTOPERATOR_H
