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
#include "tensor_computing.h"
#include "utils.hpp"

template<Arch A>
class WeightOperator: public Operator<A> {
public:
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
        this->wtm = std::shared_ptr<U8>();
        return 0;
    }

    virtual void set_wtm_memory(U32 len, std::shared_ptr<U8> wtm)
    {
        this->lenOfWtm = len;
        this->wtm = wtm;
    }

    virtual U32 get_lenOfWtm()
    {
        return this->lenOfWtm;
    }

    virtual std::shared_ptr<U8> get_wtm()
    {
        return this->wtm;
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
    std::shared_ptr<U8> wtm;
    WeightSpec ws;
};

#endif //_WEIGHTOPERATOR_H
