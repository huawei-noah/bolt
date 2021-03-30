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
#include "model_spec.h"

class WeightOperator : public Operator {
public:
    WeightOperator()
    {
        this->hasBias = false;
        this->lenOfWtm = 0;

        this->ws.mdt = DT_U8;
        this->ws.bytes_of_weight = 0;
        this->ws.weight = nullptr;
        this->ws.bytes_of_vec = 0;
        this->ws.vec = nullptr;
    }

    bool is_weight() override
    {
        return true;
    }

    U32 get_weight_size()
    {
        U32 ret = 0;
        for (auto tensor : this->weightTensors) {
            TensorDesc desc = tensor.get_desc();
            ret += tensorNumBytes(desc);
        }
        return ret;
    }

    virtual void set_weight_tensors(std::vector<Tensor> weightTensors)
    {
        this->weightTensors = weightTensors;
    }

    virtual std::vector<Tensor> get_weight_tensors()
    {
        return this->weightTensors;
    }

    virtual void set_bias_tensors(std::vector<Tensor> biasTensors)
    {
        this->biasTensors = biasTensors;
    }

    virtual std::vector<Tensor> get_bias_tensors()
    {
        return this->biasTensors;
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

    virtual void set_wtm_memory(U32 len, Tensor wtm)
    {
        this->lenOfWtm = len;
        this->temp = wtm;
    }

    virtual U32 get_lenOfWtm()
    {
        return this->lenOfWtm;
    }

    virtual Tensor *get_wtm()
    {
        return this->wtm.get();
    }

    virtual void set_weightspec_ptr(WeightSpec ws)
    {
        this->ws = ws;
    }

    virtual WeightSpec get_weightspec()
    {
        return this->ws;
    }

    virtual void set_hasBias(bool hasBiasOrNot)
    {
        this->hasBias = hasBiasOrNot;
    }

    virtual EE infer_weight_desc()
    {
        return SUCCESS;
    }

    virtual EE init_weight_bias_from_model(std::shared_ptr<U8> *modelPtr)
    {
        EE ret = this->infer_weight_desc();
        if (ret != SUCCESS) {
            return ret;
        }
        auto curOpWs = this->get_weightspec();
        CpuMemory weight_mem_src, bias_mem_src;
        std::shared_ptr<U8> weight_ptr, bias_ptr;
        if (modelPtr != nullptr) {
            weight_ptr = *modelPtr;
            bias_ptr = *modelPtr;
        } else {
            weight_ptr = std::shared_ptr<U8>(curOpWs.weight, [](U8 *) {});
            bias_ptr = std::shared_ptr<U8>(curOpWs.vec, [](U8 *) {});
        }

        std::set<OperatorType> weightReuseSet = {OT_Conv, OT_FC, OT_Deconvolution, OT_RNN};
        U32 weight_offset = 0;
        for (auto weight_tensor : this->weightTensors) {
            TensorDesc desc = weight_tensor.get_desc();
            auto weight_mem_dst = weight_tensor.get_memory();
            weight_mem_src.resize(desc);
            weight_mem_src.set_shared_ptr(
                std::shared_ptr<U8>(weight_ptr, weight_ptr.get() + weight_offset));
            if (weightReuseSet.count(this->get_type())) {
                weight_mem_dst->reuse(&weight_mem_src);
            } else {
                weight_mem_dst->copy_from(&weight_mem_src);
            }
            weight_offset += tensorNumBytes(desc);
        }

        U32 bias_offset = (modelPtr != nullptr) ? weight_offset : 0;
        if (this->hasBias) {
            for (auto bias_tensor : this->biasTensors) {
                TensorDesc desc = bias_tensor.get_desc();
                auto bias_mem_dst = bias_tensor.get_memory();
                bias_mem_src.resize(desc);
                bias_mem_src.set_shared_ptr(
                    std::shared_ptr<U8>(bias_ptr, bias_ptr.get() + bias_offset));
                bias_mem_dst->copy_from(&bias_mem_src);
                bias_offset += tensorNumBytes(desc);
            }
        } else {
            for (auto bias_tensor : this->biasTensors) {
                TensorDesc desc = bias_tensor.get_desc();
                auto bias_mem_dst = bias_tensor.get_memory();
                bias_mem_src.resize(desc);
                bias_mem_src.alloc();
                U8 *tmp = (U8 *)bias_mem_src.get_ptr();
                memset(tmp, 0, bias_mem_src.bytes());
                bias_mem_dst->reuse(&bias_mem_src);
            }
        }
        if (modelPtr != nullptr) {
            *modelPtr = std::shared_ptr<U8>(bias_ptr, bias_ptr.get() + bias_offset);
        }
        return SUCCESS;
    }

    virtual EE transform_filter()
    {
        return SUCCESS;
    }

protected:
    std::vector<Tensor> weightTensors;
    std::vector<Tensor> biasTensors;
    bool hasBias;

    U32 lenOfWtm;
    std::shared_ptr<Tensor> wtm;
    WeightSpec ws;
};

#endif  // _WEIGHTOPERATOR_H
