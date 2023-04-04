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
#include "model_spec.h"

class WeightOperator : public Operator {
public:
    WeightOperator()
    {
        UNI_MEMSET(&(this->ws), 0, sizeof(WeightSpec));
        this->ws.mdt = DT_NUM;
    }

    bool is_weight() override
    {
        bool ret = (this->ws.mdt == DT_NUM) ? false : true;
        return ret;
    }

    void set_weightspec(const WeightSpec &ws)
    {
        this->ws = ws;
    }

    virtual EE infer_weight_desc()
    {
        return SUCCESS;
    }

    virtual EE init_weight_bias_from_model(std::shared_ptr<U8> *modelPtr = nullptr)
    {
        EE ret = this->infer_weight_desc();
        if (ret != SUCCESS) {
            return ret;
        }
        CpuMemory weight_mem_src, bias_mem_src;
        std::shared_ptr<U8> weight_ptr, bias_ptr;
        if (modelPtr != nullptr) {
            weight_ptr = *modelPtr;
            bias_ptr = *modelPtr;
        } else {
            weight_ptr = std::shared_ptr<U8>(this->ws.weight, [](U8 *) {});
            bias_ptr = std::shared_ptr<U8>(this->ws.vec, [](U8 *) {});
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
            update_desc_from_tensor(&weight_tensor);
        }

        if (this->ws.num_quant_scale >= this->weightTensors.size()) {
            for (U32 i = 0; i < this->weightTensors.size(); ++i) {
                if (this->ws.weight_scale[i].num_scale > 0) {
                    this->weightTensors[i].set_scale_ptr(
                        std::shared_ptr<F32>(this->ws.weight_scale[i].scale, [](F32 *) {}));
                }
            }
        } else if (this->ws.num_quant_scale == 1) {
            for (U32 i = 0; i < this->weightTensors.size(); ++i) {
                if (this->ws.weight_scale[0].num_scale > 0) {
                    this->weightTensors[i].set_scale_ptr(
                        std::shared_ptr<F32>(this->ws.weight_scale[0].scale, [](F32 *) {}));
                }
            }
        }

        U32 bias_offset = (modelPtr != nullptr) ? weight_offset : 0;
        for (auto bias_tensor : this->biasTensors) {
            TensorDesc desc = bias_tensor.get_desc();
            auto bias_mem_dst = bias_tensor.get_memory();
            bias_mem_src.resize(desc);
            if (this->ws.bytes_of_vec > 0) {
                bias_mem_src.set_shared_ptr(
                    std::shared_ptr<U8>(bias_ptr, bias_ptr.get() + bias_offset));
                bias_mem_dst->copy_from(&bias_mem_src);
                bias_offset += tensorNumBytes(desc);
            } else {
                bias_mem_src.alloc();
                U8 *tmp = (U8 *)bias_mem_src.get_ptr();
                UNI_MEMSET(tmp, 0, bias_mem_src.bytes());
                bias_mem_dst->reuse(&bias_mem_src);
            }
            update_desc_from_tensor(&bias_tensor);
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

#ifdef _USE_GPU
protected:
    EE set_wtm_image(TensorDesc desc, std::shared_ptr<Tensor> *targetWtm = nullptr)
    {
        if (IS_QUALCOMM_GPU(this->archInfo.arch)) {
            std::shared_ptr<Tensor> tensor = std::shared_ptr<Tensor>(new Tensor(OCLMemImg));
            tensor->resize(desc);
            U32 str[3];
            auto mem = (OclMemoryImg *)(tensor->get_memory());
            mem->stride(str);
            if (gcl_check_meet_device_image3d_limits(
                    OCLContext::getInstance().handle.get(), str[0], str[1], str[2])) {
                if (targetWtm) {
                    *targetWtm = tensor;
                } else {
                    this->wtm = tensor;
                }
            }
        }
        return SUCCESS;
    }
    std::shared_ptr<Tensor> wtm;
#endif

protected:
    WeightSpec ws;
    std::vector<Tensor> weightTensors;
    std::vector<Tensor> biasTensors;
};

#endif  // _WEIGHTOPERATOR_H
