// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _SHARED_WEIGHT_OCL_H
#define _SHARED_WEIGHT_OCL_H

#include "shared_weight.hpp"

#include "ocl_desc_trans.h"
#include "ocl_data_trans.h"

class SharedWeightOCL : public SharedWeight {
public:
    SharedWeightOCL(DataType dt,
        TensorDesc desc,
        std::string outputTensorName,
        std::map<std::string, std::shared_ptr<Tensor>> *tensorMapPtr)
        : SharedWeight(dt, desc, outputTensorName, tensorMapPtr)
    {
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
    }

    ~SharedWeightOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<SharedWeightOCL> mem = std::shared_ptr<SharedWeightOCL>(
            new SharedWeightOCL(this->dt, this->desc, this->outputTensorName, tensorMapPtr));
        *mem = *this;
        return mem;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        UNUSED(inTensors);
        outTensors[0]->resize(this->desc);
        DataFormat df;
        DataType dt;
        U32 n, c, h, w;
        tensorSelectGet(this->desc, &dt, &df, &n, &c, &h, &w);
        U32 s0, s1, s2;
        s0 = w;
        s1 = h;
        s2 = c * n;
        U32 stride[3] = {s0, s1, s2};
        U32 offset[3] = {0, 0, 0};
        GCLMemType mt = GCL_MEM_BUF;
        MemFlags flags = CL_MEM_READ_WRITE;
        GCLMemDesc gclMemDesc = gclmem_build_desc();
        DataFormat mf = (df == DF_NHWC) ? DF_NHWC : DF_NCHW;
        CHECK_STATUS(gclmem_set_desc_padding(&gclMemDesc, stride, offset, dt, mf, mt, flags));
        ocl_set_desc(outTensors[0], gclMemDesc);
        return SUCCESS;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
    }

    EE init_weight_bias_from_model(std::shared_ptr<U8> *modelPtr) override
    {
        auto dstTensor = (*this->tensorMapPtr)[this->outputTensorName];
        auto dstMem = (OclMemory *)(dstTensor->get_memory());
        GCLMemDesc dstMemDesc = dstMem->get_desc();
        std::shared_ptr<U8> weight_ptr;
        auto curOpWs = this->get_weightspec();
        if (modelPtr) {
            weight_ptr = *modelPtr;
        } else {
            weight_ptr = std::shared_ptr<U8>(curOpWs.weight, [](U8 *) {});
        }
        U32 n, c, h, w;
        U32 s0, s1, s2;
        tensorSelectGet(this->desc, NULL, NULL, &n, &c, &h, &w);
        s0 = w;
        s1 = h;
        s2 = c * n;
        this->needTrans = false;
        if (dstMemDesc.stride[0] == s0 && dstMemDesc.stride[1] == s1 && dstMemDesc.stride[2] == s2) {
            CpuMemory weight_mem_src;
            weight_mem_src.resize(this->desc);
            weight_mem_src.set_shared_ptr(std::shared_ptr<U8>(weight_ptr));
            dstMem->copy_from((Memory *)&weight_mem_src);
        } else {
            this->needTrans = true;
            this->host_ptr = weight_ptr;
        }
        this->weightTensors.push_back(*dstTensor.get());
        if (modelPtr) {
            *modelPtr =
                std::shared_ptr<U8>(*modelPtr, (*modelPtr).get() + tensorNumBytes(this->desc));
        }
        return SUCCESS;
    }

    EE transform_filter() override
    {
        if (needTrans) {
            auto dstTensor = (*this->tensorMapPtr)[this->outputTensorName];
            auto dstMem = (OclMemory *)(dstTensor->get_memory());
            dstMem->alloc();
            GCLMem_t dst = (GCLMem_t)dstMem->get_ptr();
            auto tempMem = (OclMemory *)(this->temp.get_memory());
            GCLMem_t temp = (GCLMem_t)tempMem->get_ptr();
            CHECK_STATUS(ocl_set_input(OCLContext::getInstance().handle.get(), dst, this->desc,
                host_ptr.get(), temp, true));
            this->weightTensors[0] = *dstTensor.get();
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        if (needTrans) {
            bytes = tensorNumBytes(this->desc);
        }
        return bytes;
    }

    REGISTER_OCL_OPERATOR_RUN

private:
    std::shared_ptr<U8> host_ptr;
    bool needTrans;
};

#endif  // _WEIGHT_OCL_H
