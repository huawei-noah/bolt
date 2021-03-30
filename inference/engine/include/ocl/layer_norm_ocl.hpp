// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _LAYER_NORM_OCL_H
#define _LAYER_NORM_OCL_H

#include "layer_norm.hpp"

class LayerNormOCL : public LayerNorm {
public:
    LayerNormOCL(DataType dt, U32 weightNum) : LayerNorm(dt, weightNum)
    {
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
    }

    ~LayerNormOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<LayerNormOCL> mem =
            std::shared_ptr<LayerNormOCL>(new LayerNormOCL(this->dt, this->weightNum));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        auto curOpWs = this->get_weightspec();
        if (0 != curOpWs.bytes_of_weight) {
            this->weightNum = curOpWs.bytes_of_weight / bytesOf(curOpWs.mdt);
        }
        DataType dtNoQ = (DT_F16_8Q == this->dt) ? DT_F16 : this->dt;
        TensorDesc weightDesc = tensor1d(dtNoQ, this->weightNum);
        TensorDesc biasDesc = tensor1d(dtNoQ, this->weightNum);
        Tensor modelWeightTensor = Tensor(OCLMem);
        Tensor modelBiasTensor = Tensor(OCLMem);
        auto weightMem = (OclMemory *)modelWeightTensor.get_memory();
        auto vectorMem = (OclMemory *)modelBiasTensor.get_memory();
        modelWeightTensor.resize(weightDesc);
        modelBiasTensor.resize(biasDesc);
        U32 stride[3] = {(this->weightNum + 3) / 4 * 4, 1, 1};
        U32 offset[3] = {0, 0, 0};
        GCLMemType mt = GCL_MEM_BUF;
        MemFlags flags = CL_MEM_READ_WRITE;
        GCLMemDesc desc = gclmem_build_desc();
        CHECK_STATUS(gclmem_set_desc_padding(&desc, stride, offset, dtNoQ, DF_NCHW, mt, flags));
        weightMem->padding(desc);
        vectorMem->padding(desc);
        this->weightTensors.push_back(modelWeightTensor);
        this->biasTensors.push_back(modelBiasTensor);
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        Tensor inputTensor = this->inputTensors[0];
        U32 bytes = 0;
        CHECK_STATUS(normalization_infer_forward_tmp_bytes(inputTensor, &bytes, &this->archInfo));
        return bytes;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor inputTensor = this->inputTensors[0];
        Tensor weightTensor = this->weightTensors[0];
        Tensor biasTensor = this->biasTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        CHECK_STATUS(layer_normalization(
            inputTensor, weightTensor, biasTensor, this->temp, outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        CHECK_STATUS(normalization_infer_output_size(inTensors[0], outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _LAYER_NORM_OCL_H
