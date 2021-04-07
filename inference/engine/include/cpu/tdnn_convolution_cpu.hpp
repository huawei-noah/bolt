// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _TDNN_CONVOLUTION_CPU_H
#define _TDNN_CONVOLUTION_CPU_H

#include "convolution_cpu.hpp"

class TdnnConvolutionCPU : public ConvolutionCPU {
public:
    TdnnConvolutionCPU(DataType dt, TdnnParamSpec tdnnParam)
        : ConvolutionCPU(dt, ConvolutionParamSpec{}, ActivationParamSpec{}, ActivationParamSpec{})
    {
        this->tdnn = tdnnParam;
        int dilation = 1;
        if (this->tdnn.num_context >= 2) {
            dilation = this->tdnn.context[1] - this->tdnn.context[0];
        }
        for (int i = 2; i < this->tdnn.num_context; i++) {
            if (this->tdnn.context[i] - this->tdnn.context[i - 1] != dilation) {
                UNI_ERROR_LOG("TdnnCPU currently not support time context is non arithmetic "
                              "sequence\n");
            }
            if (this->tdnn.context[i] < this->tdnn.context[i - 1]) {
                UNI_ERROR_LOG("TdnnCPU currently not support time context is decreasing order\n");
            }
        }
        ConvolutionMode convMode = Convolution_Pointwise;
        if (dilation > 1) {
            convMode = Convolution_Dilation;
        }
        this->p = createConvolutionParamSpec(1, 1, this->tdnn.num_context, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 1, dilation, 1, this->tdnn.num_outputs, convMode);
        this->dwActivationParamSpec.mode = ACTIVATION_NULL;
        this->pwActivationParamSpec.mode = tdnnParam.activation_type;
    }

    OperatorType get_type() override
    {
        return OT_Tdnn;
    }

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<TdnnConvolutionCPU> mem =
            std::shared_ptr<TdnnConvolutionCPU>(new TdnnConvolutionCPU(this->dt, this->tdnn));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        TensorDesc desc = this->inputTensors[0].get_desc();
        bool nhwc = is_nhwc(desc);
        if (nhwc) {
            TensorDesc tmpInputDesc = this->nhwc_desc_process(desc);
            this->inputTensors[0].resize(tmpInputDesc);
        }
        ConvolutionCPU::run();
        if (nhwc) {
            this->inputTensors[0].resize(desc);
        }
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        TensorDesc desc = this->inputTensors[0].get_desc();
        bool nhwc = is_nhwc(desc);
        if (nhwc) {
            TensorDesc tmpInputDesc = this->nhwc_desc_process(desc);
            this->inputTensors[0].resize(tmpInputDesc);
        }
        EE ret = ConvolutionCPU::infer_forward_algorithm(algorithmMap);
        if (nhwc) {
            this->inputTensors[0].resize(desc);
        }
        return ret;
    }

    U32 infer_tmp_memory_size() override
    {
        TensorDesc desc = this->inputTensors[0].get_desc();
        bool nhwc = is_nhwc(desc);
        if (nhwc) {
            TensorDesc tmpInputDesc = this->nhwc_desc_process(desc);
            this->inputTensors[0].resize(tmpInputDesc);
        }
        U32 bytes = ConvolutionCPU::infer_tmp_memory_size();
        if (nhwc) {
            this->inputTensors[0].resize(desc);
        }
        return bytes;
    }

    bool is_nhwc(TensorDesc desc)
    {
        bool ret = false;
        if (desc.nDims == 3 && (desc.df == DF_NCHW || desc.df == DF_NHWC || desc.df == DF_MTK)) {
            ret = true;
        }
        return ret;
    }

    TensorDesc nhwc_desc_process(TensorDesc desc)
    {
        TensorDesc ret;
        if (is_nhwc(desc)) {
            ret = tensor4df(desc.dt, DF_NCHW, desc.dims[2], desc.dims[0], desc.dims[1], 1);
        } else {
            ret = desc;
        }
        return ret;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        Tensor tmpInputTensor = *inTensors[0];
        TensorDesc desc = inTensors[0]->get_desc();
        // first tdnn operator is NHWC format, but represent in NCHW format.
        TensorDesc tmpInputDesc = this->nhwc_desc_process(desc);
        tmpInputTensor.resize(tmpInputDesc);
        if (tmpInputDesc.nDims < 3) {
            return NOT_SUPPORTED;
        }
        tmpInputTensor.resize(tmpInputDesc);
        if (this->weightTensors.size() == 0) {
            this->weightTensors = std::vector<Tensor>(1);
            int num_inputs = tmpInputDesc.dims[tmpInputDesc.nDims - 2];
            this->weightTensors[0].resize(tensor4df(
                this->dt, DF_NCHW, this->tdnn.num_outputs, num_inputs, this->tdnn.num_context, 1));
        }
        if (this->biasTensors.size() == 0) {
            this->biasTensors = std::vector<Tensor>(1);
            this->biasTensors[0].resize(tensor1d(this->dt, this->tdnn.num_outputs));
        }
        CHECK_STATUS(convolution_infer_output_size(&tmpInputTensor, this->weightTensors[0], this->p,
            outTensors[0], this->dt, &this->archInfo));
        return SUCCESS;
    }

private:
    TdnnParamSpec tdnn;
};

#endif  // _TDNN_CONVOLUTION_CPU_H
