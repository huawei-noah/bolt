// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _DECONVOLUTION_CPU_H
#define _DECONVOLUTION_CPU_H

#include "deconvolution.hpp"

class DeconvolutionCPU : public Deconvolution {
public:
    DeconvolutionCPU(DataType dt, ConvolutionParamSpec p, ActivationParamSpec activationDesc)
        : Deconvolution(dt, p, activationDesc)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<DeconvolutionCPU> mem = std::shared_ptr<DeconvolutionCPU>(
            new DeconvolutionCPU(this->dt, this->p, this->activationDesc));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        auto curOpWs = this->get_weightspec();
        DataType fdt = curOpWs.mdt;
        if (curOpWs.weight == nullptr) {
            fdt = this->dt;
        }
        if (fdt == DT_BIN01 || fdt == DT_BIN11) {
            return NOT_MATCH;
        }
        TensorDesc filterTensorDesc = tensor4df(
            fdt, DF_NCHW, this->numInputs, this->p.num_outputs, this->p.kernel_h, this->p.kernel_w);
        // bias data type should be the same as input and output
        DataType dtNoQ = (dt == DT_F16_8Q) ? DT_F16 : ((dt == DT_F32_8Q) ? DT_F32 : dt);
        TensorDesc vectorTensorDesc = tensor1d(dtNoQ, this->numInputs * this->p.group);
        this->weightTensors = std::vector<Tensor>(1);
        this->weightTensors[0].resize(filterTensorDesc);
        this->biasTensors = std::vector<Tensor>(1);
        this->biasTensors[0].resize(vectorTensorDesc);
        return SUCCESS;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc oriInputDesc = inputTensor.get_desc();
        inputTensor.resize(transformDescTo4d(oriInputDesc));
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc oriOutputDesc = outputTensor.get_desc();
        outputTensor.resize(transformDescTo4d(oriOutputDesc));
        Tensor filterTensor = this->weightTensors[0];
        Tensor biasTensor = this->biasTensors[0];
        CHECK_STATUS(deconvolution(inputTensor, filterTensor, p, this->alg, nullptr, biasTensor,
            this->temp, outputTensor, this->activationDesc, &this->archInfo));
        inputTensor.resize(oriInputDesc);
        outputTensor.resize(oriOutputDesc);
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc oriInputDesc = inputTensor.get_desc();
        inputTensor.resize(transformDescTo4d(oriInputDesc));
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc oriOutputDesc = outputTensor.get_desc();
        outputTensor.resize(transformDescTo4d(oriOutputDesc));
        ConvolutionPolicy policy = CONVOLUTION_FASTEST;
        auto filterDesc = this->weightTensors[0].get_desc();
        DataType targetType = filterDesc.dt;
        I32 algo;
        if (algorithmMap->getAlgorithmInfoFromMap(this->name, &algo, 1)) {
            this->alg = (ConvolutionForwardAlgorithm)algo;
        } else {
            CHECK_STATUS(deconvolution_infer_forward_algorithm(inputTensor, this->weightTensors[0],
                outputTensor, p, policy, &(this->alg), targetType, this->activationDesc,
                &this->archInfo));
            algo = this->alg;
            algorithmMap->setAlgorithmInfoToMap(this->name, &algo, 1);
        }
        inputTensor.resize(oriInputDesc);
        outputTensor.resize(oriOutputDesc);
        return SUCCESS;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        TensorDesc inDim = transformDescTo4d(inTensors[0]->get_desc());
        Tensor tmpTensor;
        tmpTensor.resize(inDim);
        auto inputTensor = &tmpTensor;
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        CHECK_STATUS(tensor4dGet(inDim, &idt, &idf, &in, &ic, &ih, &iw));
        this->numInputs = ic / this->p.group;

        Tensor filterTensor;
        TensorDesc filterDim = tensor4df(this->dt, DF_NCHW, this->numInputs, this->p.num_outputs,
            this->p.kernel_h, this->p.kernel_w);
        filterTensor.resize(filterDim);

        DataType targetType = this->dt;
        if (DT_F16_8Q == this->dt || DT_F32_8Q == this->dt) {
            targetType = DT_I8;
        }

        CHECK_STATUS(deconvolution_infer_output_size(
            inputTensor, filterTensor, p, outTensors[0], targetType, &this->archInfo));

        if (tensorIs3d(inTensors[0]->get_desc()) && tensorIs4d(outTensors[0]->get_desc())) {
            DataType odt;
            DataFormat odf;
            U32 on, oc, oh, ow;
            CHECK_STATUS(tensor4dGet(outTensors[0]->get_desc(), &odt, &odf, &on, &oc, &oh, &ow));
            outTensors[0]->resize(tensor3df(odt, odf, on, oc, oh));
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc oriInputDesc = inputTensor.get_desc();
        inputTensor.resize(transformDescTo4d(oriInputDesc));
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc oriOutputDesc = outputTensor.get_desc();
        outputTensor.resize(transformDescTo4d(oriOutputDesc));
        U32 bytes = 0;
        CHECK_STATUS(deconvolution_infer_forward_tmp_bytes(inputTensor, this->weightTensors[0],
            outputTensor, p, this->alg, &bytes, &this->archInfo));
        inputTensor.resize(oriInputDesc);
        outputTensor.resize(oriOutputDesc);
        return bytes;
    }

    U32 infer_wtm_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(deconvolution_transform_filter_bytes(
            this->weightTensors[0], this->p, this->alg, &bytes, &this->archInfo));
        return bytes;
    }

    EE transform_filter() override
    {
        this->wtm = std::shared_ptr<Tensor>(new Tensor());
        Tensor filterTensor = this->weightTensors[0];
        auto wtmBytes = this->infer_wtm_memory_size();
        Tensor wtm = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, wtmBytes));
        EE ret = deconvolution_transform_filter(
            filterTensor, this->p, this->alg, this->temp, &wtm, &this->archInfo);
        this->weightTensors[0] = wtm;
        return ret;
    }
};

#endif  // _DECONVOLUTION_CPU_H
