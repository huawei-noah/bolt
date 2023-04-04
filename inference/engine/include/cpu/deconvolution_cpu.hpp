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
        DataType fdt = this->ws.mdt;
        if (this->ws.weight == nullptr) {
            fdt = this->dt;
        }
        if (fdt == DT_BIN01 || fdt == DT_BIN11) {
            return NOT_MATCH;
        }
        TensorDesc filterTensorDesc = tensor4df(
            fdt, DF_NCHW, this->numInputs, this->p.num_outputs, this->p.kernel_h, this->p.kernel_w);
        // bias data type should be the same as input and output
        TensorDesc vectorTensorDesc =
            tensor1d(noQuantDataType(this->dt), this->p.num_outputs * this->p.group);
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

        F32 scales[3] = {-1};
#ifdef _USE_INT8
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        scales[0] = inputTensor.get_scale();
        if (featureScale.size() > 1 && featureScale[0][0] > 0 && DT_I8 != inputDesc.dt &&
            DT_U8_Q != inputDesc.dt) {
            // inputTensor.set_scale(featureScale[0][0]);
            scales[0] = featureScale[0][0];
        }
        if (DT_I8 == outputDesc.dt || DT_U8_Q == outputDesc.dt) {
            if (featureScale.size() > 0) {
                // outputTensor.set_scale((featureScale.back())[0]);
                scales[1] = (featureScale.back())[0];
            } else {
                scales[1] = -1;
            }
        }
        scales[2] = filterTensor.get_scale();
#endif

        CHECK_STATUS(deconvolution(inputTensor, filterTensor, p, this->alg, scales, biasTensor,
            this->temp, outputTensor, this->activationDesc, &this->archInfo));
        inputTensor.resize(oriInputDesc);
        outputTensor.resize(oriOutputDesc);
#if defined(_USE_INT8)
        if (DT_I8 == outputDesc.dt || DT_U8_Q == outputDesc.dt) {
            outputTensor.set_scale(scales[1]);
        }
#endif
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

        DataType targetType = isQuantMixDataType(this->dt) ? get_activation_quant_data_type()
                                                           : this->dt;

        CHECK_STATUS(deconvolution_infer_output_size(
            inputTensor, filterTensor, p, outTensors[0], targetType, &this->archInfo));

        if (tensorIs3d(inTensors[0]->get_desc()) && tensorIs4d(outTensors[0]->get_desc())) {
            DataType odt;
            DataFormat odf;
            U32 on, oc, oh, ow;
            CHECK_STATUS(tensor4dGet(outTensors[0]->get_desc(), &odt, &odf, &on, &oc, &oh, &ow));
            outTensors[0]->resize(tensor3df(odt, odf, on, oc, oh));
        }
        TensorDesc outputDesc = outTensors[0]->get_desc();
        if (featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
            if (isQuantMixDataType(this->dt)) {
                outputDesc.dt = noQuantDataType(this->dt);
                outTensors[0]->resize(outputDesc);
            }
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

    EE transform_filter() override
    {
        Tensor filterTensor = this->weightTensors[0];
        U32 wtmBytes = 0;
        CHECK_STATUS(deconvolution_transform_filter_bytes(
            filterTensor, this->p, this->alg, &wtmBytes, &this->archInfo));
        Tensor wtm = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, wtmBytes));
#ifdef _USE_INT8
        TensorDesc desc = filterTensor.get_desc();
        bool thisIsNoQuant = (featureScale.size() > 1 && featureScale[0].back() == 0);
        if (isQuantMixDataType(this->dt) && !thisIsNoQuant && (desc.dt != DT_I8)) {
            desc.dt = DT_I8;
            Tensor qTensor = Tensor::alloc_sized<CPUMem>(desc);
            F32 scale = -1;
            CHECK_STATUS(quantize(filterTensor, &qTensor, &scale, &(this->archInfo)));
            qTensor.set_scale(scale);
            filterTensor = qTensor;
        }
        wtm.set_scale(filterTensor.get_scale());
#endif
        EE ret = deconvolution_transform_filter(
            filterTensor, this->p, this->alg, this->temp, &wtm, &this->archInfo);
        this->weightTensors[0] = wtm;
        return ret;
    }
};

#endif  // _DECONVOLUTION_CPU_H
