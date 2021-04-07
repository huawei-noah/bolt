// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _CONVELTWISEPOOLING_CPU_H
#define _CONVELTWISEPOOLING_CPU_H

#include "convolution.hpp"

class ConvolutionCPU : public Convolution {
public:
    ConvolutionCPU(DataType dt,
        ConvolutionParamSpec p,
        ActivationParamSpec dwActivationParamSpec,
        ActivationParamSpec pwActivationParamSpec)
        : Convolution(dt, p, dwActivationParamSpec, pwActivationParamSpec)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ConvolutionCPU> mem = std::shared_ptr<ConvolutionCPU>(new ConvolutionCPU(
            this->dt, this->p, this->dwActivationParamSpec, this->pwActivationParamSpec));
        *mem = *this;
        return mem;
    }

    EE init_weight_bias_from_model(std::shared_ptr<U8> *modelPtrShared) override
    {
        U8 *modelPtr = nullptr;
        if (modelPtrShared != nullptr) {
            modelPtr = (*modelPtrShared).get();
        }
        auto curOpWs = this->get_weightspec();
        DataType filterDt = curOpWs.mdt;  // weight data type may not be the same as input and output
        if (modelPtr != nullptr) {
            filterDt = this->dt;
        }
        DataType dtNoQ = (this->dt == DT_F16_8Q) ? DT_F16 : this->dt;
        U32 isBNN = 0;
        if (filterDt == DT_BIN01 || filterDt == DT_BIN11) {
            isBNN = 1;
        }

        for (U32 i = 0; i < this->weightTensors.size(); i++) {
            TensorDesc desc = this->weightTensors[i].get_desc();
            desc.dt = filterDt;
            this->weightTensors[i].resize(desc);
        }
        for (U32 i = 0; i < this->biasTensors.size(); i++) {
            TensorDesc desc = this->biasTensors[i].get_desc();
            desc.dt = dtNoQ;
            if (this->p.convolution_type == Convolution_Pointwise) {
                U32 vectorLen = this->p.num_outputs;  // bias length
                if (isBNN == 1) {
                    this->dt = dtNoQ;  // BNN convolution should not be quantized further
                    vectorLen *= 2;  // Scale has the same vector length as bias, so double the length
                }
                desc = tensor1d(dtNoQ, vectorLen);
            }
            this->biasTensors[i].resize(desc);
        }
        std::shared_ptr<U8> weight_ptr = std::shared_ptr<U8>(curOpWs.weight, [](U8 *) {});
        U32 weight_offset = 0;
        U32 bias_offset = 0;
        for (U32 j = 0; j < this->weightTensors.size(); j++) {
            U32 weight_bytes = this->weightTensors[j].bytes();
            U32 bias_bytes = this->biasTensors[j].bytes();
            U32 offset_bytes = 0;
            if (modelPtr != nullptr) {
                this->weightTensors[j].alloc();
                memcpy(((CpuMemory *)(this->weightTensors[j].get_memory()))->get_ptr(), modelPtr,
                    weight_bytes);
                offset_bytes += weight_bytes;
                if (this->hasBias) {
                    this->biasTensors[j].alloc();
                    memcpy(((CpuMemory *)(this->biasTensors[j].get_memory()))->get_ptr(),
                        modelPtr + offset_bytes, bias_bytes);
                    offset_bytes += bias_bytes;
                }
                *modelPtrShared = std::shared_ptr<U8>(*modelPtrShared, modelPtr + offset_bytes);
            } else {
                ((CpuMemory *)(this->weightTensors[j].get_memory()))
                    ->set_shared_ptr(
                        std::shared_ptr<U8>(weight_ptr, weight_ptr.get() + weight_offset));

                weight_offset += weight_bytes;
                if (this->hasBias) {
                    this->biasTensors[j].alloc();
                    memcpy(((CpuMemory *)(this->biasTensors[j].get_memory()))->get_ptr(),
                        curOpWs.vec + bias_offset, bias_bytes);
                    bias_offset += bias_bytes;
                }
            }
            if (!this->hasBias) {
                this->biasTensors[j].alloc();
                if (isBNN == 1) {
#ifdef _USE_FP16
                    U8 *ptr = (U8 *)((CpuMemory *)(this->biasTensors[j].get_memory()))->get_ptr();
                    UNI_INIT(p.num_outputs, DT_F16, 1.0, ptr);
                    ptr += bias_bytes / 2;
                    memset(ptr, 0, bias_bytes / 2);  // second half is bias
#endif
                } else {
                    memset(((CpuMemory *)(this->biasTensors[j].get_memory()))->get_ptr(), 0,
                        bias_bytes);
                }
            }
        }
        return SUCCESS;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor filterTensor = this->weightTensors[0];
        U8 *scalePtr = nullptr;
        Tensor biasTensor = this->biasTensors[0];
        Tensor outputTensor = this->outputTensors[0];

        switch (this->p.convolution_type) {
            case Convolution_Pointwise: {
                if (DT_F16_8Q == this->dt) {
#ifdef _USE_INT8
                    F16 *ptr = this->scales.get();
                    scalePtr = (U8 *)ptr;
                    auto inputDesc = inputTensor.get_desc();

                    ptr[0] = inputTensor.get_scale();
                    if (featureScale.size() > 0 && featureScale[0][0] > 0) {
                        ptr[0] = featureScale[0][0];
                    } else if (DT_F16 == inputDesc.dt) {
                        ptr[0] = -1;
                    }

                    if (featureScale.size() > 0 && (featureScale.back())[0] != -2) {
                        ptr[1] = (featureScale.back())[0];
                    } else {
                        ptr[1] = -1;
                    }
#endif
                }
                CHECK_STATUS(convolution(this->inputTensors, filterTensor, p, this->pwAlg, scalePtr,
                    biasTensor, this->temp, outputTensor, this->pwActivationParamSpec,
                    &this->archInfo));
#ifdef _USE_INT8
                auto outputDesc = outputTensor.get_desc();
                if (DT_I8 == outputDesc.dt) {
                    F16 *ptr = (F16 *)scalePtr;
                    outputTensor.set_scale(ptr[1]);
                }
#endif
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(
                    depthwise_convolution(inputTensor, filterTensor, p, this->dwAlg, biasTensor,
                        this->temp, outputTensor, this->dwActivationParamSpec, &this->archInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(
                    depthwise_pointwise_convolution(inputTensor, filterTensor, weightTensors[1], p,
                        this->dwAlg, biasTensor, biasTensors[1], this->temp, outputTensor,
                        this->dwActivationParamSpec, this->pwActivationParamSpec, &this->archInfo));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(convolution(this->inputTensors, filterTensor, p, this->pwAlg, scalePtr,
                    biasTensor, this->temp, outputTensor, this->pwActivationParamSpec,
                    &this->archInfo));
                break;
            }
            default: {
                UNI_ERROR_LOG("unsupported convolution type %d\n", this->p.convolution_type);
            }
        }
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        auto inputTensor = this->inputTensors[0];
        auto filterTensor = this->weightTensors[0];
        auto outputTensor = this->outputTensors[0];
        TensorDesc inputDesc = this->desc_process(inputTensor.get_desc());
        inputTensor.resize(inputDesc);
        TensorDesc filterDesc = filterTensor.get_desc();

        ConvolutionPolicy policy = CONVOLUTION_FASTEST;
        DataType targetType = filterDesc.dt;
        I32 algo;
        switch (this->p.convolution_type) {
            case Convolution_Pointwise: {
                if (this->dt == DT_F16_8Q) {
                    targetType = DT_I8;
                }
                if (algorithmMap->getAlgorithmInfoFromMap(this->name, &algo, 1)) {
                    this->pwAlg = (ConvolutionForwardAlgorithm)algo;
                } else if (algorithmMap->getCommonAlgoInfoFromMap(OT_Conv, this->dt,
                               inputDesc.dims[2], inputDesc.dims[1], inputDesc.dims[0],
                               filterDesc.dims[3], filterDesc.dims[1], filterDesc.dims[0],
                               this->p.stride_h, this->p.stride_w, &algo, 1)) {
                    this->pwAlg = (ConvolutionForwardAlgorithm)algo;
                } else {
                    CHECK_STATUS(convolution_infer_forward_algorithm(inputTensor, filterTensor,
                        outputTensor, p, policy, &(this->pwAlg), targetType,
                        this->pwActivationParamSpec, &this->archInfo));
                    algo = this->pwAlg;
                    algorithmMap->setAlgorithmInfoToMap(this->name, &algo, 1);
                }
                break;
            }
            case Convolution_Depthwise: {
                if (algorithmMap->getAlgorithmInfoFromMap(this->name, &algo, 1)) {
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo;
                } else {
                    CHECK_STATUS(depthwise_convolution_infer_forward_algorithm(inputTensor,
                        filterTensor, outputTensor, p, policy, &(this->dwAlg), targetType,
                        this->dwActivationParamSpec, &this->archInfo));
                    algo = this->dwAlg;
                    algorithmMap->setAlgorithmInfoToMap(this->name, &algo, 1);
                }
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                if (algorithmMap->getAlgorithmInfoFromMap(this->name, &algo, 1)) {
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo;
                } else {
                    CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_algorithm(
                        inputTensor, filterTensor, this->weightTensors[1], outputTensor, p, policy,
                        &(this->dwAlg), targetType, this->dwActivationParamSpec,
                        this->pwActivationParamSpec, &this->archInfo));
                    algo = this->dwAlg;
                    algorithmMap->setAlgorithmInfoToMap(this->name, &algo, 1);
                }
                break;
            }
            case Convolution_Dilation: {
                if (algorithmMap->getAlgorithmInfoFromMap(this->name, &algo, 1)) {
                    this->pwAlg = (ConvolutionForwardAlgorithm)algo;
                } else {
                    CHECK_STATUS(convolution_infer_forward_algorithm(inputTensor, filterTensor,
                        outputTensor, p, policy, &(this->pwAlg), targetType,
                        this->pwActivationParamSpec, &this->archInfo));
                    algo = this->pwAlg;
                    algorithmMap->setAlgorithmInfoToMap(this->name, &algo, 1);
                }
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return SUCCESS;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        TensorDesc inDim = this->desc_process(inTensors[0]->get_desc());
        Tensor tmpTensor;
        tmpTensor.resize(inDim);
        auto inputTensor = &tmpTensor;
        auto outputTensor = outTensors[0];

        DataType idt;
        DataFormat idf;
        U32 in, ic, it, ih, iw;
        if (tensorIs5d(inDim)) {
            CHECK_STATUS(tensor5dGet(inDim, &idt, &idf, &in, &ic, &it, &ih, &iw));
        } else if (tensorIs4d(inDim)) {
            CHECK_STATUS(tensor4dGet(inDim, &idt, &idf, &in, &ic, &ih, &iw));
        } else {
            return NOT_SUPPORTED;
        }
        if (DF_NCHW == idf && DT_F16_8Q == this->dt && DT_F16 == idt) {
            this->dt = DT_F16;
        }
        DataType targetType = this->dt;
        if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->p.convolution_type) {
            targetType = DT_I8;
        }
        int numChannels = ic;
        if (this->p.convolution_type == Convolution_Dilation ||
            this->p.convolution_type == Convolution_Pointwise) {
            numChannels /= this->p.group;
        }

        std::vector<TensorDesc> filterDesc, biasDesc;
        int channelAxis = 0;
        if (tensorIs5d(inDim)) {
            channelAxis = 4;
            filterDesc.push_back(tensor5d(this->dt, this->p.num_outputs, numChannels,
                this->p.kernel_t, this->p.kernel_h, this->p.kernel_w));
            if (Convolution_Depthwise_Pointwise == this->p.convolution_type) {
                filterDesc.push_back(tensor5d(this->dt, this->p.num_outputs, numChannels, 1, 1, 1));
            }
        } else if (tensorIs4d(inDim)) {
            channelAxis = 3;
            filterDesc.push_back(tensor4d(
                this->dt, this->p.num_outputs, numChannels, this->p.kernel_h, this->p.kernel_w));
            if (Convolution_Depthwise_Pointwise == this->p.convolution_type) {
                filterDesc.push_back(tensor4d(this->dt, this->p.num_outputs, numChannels, 1, 1));
            }
        }
        std::vector<Tensor> filterTensor(filterDesc.size());
        for (U32 i = 0; i < filterDesc.size(); i++) {
            filterTensor[i].resize(filterDesc[i]);
        }
        switch (this->p.convolution_type) {
            case Convolution_Pointwise: {
                biasDesc.push_back(tensor1d(this->dt, this->p.num_outputs));
                CHECK_STATUS(convolution_infer_output_size(
                    inputTensor, filterTensor[0], p, outputTensor, targetType, &this->archInfo));
                break;
            }
            case Convolution_Depthwise: {
                filterDesc[0].dims[channelAxis] = 1;
                filterTensor[0].resize(filterDesc[0]);
                biasDesc.push_back(tensor1d(this->dt, this->p.num_outputs));
                CHECK_STATUS(depthwise_convolution_infer_output_size(
                    inputTensor, filterTensor[0], p, outputTensor, targetType, &this->archInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                filterDesc[0].dims[channelAxis] = 1;
                filterTensor[0].resize(filterDesc[0]);
                biasDesc.push_back(tensor1d(this->dt, numChannels));
                biasDesc.push_back(tensor1d(this->dt, this->p.num_outputs));
                CHECK_STATUS(depthwise_pointwise_convolution_infer_output_size(inputTensor,
                    filterTensor[0], filterTensor[1], p, outputTensor, targetType, &this->archInfo));
                break;
            }
            case Convolution_Dilation: {
                biasDesc.push_back(tensor1d(this->dt, this->p.num_outputs));
                CHECK_STATUS(convolution_infer_output_size(
                    inputTensor, filterTensor[0], p, outputTensor, targetType, &this->archInfo));
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        if (DT_F16_8Q == this->dt && featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
            TensorDesc outputDesc = outputTensor->get_desc();
            outputDesc.dt = DT_F16;
            outputTensor->resize(outputDesc);
        }
        if (this->weightTensors.size() == 0) {
            this->weightTensors = filterTensor;
        }
        if (this->biasTensors.size() == 0) {
            this->biasTensors = std::vector<Tensor>(biasDesc.size());
            for (U32 i = 0; i < biasDesc.size(); i++) {
                this->biasTensors[i].resize(biasDesc[i]);
            }
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        auto inputTensor = this->inputTensors[0];
        TensorDesc inDim = this->desc_process(inputTensor.get_desc());
        inputTensor.resize(inDim);
        auto filterTensor = this->weightTensors[0];
        TensorDesc filterDesc = filterTensor.get_desc();
        if (DT_F16_8Q == filterDesc.dt) {
            filterDesc.dt = DT_I8;
            filterTensor.resize(filterDesc);
        }
        auto outputTensor = this->outputTensors[0];

        U32 bytes = 0;
        switch (this->p.convolution_type) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputTensor, filterTensor,
                    outputTensor, p, this->pwAlg, &bytes, &this->archInfo));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputTensor,
                    filterTensor, outputTensor, p, this->dwAlg, &bytes, &this->archInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_tmp_bytes(inputTensor,
                    filterTensor, this->weightTensors[1], outputTensor, p, this->dwAlg, &bytes,
                    &this->archInfo));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputTensor, filterTensor,
                    outputTensor, p, this->pwAlg, &bytes, &this->archInfo));
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return bytes;
    }

    U32 infer_filter_transform_bytes(U32 *bytesExtra)
    {
        auto filterTensor = this->weightTensors[0];
        U32 bytes = 0;
        switch (this->p.convolution_type) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_transform_filter_bytes(
                    filterTensor, this->p, this->pwAlg, &bytes, &this->archInfo));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_transform_filter_bytes(
                    filterTensor, this->p, this->dwAlg, &bytes, &this->archInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_pointwise_convolution_transform_filter_bytes(filterTensor,
                    weightTensors[1], this->p, this->dwAlg, &bytes, bytesExtra, &this->archInfo));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(convolution_transform_filter_bytes(
                    filterTensor, this->p, this->pwAlg, &bytes, &this->archInfo));
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return bytes;
    }

    EE transform_filter() override
    {
        Tensor filterTensor = this->weightTensors[0];
        this->wtm = std::shared_ptr<Tensor>(new Tensor());

        TensorDesc wtmDesc;
        if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->p.convolution_type &&
            CONVOLUTION_ALGORITHM_WINOGRAD == this->pwAlg) {  // int8 winograd
#ifdef _USE_INT8
            U32 ftBytes;
            CHECK_STATUS(convolution_transform_filter_bytes(
                filterTensor, this->p, this->pwAlg, &ftBytes, &this->archInfo));

            Tensor tFilter;
            tFilter.resize(tensor1d(DT_U8, ftBytes));
            tFilter.alloc();

            // To label as int8
            TensorDesc filterDesc = filterTensor.get_desc();
            filterDesc.dt = DT_F16_8Q;

            filterTensor.resize(filterDesc);
            CHECK_STATUS(convolution_transform_filter(
                filterTensor, this->p, this->pwAlg, this->temp, &tFilter, &this->archInfo));

            U32 ftmBytes = ftBytes / bytesOf(DT_F16);
            wtm->resize(tensor1d(DT_U8, ftmBytes));
            wtm->alloc();

            std::shared_ptr<F16> fsp((F16 *)operator new(38 * bytesOf(DT_F16)));
            this->scales = fsp;
            TensorDesc wtmDesc;
            CHECK_STATUS(quantize_tensor(tFilter.get_desc(),
                ((CpuMemory *)(tFilter.get_memory()))->get_ptr(), &wtmDesc,
                ((CpuMemory *)(wtm->get_memory()))->get_ptr(), this->scales.get() + 2));
            wtm->resize(wtmDesc);
        } else if (DT_F16_8Q == this->dt &&
            Convolution_Pointwise == this->p.convolution_type) {  // int8 tilegemm
            Tensor qFilterTensor;
            TensorDesc qDesc = filterTensor.get_desc();
            qDesc.dt = DT_I8;
            qFilterTensor.resize(qDesc);
            qFilterTensor.alloc();
            std::shared_ptr<F16> fsp((F16 *)operator new(3 * bytesOf(DT_F16)));
            this->scales = fsp;
            this->scales.get()[2] = -1;
            CHECK_STATUS(quantize_tensor(filterTensor.get_desc(),
                ((CpuMemory *)(filterTensor.get_memory()))->get_ptr(), &qDesc,
                ((CpuMemory *)(qFilterTensor.get_memory()))->get_ptr(), this->scales.get() + 2));

            U32 ftmBytes;
            qFilterTensor.resize(qDesc);
            CHECK_STATUS(convolution_transform_filter_bytes(
                qFilterTensor, this->p, this->pwAlg, &ftmBytes, &this->archInfo));

            wtm->resize(tensor1d(DT_U8, ftmBytes));
            wtm->alloc();

            // trans filter
            CHECK_STATUS(convolution_transform_filter(
                qFilterTensor, this->p, this->pwAlg, this->temp, this->wtm.get(), &this->archInfo));
#endif
        } else {  // All other cases
            U32 bytesExtra;
            auto wtmBytes = this->infer_filter_transform_bytes(&bytesExtra);
            wtm->resize(tensor1d(DT_U8, wtmBytes));
            wtm->alloc();

            switch (this->p.convolution_type) {
                case Convolution_Pointwise: {
                    CHECK_STATUS(convolution_transform_filter(filterTensor, this->p, this->pwAlg,
                        this->temp, this->wtm.get(), &this->archInfo));
                    break;
                }
                case Convolution_Depthwise: {
                    CHECK_STATUS(depthwise_convolution_transform_filter(
                        filterTensor, this->p, this->dwAlg, this->wtm.get(), &this->archInfo));
                    break;
                }
                case Convolution_Depthwise_Pointwise: {
                    Tensor pwTensor;
                    pwTensor.resize(tensor1d(DT_U8, bytesExtra));
                    pwTensor.alloc();
                    CHECK_STATUS(depthwise_pointwise_convolution_transform_filter(filterTensor,
                        weightTensors[1], this->p, this->dwAlg, this->wtm.get(), &pwTensor,
                        &this->archInfo));
                    weightTensors[1] = pwTensor;
                    break;
                }
                case Convolution_Dilation: {
                    CHECK_STATUS(convolution_transform_filter(filterTensor, this->p, this->pwAlg,
                        this->temp, this->wtm.get(), &this->archInfo));
                    break;
                }
                default:
                    CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        this->weightTensors[0] = *this->get_wtm();
        return SUCCESS;
    }
};

#endif  // _CONVELTWISEPOOLING_H
