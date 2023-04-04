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
        DataType filterDt = this->ws.mdt;  // weight data type may not be the same as input and output
        if (modelPtr != nullptr) {
            filterDt = this->dt;
        }
        DataType dtNoQ = noQuantDataType(this->dt);
        U32 isBNN = 0;
        if (filterDt == DT_BIN01 || filterDt == DT_BIN11) {
            isBNN = 1;
        }

        if (this->ws.num_quant_scale == this->weightTensors.size()) {
            for (U32 i = 0; i < this->weightTensors.size(); ++i) {
                if (this->ws.weight_scale[i].num_scale > 0) {
                    this->weightTensors[i].set_scale_ptr(
                        std::shared_ptr<F32>(this->ws.weight_scale[i].scale, [](F32 *) {}));
                }
            }
        }

        for (U32 i = 0; i < this->weightTensors.size(); i++) {
            TensorDesc desc = this->weightTensors[i].get_desc();
            desc.dt = filterDt;
            this->weightTensors[i].resize(desc);
        }
        for (U32 i = 0; i < this->biasTensors.size(); i++) {
            TensorDesc desc = this->biasTensors[i].get_desc();
            desc.dt = dtNoQ;
            if (this->p.convolution_type == CONVOLUTION_POINTWISE) {
                U32 vectorLen = this->p.num_outputs;  // bias length
                if (isBNN == 1) {
                    this->dt = dtNoQ;  // BNN convolution should not be quantized further
                    vectorLen *= 2;  // Scale has the same vector length as bias, so double the length
                }
                desc = tensor1d(dtNoQ, vectorLen);
            }
            this->biasTensors[i].resize(desc);
        }
        std::shared_ptr<U8> weight_ptr = std::shared_ptr<U8>(this->ws.weight, [](U8 *) {});
        U32 weight_offset = 0;
        U32 bias_offset = 0;
        for (U32 j = 0; j < this->weightTensors.size(); j++) {
            U32 weight_bytes = this->weightTensors[j].bytes();
            U32 bias_bytes = this->biasTensors[j].bytes();
            U32 offset_bytes = 0;
            if (modelPtr != nullptr) {
                this->weightTensors[j].alloc();
                UNI_MEMCPY(((CpuMemory *)(this->weightTensors[j].get_memory()))->get_ptr(),
                    modelPtr, weight_bytes);
                offset_bytes += weight_bytes;
                if (this->ws.bytes_of_vec != 0) {
                    this->biasTensors[j].alloc();
                    UNI_MEMCPY(((CpuMemory *)(this->biasTensors[j].get_memory()))->get_ptr(),
                        modelPtr + offset_bytes, bias_bytes);
                    offset_bytes += bias_bytes;
                }
                *modelPtrShared = std::shared_ptr<U8>(*modelPtrShared, modelPtr + offset_bytes);
            } else {
                ((CpuMemory *)(this->weightTensors[j].get_memory()))
                    ->set_shared_ptr(
                        std::shared_ptr<U8>(weight_ptr, weight_ptr.get() + weight_offset));

                weight_offset += weight_bytes;
                if (this->ws.bytes_of_vec != 0) {
                    this->biasTensors[j].alloc();
                    UNI_MEMCPY(((CpuMemory *)(this->biasTensors[j].get_memory()))->get_ptr(),
                        this->ws.vec + bias_offset, bias_bytes);
                    bias_offset += bias_bytes;
                }
            }
            if (this->ws.bytes_of_vec == 0) {
                this->biasTensors[j].alloc();
                if (isBNN == 1) {
#ifdef _USE_FP16
                    U8 *ptr = (U8 *)((CpuMemory *)(this->biasTensors[j].get_memory()))->get_ptr();
                    UNI_INIT(p.num_outputs, DT_F16, 1.0, ptr);
                    ptr += bias_bytes / 2;
                    UNI_MEMSET(ptr, 0, bias_bytes / 2);  // second half is bias
#endif
                } else {
                    UNI_MEMSET(((CpuMemory *)(this->biasTensors[j].get_memory()))->get_ptr(), 0,
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
        Tensor biasTensor = this->biasTensors[0];
        Tensor outputTensor = this->outputTensors[0];

        TensorDesc oriInputDesc = inputTensor.get_desc();
        inputTensor.resize(transformDescTo4d(oriInputDesc));
        TensorDesc oriOutputDesc = outputTensor.get_desc();
        TensorDesc outputDesc = transformDescTo4d(oriOutputDesc);
        outputTensor.resize(outputDesc);

        F32 *scalePtr = nullptr;
#if defined(_USE_INT8)
        if (isQuantMixDataType(this->dt) && this->scales.get() != nullptr) {
            TensorDesc inputDesc = inputTensor.get_desc();
            scalePtr = this->scales.get();
            scalePtr[0] = inputTensor.get_scale();
            if (DT_I8 != inputDesc.dt && DT_U8_Q != inputDesc.dt && featureScale.size() > 0 &&
                featureScale[0][0] > 0) {
                scalePtr[0] = featureScale[0][0];
            }
            if (featureScale.size() > 0 && (featureScale.back())[0] != -2) {
                scalePtr[1] = (featureScale.back())[0];
            } else {
                scalePtr[1] = -1;
            }
        }
#endif
        switch (this->p.convolution_type) {
            case CONVOLUTION_DILATION:
            case CONVOLUTION_POINTWISE: {
                std::vector<Tensor> tmpTensors(1, this->temp);
                CHECK_STATUS(convolution(this->inputTensors, filterTensor, p, this->pwAlg, scalePtr,
                    biasTensor, tmpTensors, outputTensor, this->pwActivationParamSpec,
                    &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE: {
                CHECK_STATUS(depthwise_convolution(this->inputTensors[0], filterTensor, p,
                    this->dwAlg, scalePtr, biasTensor, this->temp, outputTensor,
                    this->dwActivationParamSpec, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE_POINTWISE: {
                std::vector<Tensor> tmpTensors(1, this->temp);
                CHECK_STATUS(depthwise_pointwise_convolution(this->inputTensors, filterTensor,
                    weightTensors[1], p, this->dwAlg, scalePtr, biasTensor, biasTensors[1],
                    tmpTensors, outputTensor, this->dwActivationParamSpec,
                    this->pwActivationParamSpec, &this->archInfo));
                break;
            }
            default: {
                UNI_ERROR_LOG("unsupported convolution type %d\n", this->p.convolution_type);
            }
        }
#if defined(_USE_INT8)
        if (DT_I8 == outputDesc.dt || DT_U8_Q == outputDesc.dt) {
            outputTensor.set_scale(scalePtr[1]);
        }
#endif
        inputTensor.resize(oriInputDesc);
        outputTensor.resize(oriOutputDesc);
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        auto inputTensor = this->inputTensors[0];
        auto filterTensor = this->weightTensors[0];
        auto outputTensor = this->outputTensors[0];
        TensorDesc oriInputDesc = inputTensor.get_desc();
        TensorDesc oriOutputDesc = outputTensor.get_desc();
        TensorDesc inputDesc = transformDescTo4d(oriInputDesc);
        inputTensor.resize(inputDesc);
        TensorDesc outputDesc = transformDescTo4d(oriOutputDesc);
        outputTensor.resize(outputDesc);
        TensorDesc filterDesc = filterTensor.get_desc();

        ConvolutionPolicy policy = CONVOLUTION_FASTEST;
        DataType targetType = filterDesc.dt;
        I32 algo;
        switch (this->p.convolution_type) {
            case CONVOLUTION_DILATION:
            case CONVOLUTION_POINTWISE: {
                if (isQuantMixDataType(this->dt)) {
                    targetType = get_activation_quant_data_type();
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
            case CONVOLUTION_DEPTHWISE: {
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
            case CONVOLUTION_DEPTHWISE_POINTWISE: {
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
            default:
                UNI_ERROR_LOG("not support to infer new type convolution's algorithm.\n");
                return NOT_SUPPORTED;
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
        if (DF_NCHW == idf) {
            if (DT_F16_8Q == this->dt && DT_F16 == idt) {
                this->dt = DT_F16;
            }
            if (DT_F32_8Q == this->dt && DT_F32 == idt) {
                this->dt = DT_F32;
            }
        }
        DataType targetType = this->dt;
        int numChannels = ic;
        if (this->p.convolution_type == CONVOLUTION_DILATION ||
            this->p.convolution_type == CONVOLUTION_POINTWISE) {
            if (isQuantMixDataType(this->dt)) {
                targetType = get_activation_quant_data_type();
            }
            numChannels /= this->p.group;
        }

        std::vector<TensorDesc> filterDesc, biasDesc;
        int channelAxis = 0;
        if (tensorIs5d(inDim)) {
            channelAxis = 4;
            filterDesc.push_back(tensor5d(this->dt, this->p.num_outputs, numChannels,
                this->p.kernel_t, this->p.kernel_h, this->p.kernel_w));
            if (CONVOLUTION_DEPTHWISE_POINTWISE == this->p.convolution_type) {
                filterDesc.push_back(tensor5d(this->dt, this->p.num_outputs, numChannels, 1, 1, 1));
            }
        } else if (tensorIs4d(inDim)) {
            channelAxis = 3;
            filterDesc.push_back(tensor4d(
                this->dt, this->p.num_outputs, numChannels, this->p.kernel_h, this->p.kernel_w));
            if (CONVOLUTION_DEPTHWISE_POINTWISE == this->p.convolution_type) {
                filterDesc.push_back(tensor4d(this->dt, this->p.num_outputs, numChannels, 1, 1));
            }
        }
        std::vector<Tensor> filterTensor(filterDesc.size());
        for (U32 i = 0; i < filterDesc.size(); i++) {
            filterTensor[i].resize(filterDesc[i]);
        }
        switch (this->p.convolution_type) {
            case CONVOLUTION_DILATION:
            case CONVOLUTION_POINTWISE: {
                biasDesc.push_back(tensor1d(this->dt, this->p.num_outputs));
                CHECK_STATUS(convolution_infer_output_size(
                    inputTensor, filterTensor[0], p, outputTensor, targetType, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE: {
                filterDesc[0].dims[channelAxis] = 1;
                filterTensor[0].resize(filterDesc[0]);
                biasDesc.push_back(tensor1d(this->dt, this->p.num_outputs));
                CHECK_STATUS(depthwise_convolution_infer_output_size(
                    inputTensor, filterTensor[0], p, outputTensor, targetType, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE_POINTWISE: {
                filterDesc[0].dims[channelAxis] = 1;
                filterTensor[0].resize(filterDesc[0]);
                biasDesc.push_back(tensor1d(this->dt, numChannels));
                biasDesc.push_back(tensor1d(this->dt, this->p.num_outputs));
                CHECK_STATUS(depthwise_pointwise_convolution_infer_output_size(inputTensor,
                    filterTensor[0], filterTensor[1], p, outputTensor, targetType, &this->archInfo));
                break;
            }
            default:
                UNI_ERROR_LOG("not support to infer new type convolution's output.\n");
                return NOT_SUPPORTED;
        }
        TensorDesc outputDesc = outputTensor->get_desc();
        if (featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
            if (isQuantMixDataType(this->dt)) {
                outputDesc.dt = noQuantDataType(this->dt);
                outputTensor->resize(outputDesc);
            }
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
        if (tensorIs3d(inTensors[0]->get_desc()) && tensorIs4d(outputDesc)) {
            DataType odt;
            DataFormat odf;
            U32 on, oc, oh, ow;
            CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
            outputTensor->resize(tensor3df(odt, odf, on, oc, oh));
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc oriInputDesc = inputTensor.get_desc();
        TensorDesc inputDesc = transformDescTo4d(oriInputDesc);
        inputTensor.resize(inputDesc);
        Tensor filterTensor = this->weightTensors[0];
        TensorDesc filterDesc = filterTensor.get_desc();
        if (isQuantMixDataType(filterDesc.dt)) {
            filterDesc.dt = DT_I8;
            filterTensor.resize(filterDesc);
        }
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc oriOutputDesc = outputTensor.get_desc();
        TensorDesc outputDesc = transformDescTo4d(oriOutputDesc);
        outputTensor.resize(outputDesc);

        U32 bytes = 0;
        switch (this->p.convolution_type) {
            case CONVOLUTION_DILATION:
            case CONVOLUTION_POINTWISE: {
                CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputTensor, filterTensor,
                    outputTensor, p, this->pwAlg, &bytes, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE: {
                CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputTensor,
                    filterTensor, outputTensor, p, this->dwAlg, &bytes, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE_POINTWISE: {
                CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_tmp_bytes(inputTensor,
                    filterTensor, this->weightTensors[1], outputTensor, p, this->dwAlg, &bytes,
                    &this->archInfo));
                break;
            }
            default:
                UNI_ERROR_LOG("not support to infer new type convolution's tmp memory.\n");
                break;
        }
        inputTensor.resize(oriInputDesc);
        outputTensor.resize(oriOutputDesc);
        return bytes;
    }

    U32 infer_filter_transform_bytes(U32 *bytesExtra)
    {
        auto filterTensor = this->weightTensors[0];
        U32 bytes = 0;
        switch (this->p.convolution_type) {
            case CONVOLUTION_DILATION:
            case CONVOLUTION_POINTWISE: {
                CHECK_STATUS(convolution_transform_filter_bytes(
                    filterTensor, this->p, this->pwAlg, &bytes, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE: {
                CHECK_STATUS(depthwise_convolution_transform_filter_bytes(
                    filterTensor, this->p, this->dwAlg, &bytes, &this->archInfo));
                break;
            }
            case CONVOLUTION_DEPTHWISE_POINTWISE: {
                CHECK_STATUS(depthwise_pointwise_convolution_transform_filter_bytes(filterTensor,
                    weightTensors[1], this->p, this->dwAlg, &bytes, bytesExtra, &this->archInfo));
                break;
            }
            default:
                UNI_ERROR_LOG("not support to infer new type convolution's tramsform filter tmp "
                              "memory.\n");
                break;
        }
        return bytes;
    }

    EE transform_filter() override
    {
#if 0  //defined(_USE_LITE) && !defined(_USE_NEON)
        return SUCCESS;
#endif
        Tensor filterTensor = this->weightTensors[0];
        TensorDesc wtmDesc;
        Tensor wtm;
        // int8 winograd
        if (isQuantMixDataType(this->dt) && CONVOLUTION_POINTWISE == this->p.convolution_type &&
            CONVOLUTION_ALGORITHM_WINOGRAD == this->pwAlg) {
#if defined(_USE_INT8)
            TensorDesc filterDesc = filterTensor.get_desc();
            if ((filterDesc.dt != DT_F16_8Q) && (filterDesc.dt != DT_F16)) {
                if (filterDesc.dt == DT_I8) {
                    filterDesc.dt = DT_F16;
                    Tensor f16Filter = Tensor::alloc_sized<CPUMem>(filterDesc);
                    Tensor bias;
                    F32 scale = filterTensor.get_scale();
                    dequantize(filterTensor, &scale, bias, f16Filter, &(this->archInfo));
                    filterTensor = f16Filter;
                } else {
                    return NOT_SUPPORTED;
                }
            }
            U32 ftBytes;
            CHECK_STATUS(convolution_transform_filter_bytes(
                filterTensor, this->p, this->pwAlg, &ftBytes, &this->archInfo));

            Tensor tFilter = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, ftBytes));

            // To label as int8
            filterDesc.dt = DT_F16_8Q;

            filterTensor.resize(filterDesc);
            CHECK_STATUS(convolution_transform_filter(
                filterTensor, this->p, this->pwAlg, this->temp, &tFilter, &this->archInfo));

            U32 ftmBytes = ftBytes / bytesOf(DT_F16);
            wtm = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, ftmBytes));

            this->scales = std::shared_ptr<F32>((F32 *)operator new(38 * bytesOf(DT_F32)));
            CHECK_STATUS(
                quantize(tFilter, &wtm, this->scales.get() + 2, &(this->archInfo)));
            // int8 tilegemm
        } else if (isQuantMixDataType(this->dt) &&
            (CONVOLUTION_POINTWISE == this->p.convolution_type ||
                CONVOLUTION_DILATION == this->p.convolution_type)) {
            TensorDesc qDesc = filterTensor.get_desc();
            this->scales = std::shared_ptr<F32>((F32 *)operator new(3 * bytesOf(DT_F32)));
            if (qDesc.dt != DT_I8) {
                qDesc.dt = DT_I8;
                Tensor qFilterTensor = Tensor::alloc_sized<CPUMem>(qDesc);
                this->scales.get()[2] = -1;
                CHECK_STATUS(quantize(
                    filterTensor, &qFilterTensor, this->scales.get() + 2, &(this->archInfo)));
                filterTensor = qFilterTensor;
                filterTensor.set_scale(this->scales.get()[2]);
            } else {
                this->scales.get()[2] = filterTensor.get_scale();
            }

            U32 ftmBytes;
            CHECK_STATUS(convolution_transform_filter_bytes(
                filterTensor, this->p, this->pwAlg, &ftmBytes, &this->archInfo));
            wtm = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, ftmBytes));

            // trans filter
            CHECK_STATUS(convolution_transform_filter(
                filterTensor, this->p, this->pwAlg, this->temp, &wtm, &this->archInfo));
#endif
        } else {  // All other cases
            U32 bytesExtra;
            auto ftmBytes = this->infer_filter_transform_bytes(&bytesExtra);
            wtm = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, ftmBytes));
            switch (this->p.convolution_type) {
                case CONVOLUTION_DILATION:
                case CONVOLUTION_POINTWISE: {
                    CHECK_STATUS(convolution_transform_filter(filterTensor, this->p, this->pwAlg,
                        this->temp, &wtm, &this->archInfo));
                    break;
                }
                case CONVOLUTION_DEPTHWISE: {
                    CHECK_STATUS(depthwise_convolution_transform_filter(
                        filterTensor, this->p, this->dwAlg, &wtm, &this->archInfo));
                    break;
                }
                case CONVOLUTION_DEPTHWISE_POINTWISE: {
                    Tensor pwTensor;
                    pwTensor.resize(tensor1d(DT_U8, bytesExtra));
                    pwTensor.alloc();
                    CHECK_STATUS(depthwise_pointwise_convolution_transform_filter(filterTensor,
                        weightTensors[1], this->p, this->dwAlg, &wtm, &pwTensor,
                        &this->archInfo));
                    weightTensors[1] = pwTensor;
                    break;
                }
                default:
                    UNI_ERROR_LOG("not support to transform new type convolution's filter.\n");
                    return NOT_SUPPORTED;
            }
        }
        this->weightTensors[0] = wtm;
        return SUCCESS;
    }
};

#endif  // _CONVELTWISEPOOLING_H
