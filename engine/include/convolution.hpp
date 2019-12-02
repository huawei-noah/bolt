// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _CONVELTWISEPOOLING_H
#define _CONVELTWISEPOOLING_H
#include <optional>
#include "weight_operator.hpp"
#include "pooling.hpp"
#include "eltwise.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "model_tools.h"
#include <stdio.h>

template<Arch A>
class Convolution: public WeightOperator<A> {
public:
    Convolution(DataType dt, U32 nf, U32 ksize, U32 kstride, U32 kpadding, 
        std::optional<EltwiseType> et, std::optional<PoolingMode> pm, U32 pks, U32 pkstride, U32 pkp,
        ActivationMode dwActiveMode, ActivationMode pwActiveMode,
        ConvolutionMode convolutionType, U32 group, U32 dilation)
    {
        this->dt = dt;
        this->numFilters = nf;
        this->kernelSize = ksize;
        this->kernelStride = kstride;
        this->kernelPadding = kpadding;
        this->eltwiseType = et;
        this->poolingMode = pm;
        this->poolingSize = pks;
        this->poolingStride = pkstride;
        this->poolingPadding = pkp;
        this->set_op_type(OT_Conv);
        this->hasBias = false;
        this->dwActiveMode = dwActiveMode;
        this->pwActiveMode = pwActiveMode;
        this->convolutionType = convolutionType;
        this->group = group;
        this->dilation = dilation;
    }

    Convolution(DataType dt, U32 nf, U32 ksize, U32 kstride, U32 kpadding, std::optional<EltwiseType> et,
                              ActivationMode dwActiveMode, ActivationMode pwActiveMode,
                              ConvolutionMode convolutionType, U32 group, U32 dilation)
    {
        std::optional<PoolingMode> pm_null;
        Convolution(dt, nf, ksize, kstride, kpadding, et, pm_null, 0, 0, 0, dwActiveMode, pwActiveMode, convolutionType, group, dilation);
    }

    Convolution(DataType dt, U32 nf, U32 ksize, U32 kstride, U32 kpadding,
                              ActivationMode dwActiveMode, ActivationMode pwActiveMode,
                              ConvolutionMode convolutionType, U32 group, U32 dilation)
    {
        std::optional<EltwiseType> et_null;
        Convolution(dt, nf, ksize, kstride, kpadding, et_null, dwActiveMode, pwActiveMode, convolutionType, group, dilation);
    }

    Convolution(DataType dt, U32 nf, U32 ksize, U32 kpadding,
                              ActivationMode dwActiveMode, ActivationMode pwActiveMode,
                              ConvolutionMode convolutionType, U32 group, U32 dilation)
    {
        Convolution(dt, nf, ksize, 1, kpadding, dwActiveMode, pwActiveMode, convolutionType, group, dilation);
    }

    Convolution(DataType dt, U32 nf, U32 ksize)
    {
        Convolution(dt, nf, ksize, ksize/2);
    }

    ConvolutionDesc create_convDesc(U32 kernelStride, U32 kernelPadding, U32 kernelDilation)
    {
        ConvolutionDesc convDesc;
        convDesc.stride = kernelStride;
        convDesc.padding = kernelPadding;
        convDesc.dilatedRate = kernelDilation;
        return convDesc;
    }
    
    EE init_weight_bias_from_model(U8** modelPtr)
    {
        auto curOpWs = this->get_weightspec_ptr();
        DataType filterDt = curOpWs.mdt; // weight data type may not be the same as input and output
        if (modelPtr != nullptr) {
            filterDt = DT_F16;
        }
        DataType dtNoQ = (this->dt == DT_F16_8Q) ? DT_F16 : this->dt;
        U32 isBNN = 0;
        if (filterDt == DT_DOREFA || filterDt == DT_XNOR) {
            isBNN = 1;
        }
        DataFormat filterDf;
        U32 vectorLen = 0; // Vector must contain bias. BNN has one more scale vector.
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                filterDf = DF_NCHW;
                vectorLen = this->numFilters; // bias length
                if (isBNN == 1) {
                    this->dt = dtNoQ; // BNN convolution should not be quantized further
                    vectorLen *= 2; // Scale has the same vector length as bias, so double the length
                }
                break;
            }
            case Convolution_Depthwise: {
                filterDf = DF_NCHW;
                vectorLen = this->numFilters;
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                filterDf = DF_CHW_NC;
                vectorLen = this->numFilters + this->numChannels;
                break;
            }
            case Convolution_Dilation: {
                filterDf = DF_NCHW;
                vectorLen = this->numFilters;
                break;
            }
            default:
                CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
        }
        TensorDesc filterTensorDesc = tensor4df(filterDt, filterDf,
                                                 this->numFilters, this->numChannels,
                                                 this->kernelSize, this->kernelSize);
        TensorDesc vectorTensorDesc = tensor1d(dtNoQ, vectorLen); // bias data type should be the same as input and output

        U8* modelWeightPtr = nullptr;
        U8* modelVectorPtr = nullptr;
        if (modelPtr != nullptr) {
            modelWeightPtr = (U8 *)operator new(tensorNumBytes(filterTensorDesc));
            memcpy(modelWeightPtr, *modelPtr, tensorNumBytes(filterTensorDesc));
            *modelPtr += tensorNumBytes(filterTensorDesc);
            if (this->hasBias) {
                modelVectorPtr = (*modelPtr);
                *modelPtr += tensorNumBytes(vectorTensorDesc);
            }
        }
        else {
            modelWeightPtr = curOpWs.weight;
            modelVectorPtr = curOpWs.vec;
        }

        std::shared_ptr<U8> weightVal(modelWeightPtr);
        Tensor weightTensor = Tensor(filterTensorDesc, weightVal);
        this->weightTensors.push_back(weightTensor);

        // vector
        std::shared_ptr<U8> vectorVal;
        Tensor vectorTensor = Tensor(vectorTensorDesc, vectorVal);
        vectorTensor.alloc();
        U8* vectorPtr = vectorTensor.get_val().get();
        if (this->hasBias) {
            memcpy(vectorPtr, modelVectorPtr, tensorNumBytes(vectorTensorDesc));
        } else {
            if (isBNN == 1) {
                F16 *vec = (F16*)vectorPtr;
                for (U32 i = 0; i < this->numFilters; i++) { // first half is scale
                    *vec = 1.0;
                    vec++;
                }
                memset(vec, 0, tensorNumBytes(vectorTensorDesc) / 2); // second half is bias 
            } else {
                memset(vectorPtr, 0, tensorNumBytes(vectorTensorDesc));
            }
        }
        this->biasTensors.push_back(vectorTensor);
        return SUCCESS;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor filterTensor = this->weightTensors[0];
        TensorDesc filterDesc = filterTensor.get_desc();

        ConvolutionDesc convDesc = create_convDesc(this->kernelStride, this->kernelPadding, this->dilation);

        TensorDesc scaleDesc = filterDesc; // Dummy initialization
        F16 *scalePtr = nullptr;

        Tensor biasTensor = this->biasTensors[0];
        TensorDesc biasDesc = biasTensor.get_desc();
        F16 *biasPtr = (F16*)biasTensor.get_val().get();

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                if (filterDesc.dt == DT_DOREFA || filterDesc.dt == DT_XNOR) {
                    U32 vecLen = tensorNumElements(biasDesc) / 2;

                    scaleDesc = tensor1d(biasDesc.dt, vecLen);
                    biasDesc = tensor1d(biasDesc.dt, vecLen);
                    scalePtr = (F16*)biasTensor.get_val().get();
                    biasPtr = scalePtr + vecLen;
                } else if (DT_F16_8Q == this->dt) {
                    scalePtr = this->scales.get();
                    scalePtr[0] = inputTensor.get_scale();
                }
                CHECK_STATUS(convolution(inputDesc, inputTensor.get_val().get(),
                                         filterDesc, filterTensor.get_val().get(),
                                         convDesc, this->pwAlg,
                                         scaleDesc, scalePtr,
                                         biasDesc, biasPtr,
                                         this->lenOfTemp, this->temp.get(),
                                         outputDesc, (void*)outputTensor.get_val().get(),
                                         this->pwActiveMode, A));
                if (DT_I8 == outputDesc.dt) {
                    outputTensor.set_scale(scalePtr[1]);
                }
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution(inputDesc, inputTensor.get_val().get(),
                                                   filterDesc, filterTensor.get_val().get(),
                                                   convDesc, this->dwAlg,
                                                   biasDesc, biasPtr,
                                                   this->lenOfTemp, this->temp.get(),
                                                   outputDesc, outputTensor.get_val().get(),
                                                   this->dwActiveMode, ACTIVATION_NULL,
                                                   A));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_convolution(inputDesc, inputTensor.get_val().get(),
                                                   filterDesc, filterTensor.get_val().get(),
                                                   convDesc, this->dwAlg,
                                                   biasDesc, biasPtr,
                                                   this->lenOfTemp, this->temp.get(),
                                                   outputDesc, outputTensor.get_val().get(),
                                                   this->dwActiveMode, this->pwActiveMode,
                                                   A));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(convolution(inputDesc, inputTensor.get_val().get(),
                                         filterDesc, filterTensor.get_val().get(),
                                         convDesc, this->pwAlg,
                                         scaleDesc, scalePtr,
                                         biasDesc, biasPtr,
                                         this->lenOfTemp, this->temp.get(),
                                         outputDesc, (void*)outputTensor.get_val().get(),
                                         this->pwActiveMode, A));
                break;
            }
            default:
                std::cerr << "[ERROR] unsupported convolution type " << convolutionType << std::endl;
                exit(0);
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_forward_algorithm()
    {
        TensorDesc inputDesc = (this->inputTensors[0]).get_desc();
        TensorDesc filterDesc = (this->weightTensors[0]).get_desc();

        ConvolutionPolicy policy = CONVOLUTION_FASTEST;
        ConvolutionDesc convDesc = create_convDesc(this->kernelStride, this->kernelPadding, this->dilation);

        DataType targetType = filterDesc.dt;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                if (this->dt == DT_F16_8Q) {
                    targetType = DT_I8;
                }
                CHECK_STATUS_WITH_RETURN(convolution_infer_forward_algorithm(inputDesc, filterDesc,
                                             this->outputTensors[0].get_desc(),
                                             convDesc, policy, &(this->pwAlg), targetType, A));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS_WITH_RETURN(depthwise_convolution_infer_forward_algorithm(inputDesc, filterDesc,
                                             this->outputTensors[0].get_desc(),
                                             convDesc, policy, &(this->dwAlg),
                                             targetType, A));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS_WITH_RETURN(depthwise_convolution_infer_forward_algorithm(inputDesc, filterDesc,
                                             this->outputTensors[0].get_desc(),
                                             convDesc, policy, &(this->dwAlg), targetType, A));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS_WITH_RETURN(convolution_infer_forward_algorithm(inputDesc, filterDesc,
                                             this->outputTensors[0].get_desc(),
                                             convDesc, policy, &(this->pwAlg), targetType, A));
                break;
            }
            default:
                CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
        }
        return SUCCESS;
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inDim = inDims[0];
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        CHECK_STATUS_WITH_RETURN(tensor4dGet(inDim, &idt, &idf, &in, &ic, &ih, &iw));
        this->numChannels = ic;

        TensorDesc filterDim = tensor4df(this->dt, DF_NCHW, this->numFilters, this->numChannels, this->kernelSize,
                                    this->kernelSize);

        ConvolutionDesc convDesc = create_convDesc(this->kernelStride, this->kernelPadding, this->dilation);

        DataType targetType = DT_F16; // Default DT_F16
        if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->convolutionType) {
            targetType = DT_I8;
        }

        U32 outBytes = 0;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                CHECK_STATUS_WITH_RETURN(convolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS_WITH_RETURN(depthwise_convolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS_WITH_RETURN(depthwise_convolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS_WITH_RETURN(convolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes));
                break;
            }
            default:
                CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size()
    {
        TensorDesc inputDesc = (this->inputTensors[0]).desc;
        TensorDesc filterDesc = (this->weightTensors[0]).desc;
        TensorDesc outputDesc = (this->outputTensors[0]).desc;
        ConvolutionDesc convDesc = create_convDesc(this->kernelStride, this->kernelPadding, this->dilation);

        U32 bytes = 0;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->pwAlg, &bytes, A));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->dwAlg, &bytes, A));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->dwAlg, &bytes, A));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->pwAlg, &bytes, A));
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return bytes;
    }

    U32 infer_wtm_memory_size()
    {
        TensorDesc filterDesc = (this->weightTensors[0]).desc;
        U32 bytes = 0;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_transform_filter_bytes(filterDesc, this->pwAlg, &bytes, A));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_transform_filter_bytes(filterDesc, this->dwAlg, &bytes, A));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_convolution_transform_filter_bytes(filterDesc, this->dwAlg, &bytes, A));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(convolution_transform_filter_bytes(filterDesc, this->pwAlg, &bytes, A));
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return bytes;
    }

    EE transform_filter()
    {
        Tensor filterTensor = this->weightTensors[0];
        TensorDesc filterDesc = filterTensor.get_desc();
        U8* weightPtr = filterTensor.get_val().get();

        TensorDesc wtmDesc;
        if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->convolutionType && CONVOLUTION_ALGORITHM_WINOGRAD == this->pwAlg) { // int8 winograd
            U32 ftBytes;
            CHECK_STATUS_WITH_RETURN(convolution_transform_filter_bytes(filterDesc, this->pwAlg, &ftBytes, A));

            TensorDesc tFilterDesc;
            F16 *tFilter = (F16*)malloc(ftBytes);
            if (nullptr == tFilter) {
                std::cerr << "[ERROR] allocation failed for filter transform in int8 winograd" << std::endl;
                return ALLOC_FAILED;
            }

            filterDesc.dt = DT_F16_8Q; // To label as int8
            CHECK_STATUS_WITH_RETURN(convolution_transform_filter(filterDesc, weightPtr, this->pwAlg, &tFilterDesc, tFilter, A));

            U32 ftmBytes = ftBytes / bytesOf(DT_F16);
            std::shared_ptr<U8> sPtr((U8*) operator new(ftmBytes));
            this->set_wtm_memory(ftmBytes, sPtr);

            std::shared_ptr<F16> fsp((F16*) operator new(38*bytesOf(DT_F16)));
            this->scales = fsp;
            CHECK_STATUS_WITH_RETURN(quantize_tensor(tFilterDesc, tFilter, &wtmDesc, this->get_wtm().get(), this->scales.get()+2));
            free(tFilter);
        } else if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->convolutionType) { // int8 tilegemm
            TensorDesc qFilterDesc;
            INT8 *qFilter = (INT8*)malloc(tensorNumElements(filterDesc) * bytesOf(DT_I8));
            if (nullptr == qFilter) {
                std::cerr << "[ERROR] allocation failed for filter quantization" << std::endl;
                return ALLOC_FAILED;
            }
            std::shared_ptr<F16> fsp((F16*) operator new(3*bytesOf(DT_F16)));
            this->scales = fsp;
            CHECK_STATUS_WITH_RETURN(quantize_tensor(filterDesc, weightPtr, &qFilterDesc, qFilter, this->scales.get()+2));

            U32 ftmBytes;
            CHECK_STATUS_WITH_RETURN(convolution_transform_filter_bytes(qFilterDesc, this->pwAlg, &ftmBytes, A));
                
            std::shared_ptr<U8> sPtr((U8*) operator new(ftmBytes));
            this->set_wtm_memory(ftmBytes, sPtr);

            // trans filter
            CHECK_STATUS_WITH_RETURN(convolution_transform_filter(qFilterDesc, qFilter, this->pwAlg,
                                                &wtmDesc, this->get_wtm().get(), A));

            free(qFilter);
        } else { // All other cases
            auto wtmBytes = this->infer_wtm_memory_size();
            std::shared_ptr<U8> sPtr((U8*) operator new(wtmBytes));
            this->set_wtm_memory(wtmBytes, sPtr);

            switch (this->convolutionType) {
                case Convolution_Pointwise: {
                    CHECK_STATUS_WITH_RETURN(convolution_transform_filter(filterDesc, weightPtr, this->pwAlg, &wtmDesc, this->get_wtm().get(), A));
                    break;
                }
                case Convolution_Depthwise: {
                    CHECK_STATUS_WITH_RETURN(depthwise_convolution_transform_filter(filterDesc, weightPtr, this->dwAlg, &wtmDesc, this->get_wtm().get(), A));
                    break;
                }
                case Convolution_Depthwise_Pointwise: {
                    CHECK_STATUS_WITH_RETURN(depthwise_convolution_transform_filter(filterDesc, weightPtr, this->dwAlg, &wtmDesc, this->get_wtm().get(), A));
                    break;
                }
                case Convolution_Dilation: {
                    CHECK_STATUS_WITH_RETURN(convolution_transform_filter(filterDesc, weightPtr, this->pwAlg, &wtmDesc, this->get_wtm().get(), A));
                    break;
                }
                default:
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
            }
        }

        Tensor wtmTensor = Tensor(wtmDesc, this->get_wtm());
        this->weightTensors[0] = wtmTensor;
        return SUCCESS;
    }

public:
    U32 numFilters;
    U32 numChannels;
    U32 kernelSize;
    U32 kernelStride;
    U32 kernelPadding;
    ConvolutionMode convolutionType;
    U32 group;
    U32 dilation;
    std::optional<EltwiseType> eltwiseType;
    std::optional<PoolingMode> poolingMode;
    ActivationMode dwActiveMode;
    ActivationMode pwActiveMode;
    U32 poolingSize;
    U32 poolingStride;
    U32 poolingPadding;
    ConvolutionForwardAlgorithm pwAlg;
    DepthwiseConvolutionForwardAlgorithm dwAlg;
    std::shared_ptr<F16> scales;
};

#endif //_CONVELTWISEPOOLING_H
