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
#include <optional>
#include <stdio.h>
#include "weight_operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "op_type.h"
#include "convolution.hpp"
#include "pooling.hpp"
#include "eltwise.hpp"

class ConvolutionCPU: public Convolution {
public:
    ConvolutionCPU(DataType dt, U32 nf, U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW, U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationMode dwActiveMode, ActivationMode pwActiveMode, ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW) :
        Convolution(dt, nf, ksizeH, ksizeW, kstrideH, kstrideW, kpaddingT, kpaddingB, kpaddingL, kpaddingR, 
            dwActiveMode, pwActiveMode, convolutionType, group, dilateH, dilateW) {}

    virtual EE init_weight_bias_from_model(U8** modelPtr)override
    {
        auto curOpWs = this->get_weightspec_ptr();
        DataType filterDt = curOpWs.mdt; // weight data type may not be the same as input and output
        if (modelPtr != nullptr) {
            filterDt = this->dt;
        }
        DataType dtNoQ = (this->dt == DT_F16_8Q) ? DT_F16 : this->dt;
        U32 isBNN = 0;
        if (filterDt == DT_BIN01 || filterDt == DT_BIN11) {
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
                return NOT_SUPPORTED;
        }
        TensorDesc filterTensorDesc = tensor4df(filterDt, filterDf,
                                                 this->numFilters, this->numChannels,
                                                 this->kernelSizeH, this->kernelSizeW);
        TensorDesc vectorTensorDesc = tensor1d(dtNoQ, vectorLen); // bias data type should be the same as input and output

        std::shared_ptr<Tensor> modelWeightTensor(new Tensor());
        std::shared_ptr<Tensor> modelVectorTensor(new Tensor());
        modelWeightTensor->set_desc(filterTensorDesc);
        modelVectorTensor->set_desc(vectorTensorDesc);

        if (modelPtr != nullptr) {
            modelWeightTensor->alloc();
            memcpy((U8*)modelWeightTensor->get_val(), *modelPtr, tensorNumBytes(filterTensorDesc));
            *modelPtr += tensorNumBytes(filterTensorDesc);
        } else {
            modelWeightTensor->set_val(curOpWs.weight);
        }
       
        U8* biasVal = NULL;
        if(modelPtr != nullptr) {
            if(this->hasBias){
                biasVal = *modelPtr; 
                *modelPtr += tensorNumBytes(vectorTensorDesc);
            }
        } else {
            if(this->hasBias) biasVal = curOpWs.vec;
        }

        if (biasVal) {
            modelVectorTensor->set_val(biasVal);
        } else {
            modelVectorTensor->alloc();
            if (isBNN == 1) {
#ifdef _USE_FP16
                F16 *vec = (F16*)modelVectorTensor->get_val();
                for (U32 i = 0; i < this->numFilters; i++) { // first half is scale
                    *vec = 1.0;
                    vec++;
                }
                memset(vec, 0, tensorNumBytes(vectorTensorDesc) / 2); // second half is bias 
#endif
            } else {
                memset((U8*)modelVectorTensor->get_val(), 0, tensorNumBytes(vectorTensorDesc));
            }
        }

        this->weightTensors.push_back(*modelWeightTensor.get());
        this->biasTensors.push_back(*modelVectorTensor.get());
        return SUCCESS;
    }

    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor filterTensor = this->weightTensors[0];
        TensorDesc filterDesc = filterTensor.get_desc();

        ConvolutionDesc convDesc = Convolution::create_convDesc(this->strideH, this->strideW, this->paddingT, this->paddingB, 
            this->paddingL, this->paddingR, this->dilateH, this->dilateW);

        TensorDesc scaleDesc = filterDesc; // Dummy initialization
        U8 *scalePtr = nullptr;

        Tensor biasTensor = this->biasTensors[0];
        TensorDesc biasDesc = biasTensor.get_desc();
        U8 *biasPtr = biasTensor.get_val();

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                if (filterDesc.dt == DT_BIN01 || filterDesc.dt == DT_BIN11) {
#ifdef _USE_FP16
                    U32 vecLen = tensorNumElements(biasDesc) / 2;

                    scaleDesc = tensor1d(biasDesc.dt, vecLen);
                    biasDesc = tensor1d(biasDesc.dt, vecLen);
                    scalePtr = biasTensor.get_val();
                    biasPtr = scalePtr + vecLen * bytesOf(DT_F16);
#endif
                } else if (DT_F16_8Q == this->dt) {
#ifdef _USE_INT8
                    F16 *ptr = this->scales.get();
                    ptr[0] = inputTensor.get_scale();
                    scalePtr = (U8*)ptr;
#endif
                }
                CHECK_STATUS(convolution(inputDesc, inputTensor.get_val(),
                                         filterDesc, filterTensor.get_val(),
                                         convDesc, this->pwAlg,
                                         scaleDesc, scalePtr,
                                         biasDesc, biasPtr,
                                         this->lenOfTemp, this->temp.get(),
                                         outputDesc, (void*)outputTensor.get_val(),
                                         this->pwActiveMode, this->schedule));
#ifdef _USE_INT8
                if (DT_I8 == outputDesc.dt) {
                    F16 *ptr = (F16*)scalePtr;
                    outputTensor.set_scale(ptr[1]);
                }
#endif
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution(inputDesc, inputTensor.get_val(),
                                                   filterDesc, filterTensor.get_val(),
                                                   convDesc, this->dwAlg,
                                                   biasDesc, biasPtr,
                                                   this->lenOfTemp, this->temp.get(),
                                                   outputDesc, outputTensor.get_val(),
                                                   this->dwActiveMode, ACTIVATION_NULL,
                                                   this->schedule));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_convolution(inputDesc, inputTensor.get_val(),
                                                   filterDesc, filterTensor.get_val(),
                                                   convDesc, this->dwAlg,
                                                   biasDesc, biasPtr,
                                                   this->lenOfTemp, this->temp.get(),
                                                   outputDesc, outputTensor.get_val(),
                                                   this->dwActiveMode, this->pwActiveMode,
                                                   this->schedule));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(convolution(inputDesc, inputTensor.get_val(),
                                         filterDesc, filterTensor.get_val(),
                                         convDesc, this->pwAlg,
                                         scaleDesc, scalePtr,
                                         biasDesc, biasPtr,
                                         this->lenOfTemp, this->temp.get(),
                                         outputDesc, (void*)outputTensor.get_val(),
                                         this->pwActiveMode, this->schedule));
                break;
            }
            default:
                std::cerr << "[ERROR] unsupported convolution type " << this->convolutionType << std::endl;
                exit(1);
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_forward_algorithm(HashMap<std::string, int> &algorithmMap) override
    {
        TensorDesc inputDesc = (this->inputTensors[0]).get_desc();
        TensorDesc filterDesc = (this->weightTensors[0]).get_desc();

        ConvolutionPolicy policy = CONVOLUTION_FASTEST;
        ConvolutionDesc convDesc = Convolution::create_convDesc(this->strideH, this->strideW, this->paddingT, this->paddingB,
            this->paddingL, this->paddingR, this->dilateH, this->dilateW);

        DataType targetType = filterDesc.dt;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                if (this->dt == DT_F16_8Q) {
                    targetType = DT_I8;
                }
                if (algorithmMap.find(this->name) != algorithmMap.end()) {
                    this->pwAlg = (ConvolutionForwardAlgorithm)algorithmMap[this->name];
                } else {
                    CHECK_STATUS(convolution_infer_forward_algorithm(inputDesc, filterDesc,
                                                 this->outputTensors[0].get_desc(),
                                                 convDesc, policy, &(this->pwAlg), targetType, this->pwActiveMode, this->schedule));
                    algorithmMap[this->name] = this->pwAlg;
                }
                break;
            }
            case Convolution_Depthwise: {
                if (algorithmMap.find(this->name) != algorithmMap.end()) {
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algorithmMap[this->name];
                } else {
                    CHECK_STATUS(depthwise_convolution_infer_forward_algorithm(inputDesc, filterDesc,
                                                 this->outputTensors[0].get_desc(),
                                                 convDesc, policy, &(this->dwAlg),
                                                 targetType, this->dwActiveMode, ACTIVATION_NULL, this->schedule));
                    algorithmMap[this->name] = this->dwAlg;
                }
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                if (algorithmMap.find(this->name) != algorithmMap.end()) {
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algorithmMap[this->name];
                } else {
                    CHECK_STATUS(depthwise_convolution_infer_forward_algorithm(inputDesc, filterDesc,
                                                 this->outputTensors[0].get_desc(),
                                                 convDesc, policy, &(this->dwAlg), 
                                                 targetType, this->dwActiveMode, this->pwActiveMode, this->schedule));
                    algorithmMap[this->name] = this->dwAlg;
                }
                break;
            }
            case Convolution_Dilation: {
                if (algorithmMap.find(this->name) != algorithmMap.end()) {
                    this->pwAlg = (ConvolutionForwardAlgorithm)algorithmMap[this->name];
                } else {
                    CHECK_STATUS(convolution_infer_forward_algorithm(inputDesc, filterDesc,
                                                 this->outputTensors[0].get_desc(),
                                                 convDesc, policy, &(this->pwAlg), targetType, this->pwActiveMode, this->schedule));
                    algorithmMap[this->name] = this->pwAlg;
                }
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return SUCCESS;
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inDim = inDims[0];
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        CHECK_STATUS(tensor4dGet(inDim, &idt, &idf, &in, &ic, &ih, &iw));
        this->numChannels = ic;

        TensorDesc filterDim = tensor4df(this->dt, DF_NCHW, this->numFilters, this->numChannels, this->kernelSizeH,
                                    this->kernelSizeW);

        if (Convolution_Depthwise_Pointwise == this->convolutionType) {
            filterDim.df = DF_CHW_NC;
        }

        ConvolutionDesc convDesc = Convolution::create_convDesc(this->strideH, this->strideW, this->paddingT, this->paddingB,
            this->paddingL, this->paddingR, this->dilateH, this->dilateW);

        DataType targetType = this->dt;
        if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->convolutionType) {
            targetType = DT_I8;
        }

        U32 outBytes = 0;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes, this->schedule));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes, this->schedule));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_convolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes, this->schedule));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(convolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes, this->schedule));
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return SUCCESS;
    }

    virtual U32 infer_tmp_memory_size() override
    {
        TensorDesc inputDesc = (this->inputTensors[0]).desc;
        TensorDesc filterDesc = (this->weightTensors[0]).desc;
        TensorDesc outputDesc = (this->outputTensors[0]).desc;
        ConvolutionDesc convDesc = Convolution::create_convDesc(this->strideH, this->strideW, this->paddingT, this->paddingB,
            this->paddingL, this->paddingR, this->dilateH, this->dilateW);

        U32 bytes = 0;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->pwAlg, &bytes, this->schedule));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->dwAlg, &bytes, this->schedule));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->dwAlg, &bytes, this->schedule));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->pwAlg, &bytes, this->schedule));
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return bytes;
    }

    virtual U32 infer_wtm_memory_size() override
    {
        TensorDesc filterDesc = (this->weightTensors[0]).desc;
        U32 bytes = 0;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_transform_filter_bytes(filterDesc, this->pwAlg, &bytes, this->schedule));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_transform_filter_bytes(filterDesc, this->dwAlg, &bytes, this->schedule));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_convolution_transform_filter_bytes(filterDesc, this->dwAlg, &bytes, this->schedule));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(convolution_transform_filter_bytes(filterDesc, this->pwAlg, &bytes, this->schedule));
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return bytes;
    }

    virtual EE transform_filter() override
    {
        Tensor filterTensor = this->weightTensors[0];
        TensorDesc filterDesc = filterTensor.get_desc();
        U8* weightPtr = filterTensor.get_val();
        this->wtm = std::shared_ptr<Tensor>(new Tensor());

        TensorDesc wtmDesc;
        if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->convolutionType && CONVOLUTION_ALGORITHM_WINOGRAD == this->pwAlg) { // int8 winograd
#ifdef _USE_INT8
            U32 ftBytes;
            CHECK_STATUS(convolution_transform_filter_bytes(filterDesc, this->pwAlg, &ftBytes, this->schedule));

            TensorDesc tFilterDesc;
            F16 *tFilter = (F16*)malloc(ftBytes);
            if (nullptr == tFilter) {
                std::cerr << "[ERROR] allocation failed for filter transform in int8 winograd" << std::endl;
                CHECK_STATUS(ALLOC_FAILED);
            }

            filterDesc.dt = DT_F16_8Q; // To label as int8
            CHECK_STATUS(convolution_transform_filter(filterDesc, weightPtr, this->pwAlg, &tFilterDesc, tFilter, this->schedule));

            U32 ftmBytes = ftBytes / bytesOf(DT_F16);
            std::shared_ptr<U8> sPtr((U8*) operator new(ftmBytes));
            auto cpuMem = new CpuMemory();
            cpuMem->set_shared_ptr_caster(sPtr);
            Memory_* mem = (Memory_*)(cpuMem);
            std::shared_ptr<Memory_> memsPtr(mem);
            this->set_wtm_memory(ftmBytes, memsPtr);

            std::shared_ptr<F16> fsp((F16*) operator new(38*bytesOf(DT_F16)));
            this->scales = fsp;
            CHECK_STATUS(quantize_tensor(tFilterDesc, tFilter, &wtmDesc, this->get_wtm()->get_val(), this->scales.get()+2));
            free(tFilter);
        } else if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->convolutionType) { // int8 tilegemm
            TensorDesc qFilterDesc;
            INT8 *qFilter = (INT8*)malloc(tensorNumElements(filterDesc) * bytesOf(DT_I8));
            if (nullptr == qFilter) {
                std::cerr << "[ERROR] allocation failed for filter quantization" << std::endl;
                CHECK_STATUS(ALLOC_FAILED);
            }
            std::shared_ptr<F16> fsp((F16*) operator new(3*bytesOf(DT_F16)));
            this->scales = fsp;
            CHECK_STATUS(quantize_tensor(filterDesc, weightPtr, &qFilterDesc, qFilter, this->scales.get()+2));

            U32 ftmBytes;
            CHECK_STATUS(convolution_transform_filter_bytes(qFilterDesc, this->pwAlg, &ftmBytes, this->schedule));
                
            std::shared_ptr<U8> sPtr((U8*) operator new(ftmBytes));
            auto cpuMem = new CpuMemory();
            cpuMem->set_shared_ptr_caster(sPtr);
            Memory_* mem = (Memory_*)(cpuMem);
            std::shared_ptr<Memory_> memsPtr(mem);
            this->set_wtm_memory(ftmBytes, memsPtr);

            // trans filter
            CHECK_STATUS(convolution_transform_filter(qFilterDesc, qFilter, this->pwAlg,
                                                &wtmDesc, this->get_wtm()->get_val(), this->schedule));

            free(qFilter);
#endif
        } else { // All other cases
            auto wtmBytes = this->infer_wtm_memory_size();
            std::shared_ptr<U8> sPtr((U8*) operator new(wtmBytes));
            auto cpuMem = new CpuMemory();
            cpuMem->set_shared_ptr_caster(sPtr);
            Memory_* mem = (Memory_*)(cpuMem);
            std::shared_ptr<Memory_> memsPtr(mem);
            this->set_wtm_memory(wtmBytes, memsPtr);

            switch (this->convolutionType) {
                case Convolution_Pointwise: {
                    CHECK_STATUS(convolution_transform_filter(filterDesc, weightPtr, this->pwAlg, &wtmDesc, this->get_wtm()->get_val(), this->schedule));
                    break;
                }
                case Convolution_Depthwise: {
                    CHECK_STATUS(depthwise_convolution_transform_filter(filterDesc, weightPtr, this->dwAlg, &wtmDesc, this->get_wtm()->get_val(), this->schedule));
                    break;
                }
                case Convolution_Depthwise_Pointwise: {
                    CHECK_STATUS(depthwise_convolution_transform_filter(filterDesc, weightPtr, this->dwAlg, &wtmDesc, this->get_wtm()->get_val(), this->schedule));
                    break;
                }
                case Convolution_Dilation: {
                    CHECK_STATUS(convolution_transform_filter(filterDesc, weightPtr, this->pwAlg, &wtmDesc, this->get_wtm()->get_val(), this->schedule));
                    break;
                }
                default:
                    CHECK_STATUS(NOT_SUPPORTED);
            }
        }

        this->get_wtm()->set_desc(wtmDesc);
        this->weightTensors[0] = *this->get_wtm();
        return SUCCESS;
    }
public:
};

#endif //_CONVELTWISEPOOLING_H
