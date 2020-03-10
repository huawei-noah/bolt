// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _DECONVOLUTION_H
#define _DECONVOLUTION_H
#include "weight_operator.hpp"
#include "tensor_computing.h"
#include "op_type.h"

class Deconvolution: public WeightOperator {
public:
    Deconvolution(DataType dt, U32 nf, U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW, U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationMode dwActiveMode, ActivationMode pwActiveMode,
        ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW)
    {
        this->dt = dt;
        this->numFilters = nf;
        this->kernelSizeH = ksizeH;
        this->kernelSizeW = ksizeW;
        this->strideH = kstrideH;
        this->strideW = kstrideW;
        this->paddingT = kpaddingT;
        this->paddingB = kpaddingB;
        this->paddingL = kpaddingL;
        this->paddingR = kpaddingR;
        this->dwActiveMode = dwActiveMode;
        this->pwActiveMode = pwActiveMode;
        this->convolutionType = convolutionType;
        this->group = group;
        this->dilateH = dilateH;
        this->dilateW = dilateW;
        this->hasBias = false;
    }

    OperatorType get_op_type() override
    {
        return OT_Deconvolution;
    }

    ConvolutionDesc create_convDesc(U32 strideH, U32 strideW, U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, U32 dilateH, U32 dilateW)
    {
        ConvolutionDesc convDesc;
        convDesc.stride_h = strideH;
        convDesc.stride_w = strideW;
        convDesc.padding_top = paddingT;
        convDesc.padding_bottom = paddingB;
        convDesc.padding_left = paddingL;
        convDesc.padding_right = paddingR;
        convDesc.dilatedRate_h = dilateH;
        convDesc.dilatedRate_w = dilateW;
        return convDesc;
    }
    
    EE init_weight_bias_from_model(U8** modelPtr)
    {
        auto curOpWs = this->get_weightspec_ptr();
        DataType filterDt = curOpWs.mdt;  // weight data type may not be the same as input and output
        if (modelPtr != nullptr) {
            filterDt = DT_F16;
        }
        DataType dtNoQ = (this->dt == DT_F16_8Q) ? DT_F16 : this->dt;
        U32 isBNN = 0;
        if (filterDt == DT_BIN01 || filterDt == DT_BIN11) {
            isBNN = 1;
        }
        DataFormat filterDf;
        U32 vectorLen = 0;  // Vector must contain bias. BNN has one more scale vector.
        switch (this->convolutionType) {
            case Convolution_Deconvolution: {
                filterDf = DF_NCHW;
                vectorLen = this->numFilters;  // bias length
                if (isBNN == 1) {
                    this->dt = dtNoQ;  // BNN convolution should not be quantized further
                    vectorLen *= 2;  // Scale has the same vector length as bias, so double the length
                }
                break;
            }
            default: {
                CHECK_STATUS(NOT_SUPPORTED);
                return NOT_SUPPORTED;
            }
        }
        TensorDesc filterTensorDesc = tensor4df(filterDt, filterDf,
                                                 this->numFilters, this->numChannels,
                                                 this->kernelSizeH, this->kernelSizeW);
        TensorDesc vectorTensorDesc = tensor1d(dtNoQ, vectorLen);  // bias data type should be the same as input and output

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

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor filterTensor = this->weightTensors[0];
        TensorDesc filterDesc = filterTensor.get_desc();

        ConvolutionDesc convDesc = create_convDesc(this->strideH, this->strideW,
                                this->paddingT, this->paddingB, this->paddingL, this->paddingR, this->dilateH, this->dilateW);

        TensorDesc scaleDesc = filterDesc; // Dummy initialization
        U8 *scalePtr = nullptr;

        Tensor biasTensor = this->biasTensors[0];
        TensorDesc biasDesc = biasTensor.get_desc();
        U8 *biasPtr = (U8*)biasTensor.get_val();

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        switch (this->convolutionType) {
            case Convolution_Deconvolution: {
                if (filterDesc.dt == DT_BIN01 || filterDesc.dt == DT_BIN11) {
                    CHECK_STATUS(NOT_SUPPORTED);
                }
                CHECK_STATUS(deconvolution(inputDesc, inputTensor.get_val(),
                                         filterDesc, filterTensor.get_val(),
                                         convDesc, this->pwAlg,
                                         scaleDesc, scalePtr,
                                         biasDesc, biasPtr,
                                         this->lenOfTemp, this->temp.get(),
                                         outputDesc, (void*)outputTensor.get_val(),
                                         this->pwActiveMode, this->schedule));
                break;
            }
            default: {
                std::cerr << "[ERROR] unsupported deconvolution type " << convolutionType << std::endl;
                exit(1);
            }
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_forward_algorithm()
    {
        TensorDesc inputDesc = (this->inputTensors[0]).get_desc();
        TensorDesc filterDesc = (this->weightTensors[0]).get_desc();

        ConvolutionPolicy policy = CONVOLUTION_FASTEST;
        ConvolutionDesc convDesc = create_convDesc(this->strideH, this->strideW,
                                this->paddingT, this->paddingB, this->paddingL, this->paddingR, this->dilateH, this->dilateW);

        DataType targetType = filterDesc.dt;
        switch (this->convolutionType) {
            case Convolution_Deconvolution: {
                CHECK_STATUS(deconvolution_infer_forward_algorithm(inputDesc, filterDesc,
                                             this->outputTensors[0].get_desc(),
                                             convDesc, policy, &(this->pwAlg), targetType, this->schedule));
                break;
            }
            default: {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        return SUCCESS;
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inDim = inDims[0];
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        CHECK_STATUS(tensor4dGet(inDim, &idt, &idf, &in, &ic, &ih, &iw));
        this->numChannels = ic;

        TensorDesc filterDim = tensor4df(this->dt, DF_NCHW, this->numFilters, this->numChannels, this->kernelSizeH,
                                    this->kernelSizeW);

        ConvolutionDesc convDesc = create_convDesc(this->strideH, this->strideW,
                                this->paddingT, this->paddingB, this->paddingL, this->paddingR, this->dilateH, this->dilateW);

        DataType targetType = DT_F16; // Default DT_F16

        U32 outBytes = 0;
        switch (this->convolutionType) {
            case Convolution_Deconvolution: {
                CHECK_STATUS(deconvolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes));
                break;
            }
            default: {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        TensorDesc inputDesc = (this->inputTensors[0]).desc;
        TensorDesc filterDesc = (this->weightTensors[0]).desc;
        TensorDesc outputDesc = (this->outputTensors[0]).desc;
        ConvolutionDesc convDesc = create_convDesc(this->strideH, this->strideW,
                                this->paddingT, this->paddingB, this->paddingL, this->paddingR, this->dilateH, this->dilateW);

        U32 bytes = 0;
        switch (this->convolutionType) {
            case Convolution_Deconvolution: {
                CHECK_STATUS(deconvolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->pwAlg, &bytes, this->schedule));
                break;
            }
            default: {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        return bytes;
    }

    U32 infer_wtm_memory_size() override
    {
        TensorDesc filterDesc = (this->weightTensors[0]).desc;
        U32 bytes = 0;
        switch (this->convolutionType) {
            case Convolution_Deconvolution: {
                CHECK_STATUS(deconvolution_transform_filter_bytes(filterDesc, this->pwAlg, &bytes, this->schedule));
                break;
            }
            default: {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        return bytes;
    }

    EE transform_filter()
    {
        this->wtm = std::shared_ptr<Tensor>(new Tensor());
        Tensor filterTensor = this->weightTensors[0];
        TensorDesc filterDesc = filterTensor.get_desc();
        U8* weightPtr = filterTensor.get_val();

        TensorDesc wtmDesc;
        
        auto wtmBytes = this->infer_wtm_memory_size();
        std::shared_ptr<U8> sPtr((U8*) operator new(wtmBytes));
        auto cpuMem = new CpuMemory();
        cpuMem->set_shared_ptr_caster(sPtr);
        Memory_* mem = (Memory_*)(cpuMem);
        std::shared_ptr<Memory_> memsPtr(mem);
        this->set_wtm_memory(wtmBytes, memsPtr);

        switch (this->convolutionType) {
            case Convolution_Deconvolution: {
                CHECK_STATUS(deconvolution_transform_filter(filterDesc, weightPtr, this->pwAlg, &wtmDesc, this->get_wtm()->get_val(), this->schedule));
                break;
            }
            default: {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }

        this->get_wtm()->set_desc(wtmDesc);
        this->weightTensors[0] = *this->get_wtm();
        return SUCCESS;
    }

public:
    U32 numFilters;
    U32 numChannels;
    U32 kernelSizeH;
    U32 kernelSizeW;
    U32 strideH;
    U32 strideW;
    U32 paddingT;
    U32 paddingB;
    U32 paddingL;
    U32 paddingR;
    ConvolutionMode convolutionType;
    U32 group;
    U32 dilateH;
    U32 dilateW;

    ActivationMode dwActiveMode;
    ActivationMode pwActiveMode;

    ConvolutionForwardAlgorithm pwAlg;
    DepthwiseConvolutionForwardAlgorithm dwAlg;
};

#endif  //_DECONVOLUTION_H
