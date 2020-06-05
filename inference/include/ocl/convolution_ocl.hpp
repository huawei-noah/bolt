// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _CONVELTWISEPOOLING_OCL_H
#define _CONVELTWISEPOOLING_OCL_H
#include "weight_operator.hpp"
#include "pooling.hpp"
#include "eltwise.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "op_type.h"
#include "convolution.hpp"
#include <stdio.h>

class ConvolutionOCL: public Convolution {
public:
    ConvolutionOCL(DataType dt, U32 nf, U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW, U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationDesc dwActivationDesc, ActivationDesc pwActivationDesc, ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW) :
        Convolution(dt, nf, ksizeH, ksizeW, kstrideH, kstrideW, kpaddingT, kpaddingB, kpaddingL, kpaddingR,
            dwActivationDesc, pwActivationDesc, convolutionType, group, dilateH, dilateW) {}

    virtual EE init_weight_bias_from_model(U8** modelPtr)override
    {
        auto curOpWs = this->get_weightspec_ptr();
        DataType filterDt = curOpWs.mdt; // weight data type may not be the same as input and output
        if (modelPtr != nullptr) {
            filterDt = DT_F16;
        }
        DataType dtNoQ = (this->dt == DT_F16_8Q) ? DT_F16 : this->dt;
        DataFormat filterDf;
        U32 vectorLen = 0; // Vector must contain bias. BNN has one more scale vector.
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                filterDf = DF_NCHW;
                vectorLen = this->numFilters; // bias length
                this->oclExtInfo.maliInfo.forwardRunInfo->algorithm = CONVOLUTION_ALGORITHM_NULL;
                break;
            }
            case Convolution_Depthwise: {
                filterDf = DF_NCHW;
                vectorLen = this->numFilters;
                this->oclExtInfo.maliInfo.forwardRunInfo->algorithm = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                filterDf = DF_CHW_NC;
                vectorLen = this->numFilters + this->numChannels;
                this->oclExtInfo.maliInfo.forwardRunInfo->algorithm = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;    
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(NOT_SUPPORTED);
                return NOT_SUPPORTED;
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
                return NOT_SUPPORTED;
        }
        TensorDesc filterTensorDesc = tensor4df(filterDt, filterDf,
                                                this->numFilters, this->numChannels,
                                                this->kernelSizeH, this->kernelSizeW);
        if(this->convolutionType == Convolution_Depthwise) filterTensorDesc.dims[2] = 1;                                                
        TensorDesc vectorTensorDesc = tensor1d(dtNoQ, vectorLen); // bias data type should be the same as input and output

        std::shared_ptr<Tensor> modelWeightTensor(new Tensor(this->handle));
        std::shared_ptr<Tensor> modelVectorTensor(new Tensor(this->handle));
        std::shared_ptr<Tensor> modelWeightTensorExt;
        std::shared_ptr<Tensor> modelVectorTensorExt;
        modelWeightTensor->set_desc(filterTensorDesc);
        GCLMem_t weightMem = modelWeightTensor->get_val();
        U32 ww, wh, wc, wn;
        DataFormat df;
        DataType dt;
        U32 num, bytes;
        TensorDesc filterTensorDescTmp = filterTensorDesc;
        TensorDesc filterTensorDescExt = filterTensorDesc;
        if(this->convolutionType == Convolution_Depthwise_Pointwise){
            filterTensorDescTmp.dims[2] = 1;
            filterTensorDescTmp.dims[3] = this->numChannels;
            filterTensorDescExt.dims[0] = 1;
            filterTensorDescExt.dims[1] = 1;
            filterTensorDescExt.dims[2] = this->numChannels;
            filterTensorDescExt.dims[3] = this->numFilters;
            filterTensorDescExt.df      = DF_NCHW;
            modelWeightTensorExt        = std::shared_ptr<Tensor>(new Tensor(this->handle));
            modelWeightTensorExt->set_desc(filterTensorDescExt);
            GCLMem_t weightMemExt = modelWeightTensorExt->get_val();
            tensorSelectGet(filterTensorDescExt, &dt, &df, &wn, &wc, &wh, &ww);
            num   = tensorNumElements(filterTensorDescExt);
            bytes = tensorNumBytes(filterTensorDescExt);
            weightMemExt->desc.stride[0] = ww * wh;
            weightMemExt->desc.stride[1] = wc;
            weightMemExt->desc.stride[2] = wn;
            weightMemExt->desc.offset[0] = 0;
            weightMemExt->desc.offset[1] = 0;
            weightMemExt->desc.offset[2] = 0;
            weightMemExt->desc.memType   = GCL_MEM_BUF;
            weightMemExt->desc.memFormat = df;
            weightMemExt->desc.byteSize  = bytes;
            weightMemExt->desc.num       = num;
            weightMemExt->desc.flags     = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        }
        tensorSelectGet(filterTensorDescTmp, &dt, &df, &wn, &wc, &wh, &ww);
        num   = tensorNumElements(filterTensorDescTmp);
        bytes = tensorNumBytes(filterTensorDescTmp);
        weightMem->desc.stride[0] = ww * wh;
        weightMem->desc.stride[1] = wc;
        weightMem->desc.stride[2] = wn;
        weightMem->desc.offset[0] = 0;
        weightMem->desc.offset[1] = 0;
        weightMem->desc.offset[2] = 0;
        weightMem->desc.memType   = GCL_MEM_BUF;
        weightMem->desc.memFormat = df;
        weightMem->desc.byteSize  = bytes;
        weightMem->desc.num       = num;
        weightMem->desc.flags     = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        GCLMem_t vectorMem = modelVectorTensor->get_val();
        modelVectorTensor->set_desc(vectorTensorDesc);
        U32 vectorLenTmp = vectorLen;
        U32 vectorLenExt = vectorLen;
        TensorDesc vectorTensorDescTmp = vectorTensorDesc;
        TensorDesc vectorTensorDescExt = vectorTensorDesc;
        if(this->convolutionType == Convolution_Depthwise_Pointwise) {
            vectorLenTmp = this->numChannels;
            vectorLenExt = this->numFilters;
            vectorTensorDescTmp.dims[0] = vectorLenTmp;
            vectorTensorDescExt.dims[0] = vectorLenExt;
            modelVectorTensorExt        = std::shared_ptr<Tensor>(new Tensor(this->handle));
            modelVectorTensorExt->set_desc(vectorTensorDescExt);
            GCLMem_t vectorMemExt = modelVectorTensorExt->get_val();
            vectorMemExt->desc.stride[0] = (vectorLenExt + 3) / 4;
            vectorMemExt->desc.stride[1] = 1;
            vectorMemExt->desc.stride[2] = 1;
            vectorMemExt->desc.offset[0] = 0;
            vectorMemExt->desc.offset[1] = 0;
            vectorMemExt->desc.offset[2] = 0;
            vectorMemExt->desc.memType   = GCL_MEM_IMG_1D;
            vectorMemExt->desc.byteSize  = (vectorLenExt + 3) / 4 * 4 * bytesOf(dtNoQ);
            vectorMemExt->desc.num       = (vectorLenExt + 3) / 4;
            vectorMemExt->desc.memFormat = DF_NHWC;
        }
        U32 iw, ih;
        TensorDesc inputDesc = this->inputTensors[0].get_desc();
        tensorSelectGet(inputDesc, NULL, NULL, NULL, NULL, &ih, &iw);
        if(wn == 1 || (ww == 1 && wh == 1 && iw == 1 && ih == 1)) {
            vectorMem->desc.stride[0] = (vectorLenTmp + 7) / 8 * 8;
            vectorMem->desc.memType   = GCL_MEM_BUF;
            vectorMem->desc.byteSize  = (vectorLenTmp  + 7) / 8 * 8 * sizeof(dtNoQ);
            vectorMem->desc.num       = (vectorLenTmp + 7) / 8 * 8;
        }else{
            vectorMem->desc.stride[0] = (vectorLenTmp + 3) / 4;
            vectorMem->desc.memType   = GCL_MEM_IMG_1D;
            vectorMem->desc.byteSize  = (vectorLenTmp + 3) / 4 * 4 * bytesOf(dtNoQ);
            vectorMem->desc.num       = (vectorLenTmp + 3) / 4;
        }
        vectorMem->desc.stride[1] = 1;
        vectorMem->desc.stride[2] = 1;
        vectorMem->desc.offset[0] = 0;
        vectorMem->desc.offset[1] = 0;
        vectorMem->desc.offset[2] = 0;
        vectorMem->desc.memFormat = DF_NHWC;

        if (modelPtr != nullptr) {
            weightMem->desc.host_ptr = *modelPtr;
            if(this->convolutionType == Convolution_Depthwise_Pointwise){
                GCLMem_t weightMemExt = modelWeightTensorExt->get_val();
                weightMemExt->desc.host_ptr = *modelPtr + weightMem->desc.byteSize;
            }
            *modelPtr += tensorNumBytes(filterTensorDesc);
        } else {
            weightMem->desc.host_ptr = curOpWs.weight;
            if(this->convolutionType == Convolution_Depthwise_Pointwise){
                GCLMem_t weightMemExt = modelWeightTensorExt->get_val();
                weightMemExt->desc.host_ptr = curOpWs.weight + weightMem->desc.byteSize;
            }
        }

        U8* biasVal    = nullptr;
        U8* biasValExt = nullptr;
        U8* biasTmp    = nullptr;
        U8* biasTmpExt = nullptr;
        if(modelPtr != nullptr) {
            if(this->hasBias){
                biasVal = *modelPtr;
                if(this->convolutionType == Convolution_Depthwise_Pointwise) biasValExt= *modelPtr + vectorMem->desc.byteSize;
                *modelPtr += tensorNumBytes(vectorTensorDesc);
            }
        } else {
            if(this->hasBias) {
                biasVal = curOpWs.vec;
                if(this->convolutionType == Convolution_Depthwise_Pointwise) biasValExt = curOpWs.vec + vectorMem->desc.byteSize;
            }
        }
        
        if (biasVal != nullptr) {
            vectorMem->desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
            if((vectorLenTmp & 3) == 0){
                vectorMem->desc.host_ptr = biasVal;
            } else {
                biasTmp = (U8*)operator new(vectorMem->desc.byteSize);
                memset(biasTmp, 0, vectorMem->desc.byteSize);
                memcpy(biasTmp, biasVal, tensorNumBytes(vectorTensorDescTmp));
                vectorMem->desc.host_ptr = biasTmp;
            }
            if(this->convolutionType == Convolution_Depthwise_Pointwise){
                GCLMem_t vectorMemExt = modelVectorTensorExt->get_val();
                vectorMemExt->desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
                if((vectorLenExt & 3) == 0){
                    vectorMemExt->desc.host_ptr = biasValExt;
                } else {
                    biasTmpExt = (U8*)operator new(vectorMemExt->desc.byteSize);
                    memset(biasTmpExt, 0, vectorMemExt->desc.byteSize);
                    memcpy(biasTmpExt, biasValExt, tensorNumBytes(vectorTensorDescExt));
                    vectorMem->desc.host_ptr = biasTmpExt;
                }
            }
        } else {
            vectorMem->desc.host_ptr = nullptr;
            vectorMem->desc.flags = CL_MEM_READ_WRITE;
            if(this->convolutionType == Convolution_Depthwise_Pointwise){
                GCLMem_t vectorMemExt = modelVectorTensorExt->get_val();
                vectorMemExt->desc.host_ptr = nullptr;
                vectorMemExt->desc.flags = CL_MEM_READ_WRITE;
            }
        }

        U8* weightTmp = nullptr;
        if(wn == 1 && ww == 1 && wh == 1 && wc == 3){
            weightMem->desc.stride[1] = wc + wn;
            weightMem->desc.num = ww * wh * (wc + wn) * wn;
            weightMem->desc.byteSize = weightMem->desc.num * bytesOf(dt);
            weightTmp = (U8*)operator new(weightMem->desc.byteSize + vectorMem->desc.byteSize);
            memset(weightTmp, 0, weightMem->desc.byteSize + vectorMem->desc.byteSize);
            memcpy(weightTmp, (U8*)weightMem->desc.host_ptr, weightMem->desc.byteSize);
            if(vectorMem->desc.host_ptr){
                memcpy(weightTmp + weightMem->desc.byteSize, (U8*)vectorMem->desc.host_ptr, vectorMem->desc.byteSize);
            }
            weightMem->desc.host_ptr = weightTmp;
        }
        modelWeightTensor->alloc();
        modelVectorTensor->alloc();
        this->weightTensors.push_back(*modelWeightTensor.get());
        this->biasTensors.push_back(*modelVectorTensor.get());
        if(this->convolutionType == Convolution_Depthwise_Pointwise){
            modelWeightTensorExt->alloc();
            modelVectorTensorExt->alloc();
            this->weightTensors.push_back(*modelWeightTensorExt.get());
            this->biasTensors.push_back(*modelVectorTensorExt.get());
        }
        if(weightTmp)      delete weightTmp;
        if(biasTmp)        delete biasTmp;
        if(biasTmpExt)     delete biasTmpExt;
        if(curOpWs.weight) delete curOpWs.weight;
        if(curOpWs.vec)    delete curOpWs.vec;

        return SUCCESS;
    }

    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        this->handle->curOpName = this->get_op_name();
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor filterTensor = this->weightTensors[0];
        TensorDesc filterDesc = filterTensor.get_desc();

        TensorDesc scaleDesc = filterDesc; // Dummy initialization
        U8 *scalePtr = nullptr;

        Tensor biasTensor = this->biasTensors[0];
        TensorDesc biasDesc = biasTensor.get_desc();

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution(inputDesc, inputTensor.get_val(),
                                         filterDesc, filterTensor.get_val(),
                                         convDesc,   this->pwAlg,
                                         scaleDesc, scalePtr,
                                         biasDesc, biasTensor.get_val(),
                                         this->lenOfTemp, this->temp->get_val(),
                                         outputDesc, (void*)outputTensor.get_val(),
                                         this->pwActivationDesc, this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution(inputDesc, inputTensor.get_val(),
                                                   filterDesc, filterTensor.get_val(),
                                                   convDesc,   this->dwAlg,
                                                   biasDesc, biasTensor.get_val(),
                                                   this->lenOfTemp, this->temp->get_val(),
                                                   outputDesc, (void*)outputTensor.get_val(),
                                                   this->dwActivationDesc, this->pwActivationDesc,
                                                   this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                GCLMem filterMem[2];
                filterMem[0] = *((GCLMem_t)filterTensor.get_val());
                filterMem[1] = *((GCLMem_t)this->weightTensors[1].get_val());
                GCLMem biasMem[2];
                biasMem[0] = *((GCLMem_t)biasTensor.get_val());
                biasMem[1] = *((GCLMem_t)this->biasTensors[1].get_val());
                if(this->dwAlg == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM) biasMem[1] = *((GCLMem_t)this->bias_buf->get_val());
                CHECK_STATUS(depthwise_convolution(inputDesc, inputTensor.get_val(),
                                                   filterDesc, (void*)filterMem,
                                                   convDesc,   this->dwAlg,
                                                   biasDesc, (void*)biasMem,
                                                   this->lenOfTemp, this->temp->get_val(),
                                                   outputDesc, (void*)outputTensor.get_val(),
                                                   this->dwActivationDesc, this->pwActivationDesc,
                                                   this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(NOT_SUPPORTED);
                break;
            }
            default:
                std::cerr << "[ERROR] unsupported convolution type " << this->convolutionType << std::endl;
                exit(1);
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_forward_algorithm(HashMap<std::string, std::string> &algorithmMap)override
    {
        TensorDesc inputDesc = (this->inputTensors[0]).get_desc();
        TensorDesc filterDesc = (this->weightTensors[0]).get_desc();

        ConvolutionPolicy policy = CONVOLUTION_TUNNING;
        DataType targetType = filterDesc.dt;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                if (this->dt == DT_F16_8Q) {
                    targetType = DT_I8;
                }
                if (algorithmMap.find(this->name) != algorithmMap.end()) {
                    I32 algo[4];
                    Operator::getAlgorithmInfoFromMap(algorithmMap, this->name, algo, 4);
                    this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
                    this->runInfo.best_w[0] = algo[1];
                    this->runInfo.best_c[0] = algo[2];
                    this->runInfo.best_k[0] = algo[3];
                    this->pwAlg = (ConvolutionForwardAlgorithm)algo[0];
                } else {
                    CHECK_STATUS(convolution_infer_forward_algorithm(inputDesc, filterDesc,
                                             this->outputTensors[0].get_desc(),
                                             convDesc, policy, &(this->pwAlg), targetType, this->pwActivationDesc, this->schedule, &this->oclExtInfo));
                    I32 algo[4];
                    algo[0] = this->runInfo.algorithm;
                    algo[1] = this->runInfo.best_w[0];
                    algo[2] = this->runInfo.best_c[0];
                    algo[3] = this->runInfo.best_k[0];
                    this->pwAlg = (ConvolutionForwardAlgorithm)algo[0];
                    Operator::setAlgorithmInfoToMap(algorithmMap, this->name, algo, 4);
                }
                break;
            }
            case Convolution_Depthwise: {
                if (algorithmMap.find(this->name) != algorithmMap.end()) {
                    I32 algo[4];
                    Operator::getAlgorithmInfoFromMap(algorithmMap, this->name, algo, 4);
                    this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
                    this->runInfo.best_w[0] = algo[1];
                    this->runInfo.best_c[0] = algo[2];
                    this->runInfo.best_k[0] = algo[3];
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo[0];
                } else {
                    CHECK_STATUS(depthwise_convolution_infer_forward_algorithm(inputDesc, filterDesc,
                                             this->outputTensors[0].get_desc(),
                                             convDesc, policy, &(this->dwAlg), targetType, this->dwActivationDesc, this->pwActivationDesc, this->schedule, &this->oclExtInfo));
                    I32 algo[4];
                    algo[0] = this->runInfo.algorithm;
                    algo[1] = this->runInfo.best_w[0];
                    algo[2] = this->runInfo.best_c[0];
                    algo[3] = this->runInfo.best_k[0];
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo[0];
                    Operator::setAlgorithmInfoToMap(algorithmMap, this->name, algo, 4);
                }
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                if (algorithmMap.find(this->name) != algorithmMap.end()) {
                    I32 algo[7];
                    Operator::getAlgorithmInfoFromMap(algorithmMap, this->name, algo, 7);
                    this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
                    this->runInfo.best_w[0] = algo[1];
                    this->runInfo.best_c[0] = algo[2];
                    this->runInfo.best_k[0] = algo[3];
                    this->runInfo.best_w[1] = algo[4];
                    this->runInfo.best_c[1] = algo[5];
                    this->runInfo.best_k[1] = algo[6];
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo[0];
                } else {
                    CHECK_STATUS(depthwise_convolution_infer_forward_algorithm(inputDesc, filterDesc,
                                             this->outputTensors[0].get_desc(),
                                             convDesc, policy, &(this->dwAlg), targetType, this->dwActivationDesc, this->pwActivationDesc, this->schedule, &this->oclExtInfo));
                    I32 algo[7];
                    algo[0] = this->runInfo.algorithm;
                    algo[1] = this->runInfo.best_w[0];
                    algo[2] = this->runInfo.best_c[0];
                    algo[3] = this->runInfo.best_k[0];
                    algo[4] = this->runInfo.best_w[1];
                    algo[5] = this->runInfo.best_c[1];
                    algo[6] = this->runInfo.best_k[1];
                    this->dwAlg = (DepthwiseConvolutionForwardAlgorithm)algo[0];
                    Operator::setAlgorithmInfoToMap(algorithmMap, this->name, algo, 7);
                }
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(NOT_SUPPORTED);
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

        convDesc = Convolution::create_convDesc(this->strideH, this->strideW, this->paddingT, this->paddingB,
            this->paddingL, this->paddingR, this->dilateH, this->dilateW);

        DataType targetType = DT_F16; // Default DT_F16
        if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->convolutionType) {
            targetType = DT_I8;
        }

        U32 outBytes = 0;
        this->oclExtInfo.maliInfo.gclmemInputDesc = NULL;
        this->oclExtInfo.maliInfo.gclmemOutputDesc = NULL;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes, this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes, 
                    this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_convolution_infer_output_size(inDim, filterDim, convDesc, &((*outDims)[0]), targetType, &outBytes, 
                    this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Dilation: {
                return NOT_SUPPORTED;
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return SUCCESS;
    }

    virtual EE infer_gclmem_desc(Vec<GCLMemDesc>* gclmemInputDesc, Vec<GCLMemDesc>* gclmemOutputDesc) override
    {
        TensorDesc inputDesc  = this->inputTensors[0].get_desc();
        TensorDesc filterDesc = this->weightTensors[0].get_desc();
        DataType targetType = DT_F16; // Default DT_F16
        if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->convolutionType) { targetType = DT_I8;}
        this->oclExtInfo.maliInfo.gclmemInputDesc  = &((*gclmemInputDesc)[0]);
        this->oclExtInfo.maliInfo.gclmemOutputDesc = &((*gclmemOutputDesc)[0]);
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_infer_output_size(inputDesc, filterDesc, convDesc, NULL, 
                    targetType, NULL, this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_infer_output_size(inputDesc, filterDesc, convDesc, NULL, 
                    targetType, NULL, this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_convolution_infer_output_size(inputDesc, filterDesc, convDesc, NULL, 
                    targetType, NULL, this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Dilation: {
                return NOT_SUPPORTED;
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return SUCCESS;
        
    }

    virtual U32 infer_tmp_memory_size()override
    {
        TensorDesc inputDesc = (this->inputTensors[0]).get_desc();
        TensorDesc filterDesc = (this->weightTensors[0]).get_desc();
        TensorDesc outputDesc = (this->outputTensors[0]).get_desc();

        U32 bytes = 0;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->pwAlg, &bytes, this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->dwAlg, &bytes, this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, this->dwAlg, &bytes, this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(NOT_SUPPORTED);
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return bytes;
    }

    virtual GCLMemDesc infer_wtm_memory_size_mali()override
    {
        TensorDesc filterDesc = (this->weightTensors[0]).get_desc();
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        GCLMemDesc tmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc gclmemWtmDesc[2];
        gclmemWtmDesc[0] = tmpDesc;
        gclmemWtmDesc[1] = tmpDesc;
        U32 bytes = 0;
        this->oclExtInfo.maliInfo.gclmemFilterDesc = gclmemWtmDesc;
        switch (this->convolutionType) {
            case Convolution_Pointwise: {
                CHECK_STATUS(convolution_transform_filter_bytes(filterDesc, this->pwAlg, &bytes, this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Depthwise: {
                CHECK_STATUS(depthwise_convolution_transform_filter_bytes(filterDesc, this->dwAlg, &bytes, this->schedule, &this->oclExtInfo));
                break;
            }
            case Convolution_Depthwise_Pointwise: {
                CHECK_STATUS(depthwise_convolution_transform_filter_bytes(filterDesc, this->dwAlg, &bytes, this->schedule, &this->oclExtInfo));
                wtm_dp = std::shared_ptr<Tensor>(new Tensor(this->handle));
                OclMemory* wtmMem = (OclMemory*)wtm_dp->get_memory();
                wtmMem->set_mem_desc(gclmemWtmDesc[1]);
                wtm_dp->alloc();
                if(this->dwAlg == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM) {
                    bias_buf = std::shared_ptr<Tensor>(new Tensor(this->handle));
                    DataType   dt      = this->biasTensors[1].get_desc().dt;
                    GCLMem_t   biasImg = (GCLMem_t)this->biasTensors[1].get_val();
                    GCLMem_t   biasBuf = bias_buf->get_val();
                    GCLMemDesc descImg = biasImg->desc;
                    U32 s1_img = descImg.stride[0];
                    U32 s1 = (s1_img * 4 + 7) / 8 * 8;
                    U32 s2 = 1;
                    U32 s3 = 1;
                    U32 num = s1 * s2 * s3;
                    U32 byteSize = num * bytesOf(dt);
                    biasBuf->desc.stride[0] = s1;
                    biasBuf->desc.stride[1] = s2;
                    biasBuf->desc.stride[2] = s3;
                    biasBuf->desc.offset[0] = 0;
                    biasBuf->desc.offset[1] = 0;
                    biasBuf->desc.offset[2] = 0;
                    biasBuf->desc.memType   = GCL_MEM_BUF;
                    biasBuf->desc.num       = num;
                    biasBuf->desc.byteSize  = byteSize;
                    biasBuf->desc.memFormat = DF_NHWC;
                    bias_buf->alloc();
                    U32 region[3] = {s1_img, 1, 1};
                    CHECK_STATUS(gcl_trans_memory(this->handle.get(), (void*)biasImg, (void*)biasBuf, region, DEVICE_IMG_TO_BUF, CL_TRUE));
                }
                break;
            }
            case Convolution_Dilation: {
                CHECK_STATUS(NOT_SUPPORTED);
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        return gclmemWtmDesc[0];
    }

    virtual EE transform_filter()override
    {
        Tensor filterTensor = this->weightTensors[0];
        TensorDesc filterDesc = filterTensor.get_desc();
        GCLMem_t weightPtr = filterTensor.get_val();

        TensorDesc wtmCpuDesc;
        if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->convolutionType && CONVOLUTION_ALGORITHM_WINOGRAD == this->pwAlg) { // int8 winograd
            return NOT_SUPPORTED;
        } else if (DT_F16_8Q == this->dt && Convolution_Pointwise == this->convolutionType) { // int8 tilegemm
            return NOT_SUPPORTED;
        } else { // All other cases
            auto wtmDesc = this->infer_wtm_memory_size_mali();
            this->wtm = std::shared_ptr<Tensor>(new Tensor(this->handle));
            OclMemory* wtmMem = (OclMemory*)this->wtm->get_memory();
            wtmMem->set_mem_desc(wtmDesc);
            this->wtm->alloc();

            switch (this->convolutionType) {
                case Convolution_Pointwise: {
                    CHECK_STATUS(convolution_transform_filter(filterDesc, weightPtr, this->pwAlg, &wtmCpuDesc, this->get_wtm()->get_val(), this->temp->get_val(), this->schedule, &this->oclExtInfo));
                    break;
                }
                case Convolution_Depthwise: {
                    CHECK_STATUS(depthwise_convolution_transform_filter(filterDesc, weightPtr, this->dwAlg, &wtmCpuDesc, this->get_wtm()->get_val(), 
                        this->schedule, &this->oclExtInfo));
                    break;
                }
                case Convolution_Depthwise_Pointwise: {
                    GCLMem weightPtrDp[2];
                    weightPtrDp[0] = *((GCLMem_t)this->weightTensors[0].get_val());
                    weightPtrDp[1] = *((GCLMem_t)this->weightTensors[1].get_val());
                    GCLMem weightPtrTranDp[2];
                    weightPtrTranDp[0] = *((GCLMem_t)(this->get_wtm()->get_val()));
                    weightPtrTranDp[1] = *((GCLMem_t)(wtm_dp->get_val()));
                    CHECK_STATUS(depthwise_convolution_transform_filter(filterDesc, weightPtrDp, this->dwAlg, &wtmCpuDesc, weightPtrTranDp, 
                        this->schedule, &this->oclExtInfo));
                    this->weightTensors[1] = *wtm_dp.get();
                    break;
                }
                case Convolution_Dilation: {
                    CHECK_STATUS(NOT_SUPPORTED);
                    break;
                }
                default:
                    CHECK_STATUS(NOT_SUPPORTED);
            }
            }

        this->get_wtm()->set_desc(wtmCpuDesc);
        this->weightTensors[0] = *this->get_wtm();
        return SUCCESS;
    }

    private:
    std::shared_ptr<Tensor> wtm_dp;
    std::shared_ptr<Tensor> bias_buf;
    ConvolutionDesc convDesc;
};

#endif //_CONVELTWISEPOOLING_H
