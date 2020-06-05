// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


/**
 * Project deploy
 */
#ifndef _FCELTWISE_CPU_H
#define _FCELTWISE_CPU_H

#include "weight_operator.hpp"
#include "tensor_computing.h"
#include "fully_connected.hpp"
#include "blas-enhance.h"
#include <cmath>

class FullyConnectedCPU: public FullyConnected {
public:
    FullyConnectedCPU(DataType dt, U32 numInput, U32 numOutput,
        U32 numSlice, I32* slicePoint):
    FullyConnected(dt, numInput, numOutput, numSlice, slicePoint) { }

    virtual EE init_weight_bias_from_model(U8** modelPtr) override
    {
        DataType dtNoQ = (DT_F16_8Q == this->dt) ? DT_F16 : this->dt;
        TensorDesc weightDesc = tensor2df(dtNoQ, DF_NORMAL, this->numOutput, this->numInput);
        TensorDesc biasDesc = tensor1d(dtNoQ, this->numOutput);

        std::shared_ptr<Tensor> modelWeightTensor(new Tensor());
        std::shared_ptr<Tensor> modelBiasTensor(new Tensor());
        modelWeightTensor->set_desc(weightDesc);
        modelBiasTensor->set_desc(biasDesc);

        auto curOpWs = this->get_weightspec_ptr();
        if(modelPtr != nullptr){
            modelWeightTensor->alloc();
            memcpy((U8*)modelWeightTensor->get_val(), *modelPtr, tensorNumBytes(weightDesc));
            *modelPtr += tensorNumBytes(weightDesc);
        } else {
            modelWeightTensor->set_shared_ptr(std::shared_ptr<U8>(curOpWs.weight));
        }

        U8* biasVal = nullptr;
        if (modelPtr != nullptr) {
            if (this->hasBias) {
                biasVal = *modelPtr;
                *modelPtr += tensorNumBytes(biasDesc);
            }
        } else if (this->hasBias) {
            biasVal = curOpWs.vec;
        }

        if (biasVal) {
            modelBiasTensor->set_shared_ptr(std::shared_ptr<U8>(biasVal));
        } else {
            modelBiasTensor->alloc();
            memset((U8*)modelBiasTensor->get_val(), 0, tensorNumBytes(biasDesc));
        }

        this->weightTensors.push_back(*modelWeightTensor.get());
        this->biasTensors.push_back(*modelBiasTensor.get());
        return SUCCESS;
    }

    TensorDesc desc_process(TensorDesc inDim)
    {
        TensorDesc inputDesc;
        DataType dt;
        DataFormat df;
        U32 in, ic, ih, iw;
        switch (inDim.nDims) {
            case 2: {
                CHECK_STATUS(tensor2dGet(inDim, &dt, &in, &(this->numInput)));
                inputDesc = inDim;
                break;
            }
            case 3: {
                CHECK_STATUS(tensor3dGet(inDim, &dt, &df, &in, &ih, &iw));
                this->numInput = iw;
                inputDesc = tensor2df(dt, DF_NORMAL, in*ih, iw);
                break;
            }
            case 4: {
                CHECK_STATUS(tensor4dGet(inDim, &dt, &df, &in, &ic, &ih, &iw));
                this->numInput = ic*ih*iw;
                inputDesc = inDim;
                break;
            }
            default:
                break;
        }
        return inputDesc;
    }

    TensorDesc desc_process_reverse(TensorDesc inDim, TensorDesc outDim)
    {
        TensorDesc outDesc;
        DataType dt;
        DataFormat df;
        U32 in, ih, iw;
        switch (inDim.nDims) {
            case 2: {
                outDesc = outDim;
                break;
            }
            case 3: {
                CHECK_STATUS(tensor3dGet(inDim, &dt, &df, &in, &ih, &iw));
                outDesc = tensor3df(dt, df, in, ih, this->numOutput);
                break;
            }
            case 4: {
                outDesc = outDim;
                break;
            }
            default:
                break;
        }
        return outDesc;
    }


    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc = desc_process(inputTensor.get_desc());

        Tensor weightTensor = this->weightTensors[0];
        TensorDesc weightDesc = weightTensor.get_desc();

        Tensor biasTensor = this->biasTensors[0];
        TensorDesc biasDesc = biasTensor.get_desc();
        U8 *bias = biasTensor.get_val();

        U8 *tmp = (U8*)this->temp->get_val();

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        outputDesc.dims[0] = this->numOutput;
        U32 numRow = outputDesc.dims[1];
        outputDesc = desc_process(outputDesc);
        U8 *fcOutput;
        if (1 == numSlice) {
            fcOutput = outputTensor.get_val();
        } else {
            fcOutput = tmp;
            if (DT_I8 == weightDesc.dt) {
                tmp += tensorNumElements(outputDesc) * bytesOf(DT_I32);
            } else {
                tmp += tensorNumBytes(outputDesc);
            }
        }

        if (DT_I8 == weightDesc.dt) {
#ifdef _USE_INT8
            U8 *inputPtr = inputTensor.get_val();
            F32 scaleI = 1;
            if (DT_F16 == inputDesc.dt) {
                F16 *inD = (F16*)inputPtr;
                INT8 *inQ = (INT8*)tmp;
                F16 scale = -1;
                if (featureScale.size() > 1 && featureScale[0][0] > 0) {
                    scale = featureScale[0][0];
                }
                quantize_tensor(inputDesc, inD, &inputDesc, inQ, &scale);
                scaleI = scale;
                inputPtr = (U8*)tmp;
                tmp += tensorNumBytes(inputDesc);
            } else {
                scaleI = inputTensor.get_scale();
            }
            // The first portion of tmp is used for quantized bias and results before quantization
            if (this->hasBias && DT_F16 != outputDesc.dt) {
                biasDesc.dt = DT_I32;
                bias = (U8*)biasScaled.data();
            } else {
                bias = nullptr;
            }
            outputDesc.dt = DT_I32;

            I32 *result = (I32*)tmp;
            U8 *tmpReal = tmp + tensorNumBytes(outputDesc);

            if (nullptr == bias) {
                memset(result, 0, tensorNumBytes(outputDesc));
            } else {
                F16 *biasF = (F16*)biasTensor.get_val();
                I32 *biasI = biasScaled.data();
                for (U32 i = 0; i < numSlice; i++) {
                    F32 scale = scaleI * weightScale[i];
                    for (int j = 0; j < slicePoints[i]; j++) {
                        biasI[j] = round(scale * biasF[j]);
                    }
                    biasI += slicePoints[i];
                    biasF += slicePoints[i];
                }
            }
            CHECK_STATUS(fully_connected(inputDesc, inputPtr,
                                     weightDesc, weightTensor.get_val(),
                                     tmpReal, this->lenOfTemp,
                                     outputDesc, result,
                                     biasDesc, bias, this->schedule));

            if (1 == this->numSlice) {
                F32 scale = scaleI * weightScale[0];
                if (DT_I8 == outputTensor.get_desc().dt) {
                    CHECK_STATUS(quantize_tensor(outputDesc, result, &outputDesc, fcOutput, &scale));
                    this->outputTensors[0].set_scale(scale);
                } else {
                    CHECK_REQUIREMENT(DT_F16 == outputTensor.get_desc().dt);
                    F16 *output = outputTensor.get_val();
                    dequantize_int32_to_fp16(tensorNumElements(outputDesc), result, scale, output, tensorNumElements(biasDesc), (F16*)this->biasTensors[0].get_val());
                }
            } else {
                CHECK_REQUIREMENT(this->numSlice == this->outputTensors.size());
                Vec<U8*> bufD(this->numSlice);
                bufD[0] = fcOutput;
                for (U32 i = 1; i < this->numSlice; i++) {
                    bufD[i] = bufD[i - 1] + tensorNumElements(this->outputTensors[i - 1].get_desc()) * bytesOf(DT_I32);
                }
                CHECK_REQUIREMENT(numRow * this->numOutput == tensorNumElements(outputDesc));
                for (U32 i = 0; i < numRow; i++) {
                    for (U32 j = 0; j < this->numSlice; j++) {
                        U32 sliceSize = this->slicePoints[j] * bytesOf(DT_I32);
                        memcpy(bufD[j], result, sliceSize);
                        bufD[j] += sliceSize;
                        result += this->slicePoints[j];
                    }
                }
                F16 *biasPtr = (F16*)this->biasTensors[0].get_val();
                for (U32 i = 0; i < this->numSlice; i++) {
                    F32 scale = scaleI * weightScale[i];
                    outputDesc.dims[0] = slicePoints[i];
                    if (DT_I8 == outputTensor.get_desc().dt) {
                        CHECK_STATUS(quantize_tensor(outputDesc, fcOutput, &outputDesc, this->outputTensors[i].get_val(), &scale));
                        this->outputTensors[i].set_scale(scale);
                    } else {
                        CHECK_REQUIREMENT(DT_F16 == outputTensor.get_desc().dt);
                        F16 *output = outputTensors[i].get_val();
                        dequantize_int32_to_fp16(tensorNumElements(outputDesc), (I32*)fcOutput, scale, output, slicePoints[i], biasPtr);
                        biasPtr += slicePoints[i];
                    }
                    
                    outputDesc.dt = DT_I32;
                    fcOutput += tensorNumBytes(outputDesc);
                }
            }
#endif
        } else {
            if (nullptr == bias) {
                memset(fcOutput, 0, tensorNumBytes(outputDesc));
            }
            CHECK_STATUS(fully_connected(inputDesc, inputTensor.get_val(),
                                     weightDesc, weightTensor.get_val(),
                                     tmp, this->lenOfTemp,
                                     outputDesc, fcOutput,
                                     biasDesc, bias, this->schedule));

            if (1 != this->numSlice) {
                CHECK_REQUIREMENT(this->numSlice == this->outputTensors.size());
                Vec<U8*> outputPtr(this->numSlice);
                for (U32 i = 0; i < this->numSlice; i++) {
                    outputPtr[i] = this->outputTensors[i].get_val();
                }
                CHECK_REQUIREMENT(numRow * this->numOutput == tensorNumElements(outputDesc));
                for (U32 i = 0; i < numRow; i++) {
                    for (U32 j = 0; j < this->numSlice; j++) {
                        U32 sliceSize = this->slicePoints[j] * bytesOf(outputDesc.dt);
                        memcpy(outputPtr[j], fcOutput, sliceSize);
                        outputPtr[j] += sliceSize;
                        fcOutput += sliceSize;
                    }
                }
            }
        }
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        this->mvm = false;
        TensorDesc inputDesc = desc_process(inDims[0]);
        TensorDesc weightDesc = tensor2df(inputDesc.dt, DF_NORMAL, this->numOutput, this->numInput);
        TensorDesc outputDesc;
        
        DataType idt;
        DataFormat idf;
        U32 in = 0, ic, ih, iw;
        if (tensorIs2d(inputDesc)) {
            CHECK_STATUS(tensor2dfGet(inputDesc, &idt, &idf, &in, &iw));
        } else if (tensorIs4d(inputDesc)) {
            CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        } else {
            CHECK_STATUS(NOT_MATCH);
        }
        if (1 == in) {
            this->mvm = true;
        }

        CHECK_STATUS(fully_connected_infer_output_size(inputDesc, weightDesc, &outputDesc, this->schedule));
        if (1 == this->numSlice) {
            (*outDims)[0] = desc_process_reverse(inDims[0], outputDesc);
            if (DT_F16_8Q == this->dt) {
                if (featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
                    (*outDims)[0].dt = DT_F16;
                } else {
                    (*outDims)[0].dt = DT_I8;
                }
            }
        } else {
            outputDesc = desc_process_reverse(inDims[0], outputDesc);
            for (U32 i = 0; i < this->numSlice; i++) {
                (*outDims)[i] = outputDesc;
                (*outDims)[i].dims[0] = this->slicePoints[i];
                if (DT_F16_8Q == this->dt) {
                    if (featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
                        (*outDims)[i].dt = DT_F16;
                    } else {
                        (*outDims)[i].dt = DT_I8;
                    }
                }
            }
        }
        return SUCCESS;
    }

    virtual U32 infer_tmp_memory_size() override
    {
        TensorDesc inputDesc = desc_process((this->inputTensors[0]).get_desc());
        TensorDesc castDesc = inputDesc;
        TensorDesc filterDesc = (this->weightTensors[0]).get_desc();
        TensorDesc outputDesc = this->outputTensors[0].get_desc();
        outputDesc.dims[0] = this->numOutput;
        U32 bytes = 0;

        castDesc.dt = filterDesc.dt;
        CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(castDesc, filterDesc, &bytes, this->schedule));
        if (DT_I8 == filterDesc.dt) {
            if (DT_F16 == inputDesc.dt) {
                bytes += tensorNumElements(castDesc);
            }
            outputDesc.dt = DT_I32;
            bytes += tensorNumBytes(outputDesc);  // Results before quantization
        }
        if (1 != this->numSlice) {
            bytes += tensorNumBytes(outputDesc);
        }
        return bytes;
    }

    virtual U32 infer_wtm_memory_size() override
    {
        TensorDesc weightDesc = (this->weightTensors[0]).get_desc();
        U32 bytes = 0;
        CHECK_STATUS(fully_connected_transform_filter_bytes(weightDesc, &bytes, this->schedule));
        return bytes;
    }

    virtual EE transform_filter() override
    {
        this->wtm = std::shared_ptr<Tensor>(new Tensor());
        TensorDesc inputDesc = desc_process((this->inputTensors[0]).get_desc());

        Tensor weightTensor = this->weightTensors[0];
        TensorDesc weightDesc = weightTensor.get_desc();
        U8* weightPtr = weightTensor.get_val();

        TensorDesc wtmDesc;
        auto wtm_bytes = this->infer_wtm_memory_size();

        if (DT_F16_8Q == this->dt) {
#ifdef _USE_INT8
            TensorDesc tFilterDesc;
            F16 *tFilter = (F16*)malloc(wtm_bytes + bytesOf(DT_I8) * tensorNumElements(weightDesc));
            if (nullptr == tFilter) {
                std::cerr << "[ERROR] allocation failed for filter transform in int8 FC" << std::endl;
                CHECK_STATUS(ALLOC_FAILED);
            }
            TensorDesc qFilterDesc;
            INT8 *qFilter = (INT8*)(tFilter + wtm_bytes / bytesOf(DT_F16));

            inputDesc.dt = DT_F16;
            CHECK_STATUS(fully_connected_transform_filter(inputDesc, weightDesc, weightPtr, &tFilterDesc, tFilter, this->schedule));
            U32 ftm_bytes = wtm_bytes / bytesOf(DT_F16);
            std::shared_ptr<U8> wtmPtr((U8*) operator new(ftm_bytes));
            auto cpuMem = new CpuMemory();
            cpuMem->set_shared_ptr_caster(wtmPtr);
            Memory_* mem = (Memory_*)(cpuMem);
            std::shared_ptr<Memory_> memWtmPtr(mem);
            this->set_wtm_memory(wtm_bytes, memWtmPtr);

            F16 scale;
            this->weightScale = Vec<F32>(numSlice);
            if (this->mvm) {
                F16 *inD = tFilter;
                INT8 *inQ = this->get_wtm()->get_val();
                for (U32 i = 0; i < numSlice; i++) {
                    tFilterDesc.dims[1] = slicePoints[i];
                    scale = -1;
                    CHECK_STATUS(quantize_tensor(tFilterDesc, inD, &qFilterDesc, inQ, &scale));
                    weightScale[i] = scale;
                    inD += tensorNumElements(tFilterDesc);
                    inQ += tensorNumElements(qFilterDesc);
                }
                wtmDesc = qFilterDesc;
                wtmDesc.dims[1] = numOutput;
            } else if (featureScale.size() > 0 && featureScale[0][0] > 0) {
                F16 *inD = tFilter;
                INT8 *inQ = qFilter;
                scale = -1;
                CHECK_STATUS(quantize_tensor(tFilterDesc, inD, &qFilterDesc, inQ, &scale));

                for (U32 i = 0; i < numSlice; i++) {
                    weightScale[i] = scale;
                }
                CHECK_STATUS(matrix_matrix_multiply_transform_rhs(qFilterDesc, qFilter, &wtmDesc, this->get_wtm()->get_val()));
            } else {
                F16 *inD = tFilter;
                INT8 *inQ = qFilter;
                for (U32 i = 0; i < numSlice; i++) {
                    tFilterDesc.dims[0] = slicePoints[i];
                    scale = -1;
                    CHECK_STATUS(quantize_tensor(tFilterDesc, inD, &qFilterDesc, inQ, &scale));
                    weightScale[i] = scale;
                    inD += tensorNumElements(tFilterDesc);
                    inQ += tensorNumElements(qFilterDesc);
                }
                qFilterDesc.dims[0] = numOutput;
                CHECK_STATUS(matrix_matrix_multiply_transform_rhs(qFilterDesc, qFilter, &wtmDesc, this->get_wtm()->get_val()));
            }
            this->get_wtm()->set_scale(scale);
            biasScaled.resize(this->numOutput);

            free(tFilter);
#endif
        } else {
            std::shared_ptr<U8> wtmPtr((U8*) operator new(wtm_bytes));
            auto cpuMem = new CpuMemory();
            cpuMem->set_shared_ptr_caster(wtmPtr);
            Memory_* mem = (Memory_*)(cpuMem);
            std::shared_ptr<Memory_> memWtmPtr(mem);
            this->set_wtm_memory(wtm_bytes, memWtmPtr);

            if (this->mvm) {
                CHECK_STATUS(fully_connected_transform_filter(inputDesc, weightDesc, weightPtr, &wtmDesc, this->get_wtm()->get_val(), this->schedule));
            } else {
                TensorDesc tFilterDesc;
                U8 *tFilter = (U8*)malloc(wtm_bytes);
                if (nullptr == tFilter) {
                    std::cerr << "[ERROR] allocation failed for filter transform in FC" << std::endl;
                    CHECK_STATUS(ALLOC_FAILED);
                }
                CHECK_STATUS(fully_connected_transform_filter(inputDesc, weightDesc, weightPtr, &tFilterDesc, tFilter, this->schedule));
                CHECK_STATUS(matrix_matrix_multiply_transform_rhs(tFilterDesc, tFilter, &wtmDesc, this->get_wtm()->get_val()));
                free(tFilter);
            }
        }

        this->get_wtm()->set_desc(wtmDesc);
        this->weightTensors[0] = *this->get_wtm();
        return SUCCESS;
    }

    bool mvm;
    Vec<F32> weightScale;
#ifdef _USE_INT8
    Vec<I32> biasScaled;
#endif
};

#endif  //_FCELTWISE_CPU_H
