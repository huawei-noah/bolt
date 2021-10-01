// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _FULLY_CONNECTED_CPU_H
#define _FULLY_CONNECTED_CPU_H

#include "fully_connected.hpp"
#include "blas_enhance.h"

class FullyConnectedCPU : public FullyConnected {
public:
    FullyConnectedCPU(DataType dt, FullyConnectedParamSpec p, U32 numInput)
        : FullyConnected(dt, p, numInput)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<FullyConnectedCPU> mem = std::shared_ptr<FullyConnectedCPU>(
            new FullyConnectedCPU(this->dt, this->p, this->numInput));
        *mem = *this;
        return mem;
    }

    DataType get_float_precision()
    {
        DataType ret = this->dt;
        if (this->dt == DT_F16_8Q) {
            ret = DT_F16;
        } else if (this->dt == DT_F32_8Q) {
            ret = DT_F32;
        }
        return ret;
    }

    EE infer_weight_desc() override
    {
        DataType dtNoQ = this->get_float_precision();
        auto curOpWs = this->get_weightspec();
        if (curOpWs.bytes_of_weight > 0) {
            this->weightTensors = std::vector<Tensor>(1);
            this->weightTensors[0].resize(
                tensor2df(dtNoQ, DF_TRANSPOSE, this->p.num_outputs, this->numInput));
        }
        if (curOpWs.bytes_of_vec > 0) {
            this->biasTensors = std::vector<Tensor>(1);
            this->biasTensors[0].resize(tensor1d(dtNoQ, this->p.num_outputs));
        }
        return SUCCESS;
    }

    Tensor get_weight_tensor()
    {
        Tensor weightTensor;
        if (weightTensors.size() > 0) {
            weightTensor = this->weightTensors[0];
        } else {
            CHECK_REQUIREMENT(1 < this->inputTensors.size());
            weightTensor = this->inputTensors[1];
            TensorDesc desc = weightTensor.get_desc();
            if (this->mvm) {
                desc.df = DF_TRANSPOSE;
            } else {
                desc.df = DF_NORMAL;
            }
            weightTensor.resize(desc);
        }
        return weightTensor;
    }

    Tensor get_bias_tensor()
    {
        Tensor biasTensor;
        U32 inputCount = 1;
        if (weightTensors.size() == 0) {
            inputCount++;
        }
        if (biasTensors.size() > 0) {
            biasTensor = this->biasTensors[0];
        } else {
            if (inputCount < this->inputTensors.size()) {
                biasTensor = this->inputTensors[inputCount++];
            }
        }
        return biasTensor;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor weightTensor = get_weight_tensor();
        Tensor biasTensor = get_bias_tensor();
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        if (featureScale.size() > 1 && featureScale[0][0] > 0 && DT_I8 != inputDesc.dt &&
            DT_U8_Q != inputDesc.dt) {
            inputTensor.set_scale(featureScale[0][0]);
        }
        if (DT_I8 == outputDesc.dt || DT_U8_Q == outputDesc.dt) {
            if (featureScale.size() > 0) {
                outputTensor.set_scale((featureScale.back())[0]);
            } else {
                outputTensor.set_scale(-1);
            }
        }

        std::vector<Tensor> tmpTensor(1, this->temp);
        CHECK_STATUS(fully_connected(
            inputTensor, weightTensor, biasTensor, tmpTensor, outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        TensorDesc inputDesc = inTensors[0]->get_desc();
        auto curOpWs = this->get_weightspec();
        if (curOpWs.bytes_of_weight > 0) {
            this->numInput =
                curOpWs.bytes_of_weight / this->p.num_outputs / UNI_MAX(1, bytesOf(curOpWs.mdt));
        } else {
            this->numInput = inputDesc.dims[0];
        }
        U32 inputNum = tensorNumElements(inputDesc);
        U32 M = inputNum / this->numInput;
        TensorDesc weightDesc =
            tensor2df(inputDesc.dt, DF_TRANSPOSE, this->p.num_outputs, this->numInput);
        if (1 == M) {
            this->mvm = true;
        } else {
            this->mvm = false;
        }

        Tensor tmpFilter;
        tmpFilter.resize(weightDesc);
        CHECK_STATUS(fully_connected_infer_output_size(
            inTensors[0], tmpFilter, outTensors[0], &this->archInfo));
        TensorDesc outputDesc = outTensors[0]->get_desc();
        if (1 == this->p.num_slices) {
            if (DT_F16_8Q == this->dt || DT_F32_8Q == this->dt) {
                if (featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
                    outputDesc.dt = (DT_F16_8Q == this->dt) ? DT_F16 : DT_F32;
                } else {
#ifdef _USE_X86
                    outputDesc.dt = DT_U8_Q;
#else
                    outputDesc.dt = DT_I8;
#endif
                }
            }
            outTensors[0]->resize(outputDesc);
        } else {
            UNI_ERROR_LOG("FC merge is deprecated\n");
            for (U32 i = 0; i < this->p.num_slices; i++) {
                outputDesc.dims[0] = this->p.slice_point[i];
                if (DT_F16_8Q == this->dt || DT_F32_8Q == this->dt) {
                    if (featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
                        outputDesc.dt = (DT_F16_8Q == this->dt) ? DT_F16 : DT_F32;
                    } else {
#ifdef _USE_X86
                        outputDesc.dt = DT_U8_Q;
#else
                        outputDesc.dt = DT_I8;
#endif
                    }
                }
            }
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        Tensor tmpFilter = get_weight_tensor();
        CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(
            this->inputTensors[0], tmpFilter, this->outputTensors[0], &bytes, &this->archInfo));
        return bytes;
    }

    U32 infer_wtm_memory_size() override
    {
        U32 bytes = 0;
        if (weightTensors.size() > 0) {
            CHECK_STATUS(
                fully_connected_transform_filter_bytes(weightTensors[0], &bytes, &this->archInfo));
        }
        return bytes;
    }

    EE transform_filter() override
    {
        EE ret = SUCCESS;
        if (weightTensors.size() > 0) {
            ret = transform_filter(this->inputTensors[0].get_desc());
        }
        return ret;
    }

    virtual EE transform_filter(const TensorDesc &originalInputDesc)
    {
        TensorDesc inputDesc = originalInputDesc;
        Tensor weightTensor = this->weightTensors[0];
        TensorDesc weightDesc = weightTensor.get_desc();
        TensorDesc wtmDesc;
        auto wtm_bytes = this->infer_wtm_memory_size();

        TensorDesc tmpDesc;
        Tensor tmpFilter;

        Tensor tmpInput;
        tmpInput.resize(inputDesc);
        int hw = 1;
        for (int i = 0; i < (int)inputDesc.nDims - 2; i++) {
            hw *= inputDesc.dims[i];
        }
        if (inputDesc.df == DF_NCHWC8 && hw > 1) {
            tmpFilter.resize(tensor1d(DT_U8, wtm_bytes));
            tmpFilter.alloc();
            CHECK_STATUS(fully_connected_transform_filter(
                tmpInput, weightTensors[0], &tmpFilter, &this->archInfo));
        } else {
            tmpDesc = weightDesc;
            if (this->mvm) {
                tmpDesc.df = DF_NORMAL;
            }
            tmpFilter = weightTensor;
            tmpFilter.resize(tmpDesc);
        }

#ifdef _USE_INT8
        bool thisIsNoQuant = (featureScale.size() > 1 && featureScale[0].back() == 0);
        if ((DT_F16_8Q == this->dt || DT_F32_8Q == this->dt) && !thisIsNoQuant) {
            tmpDesc.dt = DT_I8;
            Tensor qFilter = Tensor::alloc_sized<CPUMem>(tmpDesc);
            F32 scale = -1;
            CHECK_STATUS(quantize(tmpFilter, &qFilter, &scale, &(this->archInfo)));
            qFilter.set_scale(scale);
            tmpFilter = qFilter;
        }
#endif
        this->wtm = std::shared_ptr<Tensor>(new Tensor());
        wtm->resize(tensor1d(DT_U8, wtm_bytes));
        wtm->alloc();
        wtm->set_scale(tmpFilter.get_scale());
        if (this->mvm) {
            CHECK_STATUS(matrix_vector_multiply_transform_weight(tmpFilter.get_desc(),
                ((CpuMemory *)(tmpFilter.get_memory()))->get_ptr(), &wtmDesc,
                ((CpuMemory *)(wtm->get_memory()))->get_ptr(), this->archInfo.arch));
            wtm->resize(wtmDesc);
        } else {
            CHECK_STATUS(matrix_matrix_multiply_transform_rhs(tmpFilter.get_desc(),
                ((CpuMemory *)(tmpFilter.get_memory()))->get_ptr(), &wtmDesc,
                ((CpuMemory *)(wtm->get_memory()))->get_ptr(), this->archInfo.arch));
            wtm->resize(wtmDesc);
        }
        this->weightTensors[0] = *this->get_wtm();
        return SUCCESS;
    }

    bool mvm;
};

#endif  // _FULLY_CONNECTED_CPU_H
