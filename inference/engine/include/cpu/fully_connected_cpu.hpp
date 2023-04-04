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

    EE infer_weight_desc() override
    {
        DataType weightDt = this->ws.mdt;
        if (this->ws.bytes_of_weight > 0) {
            this->weightTensors = std::vector<Tensor>(1);
            this->weightTensors[0].resize(
                tensor2df(weightDt, DF_TRANSPOSE, this->p.num_outputs, this->numInput));
        }
        if (this->ws.bytes_of_vec > 0) {
            this->biasTensors = std::vector<Tensor>(1);
            this->biasTensors[0].resize(tensor1d(noQuantDataType(this->dt), this->p.num_outputs));
        }
        return SUCCESS;
    }

    Tensor get_weight_tensor()
    {
        if (weightTensors.size() > 0) {
            return this->weightTensors[0];
        } else {
            CHECK_REQUIREMENT(1 < this->inputTensors.size());
            TensorDesc desc = this->inputTensors[1].get_desc();
            if (this->mvm) {
                desc.df = DF_TRANSPOSE;
            } else {
                desc.df = DF_NORMAL;
            }
            Tensor weightTensor = this->inputTensors[1];
            weightTensor.resize(desc);
            return weightTensor;
        }
    }

    Tensor get_bias_tensor()
    {
        if (biasTensors.size() > 0) {
            return this->biasTensors[0];
        } else {
            U32 inputCount = 1;
            if (weightTensors.size() == 0) {
                inputCount++;
            }
            if (inputCount < this->inputTensors.size()) {
                return this->inputTensors[inputCount++];
            }
            Tensor biasTensor;
            return biasTensor;
        }
    }

    void run() override
    {
        Tensor weightTensor = get_weight_tensor();
        Tensor biasTensor = get_bias_tensor();
        Tensor outputTensor = this->outputTensors[0];
#ifdef _USE_INT8
        TensorDesc inputDesc = this->inputTensors[0].get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        if (featureScale.size() > 1 && featureScale[0][0] > 0 && DT_I8 != inputDesc.dt &&
            DT_U8_Q != inputDesc.dt) {
            this->inputTensors[0].set_scale(featureScale[0][0]);
        }
        if (DT_I8 == outputDesc.dt || DT_U8_Q == outputDesc.dt) {
            if (featureScale.size() > 0) {
                outputTensor.set_scale((featureScale.back())[0]);
            } else {
                outputTensor.set_scale(-1);
            }
        }
#endif
        std::vector<Tensor> tmpTensor(1, this->temp);
        CHECK_STATUS(fully_connected(this->inputTensors[0], weightTensor, biasTensor, tmpTensor,
            outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        TensorDesc inputDesc = inTensors[0]->get_desc();
        if (this->ws.bytes_of_weight > 0) {
            this->numInput =
                this->ws.bytes_of_weight / this->p.num_outputs / UNI_MAX(1, bytesOf(this->ws.mdt));
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
        if (1 == this->p.num_slices) {
            TensorDesc outputDesc = outTensors[0]->get_desc();
            if (isQuantMixDataType(this->dt)) {
                if (featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
                    outputDesc.dt = noQuantDataType(this->dt);
                } else {
                    outputDesc.dt = get_activation_quant_data_type((featureScale.back())[0]);
                }
            }
            outTensors[0]->resize(outputDesc);
        } else {
            //UNI_ERROR_LOG("FC merge is deprecated\n");
            for (U32 i = 0; i < this->p.num_slices; i++) {
                TensorDesc outputDesc = outTensors[i]->get_desc();
                outputDesc.dims[0] = this->p.slice_point[i];
                if (isQuantMixDataType(this->dt)) {
                    if (featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
                        outputDesc.dt = noQuantDataType(this->dt);
                    } else {
                        outputDesc.dt = get_activation_quant_data_type((featureScale.back())[0]);
                    }
                }
                outTensors[i]->resize(outputDesc);
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

    EE transform_filter() override
    {
        EE ret = SUCCESS;
        if (weightTensors.size() > 0) {
            ret = transform_filter(this->inputTensors[0].get_desc());
        }
        return ret;
    }

    bool use_nchwc8(const TensorDesc &inputDesc)
    {
        bool ret = false;
        int hw = 1;
        for (int i = 0; i < (int)inputDesc.nDims - 2; i++) {
            hw *= inputDesc.dims[i];
        }
        if (inputDesc.df == DF_NCHWC8 && hw > 1) {
            ret = true;
        }
        return ret;
    }

    virtual EE transform_filter(const TensorDesc &inputDesc)
    {
        Tensor tTensor;
        Tensor wTensor = this->weightTensors[0];
        if (use_nchwc8(inputDesc)) {
            Tensor input;
            input.resize(inputDesc);
            U32 wtmBytes = 0;
            CHECK_STATUS(fully_connected_transform_filter_bytes(wTensor, &wtmBytes, &this->archInfo));
            tTensor = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, wtmBytes));
            CHECK_STATUS(
                fully_connected_transform_filter(input, wTensor, &tTensor, &this->archInfo));
        } else {
            tTensor = wTensor;
            TensorDesc desc = wTensor.get_desc();
            if (this->mvm) {
                desc.df = DF_NORMAL;
            }
            tTensor.resize(desc);
        }

#ifdef _USE_INT8
        TensorDesc desc = tTensor.get_desc();
        bool thisIsNoQuant = (featureScale.size() > 1 && featureScale[0].back() == 0);
        if (isQuantMixDataType(this->dt) && !thisIsNoQuant && (desc.dt != DT_I8)) {
            desc.dt = DT_I8;
            Tensor qTensor = Tensor::alloc_sized<CPUMem>(desc);
            F32 scale = -1;
            CHECK_STATUS(quantize(tTensor, &qTensor, &scale, &(this->archInfo)));
            qTensor.set_scale(scale);
            tTensor = qTensor;
        }
#endif

        TensorDesc fDesc = tTensor.get_desc();
        auto f = ((CpuMemory *)(tTensor.get_memory()))->get_ptr();
        Tensor wtm;
        TensorDesc tDesc;
        U32 tBytes = 0;
        if (this->mvm) {
            CHECK_STATUS(
                matrix_vector_multiply_transform_weight_bytes(fDesc, &tBytes, this->archInfo.arch));
            wtm = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, tBytes));
            CHECK_STATUS(matrix_vector_multiply_transform_weight(fDesc, f, &tDesc,
                ((CpuMemory *)(wtm.get_memory()))->get_ptr(), this->archInfo.arch));
        } else {
            CHECK_STATUS(matrix_matrix_multiply_transform_rhs_bytes(
                fDesc, &tBytes, nullptr, this->archInfo.arch));
            wtm = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, tBytes));
            CHECK_STATUS(matrix_matrix_multiply_transform_rhs(fDesc, f, &tDesc,
                ((CpuMemory *)(wtm.get_memory()))->get_ptr(), this->archInfo.arch));
        }
        wtm.resize(tDesc);
        wtm.set_scale(tTensor.get_scale());
        this->weightTensors[0] = wtm;
        return SUCCESS;
    }

    bool mvm;
};

#endif  // _FULLY_CONNECTED_CPU_H
