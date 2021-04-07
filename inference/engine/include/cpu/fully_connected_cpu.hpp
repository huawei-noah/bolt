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
        DataType dtNoQ = (DT_F16_8Q == this->dt) ? DT_F16 : this->dt;
        this->weightTensors = std::vector<Tensor>(1);
        this->weightTensors[0].resize(
            tensor2df(dtNoQ, DF_TRANSPOSE, this->p.num_outputs, this->numInput));
        this->biasTensors = std::vector<Tensor>(1);
        this->biasTensors[0].resize(tensor1d(dtNoQ, this->p.num_outputs));
        return SUCCESS;
    }

    TensorDesc desc_process(TensorDesc inDim, U32 *k = nullptr)
    {
        TensorDesc inputDesc;
        DataType dt;
        DataFormat df;
        U32 in, ic, ih, iw;
        U32 num = 0;
        switch (inDim.nDims) {
            case 2: {
                CHECK_STATUS(tensor2dGet(inDim, &dt, &df, &in, &num));
                inputDesc = inDim;
                break;
            }
            case 3: {
                CHECK_STATUS(tensor3dGet(inDim, &dt, &df, &in, &ih, &iw));
                num = iw;
                inputDesc = tensor2df(dt, DF_NORMAL, in * ih, iw);
                break;
            }
            case 4: {
                CHECK_STATUS(tensor4dGet(inDim, &dt, &df, &in, &ic, &ih, &iw));
                num = ic * ih * iw;
                inputDesc = inDim;
                break;
            }
            default:
                break;
        }
        if (k != nullptr) {
            *k = num;
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
                outDesc = tensor3df(dt, df, in, ih, this->p.num_outputs);
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

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        inputTensor.resize(desc_process(inputDesc));

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        outputDesc.dims[0] = this->p.num_outputs;
        outputTensor.resize(desc_process(outputDesc));

        if (featureScale.size() > 1 && featureScale[0][0] > 0 && DT_I8 != inputDesc.dt) {
            inputTensor.set_scale(featureScale[0][0]);
        }

        CHECK_STATUS(fully_connected(inputTensor, weightTensors[0], biasTensors[0], this->temp,
            outputTensor, &this->archInfo));
        inputTensor.resize(inputDesc);
        outputTensor.resize(outputDesc);
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->mvm = false;
        TensorDesc inputDesc = desc_process(inTensors[0]->get_desc(), &(this->numInput));
        TensorDesc weightDesc =
            tensor2df(inputDesc.dt, DF_TRANSPOSE, this->p.num_outputs, this->numInput);
        TensorDesc outputDesc;

        DataType idt;
        DataFormat idf;
        U32 in = 0, ic, ih, iw;
        if (tensorIs2d(inputDesc)) {
            CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &iw));
        } else if (tensorIs4d(inputDesc)) {
            CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        } else {
            CHECK_STATUS(NOT_MATCH);
        }
        if (1 == in) {
            this->mvm = true;
        }

        Tensor tmpInput;
        tmpInput.resize(inputDesc);
        Tensor tmpFilter;
        tmpFilter.resize(weightDesc);
        CHECK_STATUS(
            fully_connected_infer_output_size(&tmpInput, tmpFilter, outTensors[0], &this->archInfo));
        if (1 == this->p.num_slices) {
            outputDesc = outTensors[0]->get_desc();
            outputDesc = desc_process_reverse(inTensors[0]->get_desc(), outputDesc);
            if (DT_F16_8Q == this->dt) {
                if (featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
                    outputDesc.dt = DT_F16;
                } else {
                    outputDesc.dt = DT_I8;
                }
            }
            outTensors[0]->resize(outputDesc);
        } else {
            UNI_ERROR_LOG("FC merge is deprecated\n");
            outputDesc = desc_process_reverse(inTensors[0]->get_desc(), outputDesc);
            for (U32 i = 0; i < this->p.num_slices; i++) {
                outputDesc.dims[0] = this->p.slice_point[i];
                if (DT_F16_8Q == this->dt) {
                    if (featureScale.size() > 0 && -2 == (featureScale.back())[0]) {
                        outputDesc.dt = DT_F16;
                    } else {
                        outputDesc.dt = DT_I8;
                    }
                }
            }
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        TensorDesc inputDesc = desc_process((this->inputTensors[0]).get_desc());
        U32 bytes = 0;

        Tensor tmpInput, tmpFilter;
        tmpInput.resize(inputDesc);

        CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(
            tmpInput, weightTensors[0], &bytes, &this->archInfo));
        return bytes;
    }

    U32 infer_wtm_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(
            fully_connected_transform_filter_bytes(weightTensors[0], &bytes, &this->archInfo));
        return bytes;
    }

    EE transform_filter() override
    {
        return transform_filter(this->inputTensors[0].get_desc());
    }

    virtual EE transform_filter(const TensorDesc &originalInputDesc)
    {
        TensorDesc inputDesc = desc_process(originalInputDesc);
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
        if (DT_F16_8Q == this->dt) {
            std::shared_ptr<U8> qFilter = std::shared_ptr<U8>(
                (U8 *)operator new(bytesOf(DT_I8) * tensorNumElements(tmpDesc)));

            F16 scale = -1;
            F16 *inD = (F16 *)((CpuMemory *)(tmpFilter.get_memory()))->get_ptr();
            CHECK_STATUS(
                quantize_tensor(tmpFilter.get_desc(), inD, &tmpDesc, qFilter.get(), &scale));
            tmpFilter.resize(tmpDesc);
            ((CpuMemory *)(tmpFilter.get_memory()))->set_shared_ptr(qFilter);
            tmpFilter.set_scale(scale);
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
