// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _RNNCELL_CPU_H
#define _RNNCELL_CPU_H

#include "rnncell.hpp"

class RNNCellCPU : public RNNCell {
public:
    RNNCellCPU(DataType dt, RNNParamSpec p) : RNNCell(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<RNNCellCPU> mem =
            std::shared_ptr<RNNCellCPU>(new RNNCellCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor xTensor = this->inputTensors[0];
        Tensor stateTensor = this->inputTensors[1];
        Tensor hTensor = this->outputTensors[0];
        Tensor tmpTensor = this->temp;
        U32 tmpOffset = 0;
        if (this->featureScale.size() > 1) {
            tmpTensor.resize(xTensor.get_desc());
            CHECK_STATUS(clip(xTensor, this->clipParam, tmpTensor, &this->archInfo));
            xTensor = tmpTensor;
            tmpOffset = xTensor.bytes();
        }
        CHECK_STATUS(rnncell(xTensor, this->weightTensors, this->biasTensors, stateTensor, this->p,
            this->xDim, this->p.numOutput, tmpOffset, tmpTensor, hTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        TensorDesc inDim = inTensors[0]->get_desc();
        DataType dt;
        DataFormat df;
        U32 iB, iX;
        CHECK_STATUS(tensor2dGet(inDim, &dt, &df, &iB, &iX));
        this->xDim = iX;
        CHECK_STATUS(rnncell_infer_output_size(inTensors, this->p, outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(rnncell_infer_forward_tmp_bytes(this->inputTensors[0], this->weightTensors[0],
            this->outputTensors[0], this->p, &bytes, &this->archInfo));

        if (featureScale.size() > 1) {
            CHECK_REQUIREMENT(featureScale[0][0] > 0);
            CHECK_REQUIREMENT(featureScale[0][0] == featureScale[1][0]);
            this->clipParam.max = 127.0 / featureScale[0][0];
            this->clipParam.min = -1 * this->clipParam.max;
            bytes += this->inputTensors[0].bytes();
        }
        return bytes;
    }

    EE transform_filter() override
    {
        I32 filter_num = this->weightTensors.size();
        std::vector<U32> bytes(filter_num);
        CHECK_STATUS(
            rnn_transform_filter_bytes(this->weightTensors, this->p, bytes.data(), &this->archInfo));
        std::vector<Tensor> ftmTensors(filter_num);
        std::vector<Tensor *> tmp(filter_num);
        for (I32 i = 0; i < filter_num; i++) {
            ftmTensors[i].resize(tensor1d(DT_U8, bytes[i]));
            ftmTensors[i].alloc();
            tmp[i] = &ftmTensors[i];
        }
        CHECK_STATUS(rnn_transform_filter(this->weightTensors, this->p, tmp, &this->archInfo));
        this->weightTensors = ftmTensors;
        return SUCCESS;
    }

    EE infer_weight_desc() override
    {
        int directions = (this->p.biDirection) ? 2 : 1;
        int weightNum, biasNum, column;
        if (this->p.numProjection > 0) {
            weightNum = biasNum = 2;
            column = this->p.numProjection;
        } else {
            weightNum = biasNum = 1;
            column = this->p.numOutput;
        }
        int gates = 0;
        switch (this->p.mode) {
            case RNN_LSTM:
                gates = 4;
                break;
            case RNN_GRU:
                gates = 3;
                break;
            case RNN_GRU_LBR:
                gates = 3;
                biasNum++;
                break;
            default:
                return NOT_SUPPORTED;
        }
        U32 filterRow = gates * column;
        U32 filterCol = this->xDim + this->p.numOutput;
        std::vector<TensorDesc> weight_desc(2), bias_desc(2);
        weight_desc[0] = tensor2df(this->dt, DF_NK, filterRow, filterCol);
        weight_desc[1] = tensor2df(this->dt, DF_NK, this->p.numOutput, this->p.numProjection);
        bias_desc[0] = tensor1d(this->dt, filterRow);
        bias_desc[1] = tensor1d(this->dt, this->p.numOutput);
        this->weightTensors = std::vector<Tensor>(directions * weightNum);
        this->biasTensors = std::vector<Tensor>(directions * biasNum);
        for (int i = 0, wid = 0, vid = 0; i < directions; i++) {
            for (int j = 0; j < weightNum; j++, wid++) {
                this->weightTensors[wid].resize(weight_desc[j]);
            }
            for (int j = 0; j < biasNum; j++, vid++) {
                this->biasTensors[vid].resize(bias_desc[j]);
            }
        }
        return SUCCESS;
    }
};

#endif  // _RNNCELL_CPU_H
