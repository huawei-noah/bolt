// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _TFSLICE_CPU_H
#define _TFSLICE_CPU_H

#include "tfslice.hpp"

class TfSliceCPU : public TfSlice {
public:
    TfSliceCPU(DataType dt, TfSliceParamSpec p) : TfSlice(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<TfSliceCPU> mem =
            std::shared_ptr<TfSliceCPU>(new TfSliceCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    TfSliceParamSpec get_param(std::vector<TensorDesc> descs)
    {
        TfSliceParamSpec ps = this->p;
        int j = 0, jj = 0;
        for (U32 i = 0; i < ps.num_dims; i++) {
            if (ps.begin[i] == UNI_RESERVE) {
                ps.begin[i] = descs[j].dims[descs[j].nDims + jj];
                jj++;
            }
        }
        if (jj != 0) {
            j++;
            jj = 0;
        }
        for (U32 i = 0; i < ps.num_dims; i++) {
            if (ps.end[i] == UNI_RESERVE) {
                ps.end[i] = descs[j].dims[descs[j].nDims + jj];
                jj++;
            }
        }
        return ps;
    }

    void run() override
    {
        TfSliceParamSpec ps = p;
        if (inputTensors.size() > 1) {
            std::vector<TensorDesc> descs;
            for (U32 i = 1; i < inputTensors.size(); i++) {
                descs.push_back(inputTensors[i].get_desc());
            }
            ps = get_param(descs);
        }
        Tensor inputTensor = this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        CHECK_STATUS(tfslice(inputTensor, ps, this->temp, outputTensor, &this->archInfo));
        this->outputTensors[0].set_scale(this->inputTensors[0].get_scale());
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inputTensors, std::vector<Tensor *> outputTensors) override
    {
        TfSliceParamSpec ps = p;
        if (inputTensors.size() > 1) {
            std::vector<TensorDesc> descs;
            for (U32 i = 1; i < inputTensors.size(); i++) {
                descs.push_back(inputTensors[i]->get_desc());
            }
            ps = get_param(descs);
        }
        return tfslice_infer_output_size(inputTensors[0], ps, outputTensors[0], &this->archInfo);
    }
};

#endif  // _TFSLICE_CPU_H
