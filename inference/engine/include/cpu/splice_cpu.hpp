// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _SPLICE_CPU_H
#define _SPLICE_CPU_H

#include "splice.hpp"

class SpliceCPU : public Splice {
public:
    SpliceCPU(DataType dt, SpliceParamSpec p) : Splice(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<SpliceCPU> mem =
            std::shared_ptr<SpliceCPU>(new SpliceCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        EmbedParamSpec embedParamSpec;
        embedParamSpec.input_dim = inputDesc.dims[1];
        embedParamSpec.num_output = inputDesc.dims[0];
        embedParamSpec.transpose = false;
        CHECK_STATUS(embedding(
            weightTensors[0], inputTensor, embedParamSpec, outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        auto inDim = inTensors[0]->get_desc();
        CHECK_REQUIREMENT(this->p.outputDim % inDim.dims[0] == 0);
        auto outDim = inDim;
        outDim.dims[1] = this->p.numIndices / (this->p.outputDim / inDim.dims[0]);
        outDim.dims[0] = this->p.outputDim;
        outTensors[0]->resize(outDim);
        return SUCCESS;
    }

    EE infer_weight_desc() override
    {
        auto curOpWs = this->get_weightspec();
        if (curOpWs.weight != nullptr) {
            Tensor weightTensor;
            weightTensor.resize(tensor1d(DT_U32, this->p.numIndices));
            this->weightTensors.push_back(weightTensor);
        }
        return SUCCESS;
    }
};

#endif  // _SPLICE_CPU_H
