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
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        if (this->inputFrameSize != (int)inputDesc.dims[1]) {
            UNI_WARNING_LOG("use dynamic input Splice may encounter error when using clone, "
                            "because many threads may change weight tensor at same time\n");
            this->inputFrameSize = inputDesc.dims[1];
            this->outputFrameSize = outputDesc.dims[1];
            this->transform_filter();
        }
        EmbedParamSpec embedParamSpec;
        embedParamSpec.input_dim = this->inputFrameSize;
        embedParamSpec.num_output = inputDesc.dims[0];
        embedParamSpec.transpose = false;
        CHECK_STATUS(embedding(
            this->weightTensors[0], inputTensor, embedParamSpec, outputTensor, &this->archInfo));
    }

    void get_context_min_max(int *context_min, int *context_max)
    {
        if (this->p.num_context == 0) {
            *context_min = *context_max = 0;
        } else {
            *context_min = *context_max = this->p.context[0];
        }
        for (int i = 0; i < this->p.num_context; i++) {
            *context_min = UNI_MIN(*context_min, this->p.context[i]);
            *context_max = UNI_MAX(*context_max, this->p.context[i]);
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        auto inputDesc = inTensors[0]->get_desc();
        CHECK_REQUIREMENT(inputDesc.nDims == 3);
        this->inputFrameSize = inputDesc.dims[1];
        auto outputDesc = inputDesc;
        if (this->is_valid_indexes()) {
            this->outputFrameSize = this->get_num_indexes() / this->p.num_context;
        } else {
            int context_min, context_max;
            get_context_min_max(&context_min, &context_max);
            this->outputFrameSize = this->inputFrameSize - (context_max - context_min + 1) + 1;
        }
        this->outputFrameSize = UNI_MAX(this->outputFrameSize, 0);
        outputDesc.dims[1] = this->outputFrameSize;
        outputDesc.dims[0] *= this->p.num_context;
        outTensors[0]->resize(outputDesc);
        return SUCCESS;
    }

    bool is_valid_indexes()
    {
        bool ret;
        if (this->p.index_min == 0 && this->p.index_max == this->inputFrameSize - 1) {
            ret = true;
        } else {
            ret = false;
        }
        return ret;
    }

    int get_num_indexes()
    {
        auto curOpWs = this->get_weightspec();
        return curOpWs.bytes_of_weight / bytesOf(curOpWs.mdt);
    }

    EE infer_weight_desc() override
    {
        auto curOpWs = this->get_weightspec();
        int num_indexes = this->get_num_indexes();
        if (curOpWs.weight != nullptr) {
            Tensor weightTensor;
            weightTensor.resize(tensor1d(DT_U32, num_indexes));
            this->weightTensors.push_back(weightTensor);
        }
        return SUCCESS;
    }

    EE transform_filter() override
    {
        if (!this->is_valid_indexes()) {
            Tensor newWeight = Tensor::alloc_sized<CPUMem>(
                tensor1d(DT_U32, this->outputFrameSize * this->p.num_context));
            int context_min, context_max;
            get_context_min_max(&context_min, &context_max);
            U32 *ptr = (U32 *)((CpuMemory *)newWeight.get_memory())->get_ptr();
            int id = 0;
            for (int i = 0; i < this->inputFrameSize; i++) {
                if (i + context_min < 0 || i + context_min >= this->inputFrameSize ||
                    i + context_max < 0 || i + context_max >= this->inputFrameSize) {
                    continue;
                }
                for (int j = 0; j < this->p.num_context; j++, id++) {
                    ptr[id] = i + this->p.context[j];
                }
            }
            CHECK_REQUIREMENT(this->outputFrameSize * this->p.num_context == id);
            this->weightTensors[0] = newWeight;
        }
        return SUCCESS;
    }
};

#endif  // _SPLICE_CPU_H
