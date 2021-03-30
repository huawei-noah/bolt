// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _TDNN_FULLY_CONNECTED_CPU_H
#define _TDNN_FULLY_CONNECTED_CPU_H

#include "fully_connected_cpu.hpp"

//#define TDNN_SLIDE_OPT

class TdnnFullyConnectedCPU : public FullyConnectedCPU {
public:
    TdnnFullyConnectedCPU(DataType dt, TdnnParamSpec tdnnParam)
        : FullyConnectedCPU(dt, FullyConnectedParamSpec{}, 0)
    {
        this->slide_size = 1;
        this->tdnn = tdnnParam;
        this->inputFrameSize = 0;
        this->outputFrameSize = 0;
        this->p.num_outputs = this->tdnn.num_outputs;
        this->p.num_slices = 1;
    }

    OperatorType get_type() override
    {
        return OT_Tdnn;
    }

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<TdnnFullyConnectedCPU> mem =
            std::shared_ptr<TdnnFullyConnectedCPU>(new TdnnFullyConnectedCPU(this->dt, this->tdnn));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        if (inputDesc.nDims != 3 || outputDesc.nDims != 3 ||
            inputDesc.dims[inputDesc.nDims - 1] != 1) {
            UNI_ERROR_LOG("TdnnFullyConnectedCPU not support batch\n");
        }
        if (this->inputFrameSize != (int)inputDesc.dims[1]) {
            UNI_WARNING_LOG("use dynamic input Tdnn may encounter error when using clone, "
                            "because many threads may change weight tensor at same time\n");
            this->inputFrameSize = inputDesc.dims[1];
            this->outputFrameSize = outputDesc.dims[1];
            this->generate_index();
        }
        if (this->slide) {
            // slide output tensor
            U32 tileSize = bytesOf(outputDesc.dt) * outputDesc.dims[0];
            U8 *output = (U8 *)((CpuMemory *)outputTensor.get_memory())->get_ptr();
            int windowSize = this->outputFrameSize;
#ifdef TDNN_SLIDE_OPT
            if (this->allWindowSize.find(output) != this->allWindowSize.end()) {
                windowSize = this->allWindowSize[output];
            }
#endif
            // batch
            for (U32 i = 0; i < inputDesc.dims[inputDesc.nDims - 1]; i++) {
                // frame num
                for (int j = this->outputFrameSize - windowSize;
                     j < this->outputFrameSize - this->slide_size; j++) {
                    U8 *dst = output + (i * this->outputFrameSize + j) * tileSize;
                    U8 *src = dst + tileSize;
                    memcpy(dst, src, tileSize);
                }
            }
        }
        TensorDesc spliceDesc = this->get_splice_desc(inputDesc);
        Tensor spliceResult;
        spliceResult.resize(spliceDesc);
        if (this->slide && this->is_increasing_context()) {
            int offset = tensorNumBytes(inputDesc) - tensorNumBytes(spliceDesc);
            std::shared_ptr<U8> validInput =
                ((CpuMemory *)inputTensor.get_memory())->get_shared_ptr();
            std::shared_ptr<U8> spliceBuffer =
                std::shared_ptr<U8>(validInput, validInput.get() + offset);
            ((CpuMemory *)spliceResult.get_memory())->set_shared_ptr(spliceBuffer);
        } else {
            int offset = this->temp.capacity() - tensorNumBytes(spliceDesc);
            std::shared_ptr<U8> tmpBuffer = ((CpuMemory *)this->temp.get_memory())->get_shared_ptr();
            std::shared_ptr<U8> spliceBuffer =
                std::shared_ptr<U8>(tmpBuffer, tmpBuffer.get() + offset);
            ((CpuMemory *)spliceResult.get_memory())->set_shared_ptr(spliceBuffer);

            EmbedParamSpec embedParamSpec;
            embedParamSpec.input_dim = this->inputFrameSize;
            embedParamSpec.num_output = inputDesc.dims[0];
            embedParamSpec.transpose = false;
            CHECK_STATUS(
                embedding(this->index, inputTensor, embedParamSpec, spliceResult, &this->archInfo));
        }
        Tensor tmpOutputTensor;
        if (this->slide) {
            TensorDesc tmpOutputDesc = outputDesc;
            tmpOutputDesc.dims[1] = this->slide_size;
            tmpOutputTensor.resize(tmpOutputDesc);
            int offset = tensorNumBytes(outputDesc) - tensorNumBytes(tmpOutputDesc);
            std::shared_ptr<U8> outputBuffer =
                ((CpuMemory *)outputTensor.get_memory())->get_shared_ptr();
            std::shared_ptr<U8> validOutputBuffer =
                std::shared_ptr<U8>(outputBuffer, outputBuffer.get() + offset);
            ((CpuMemory *)tmpOutputTensor.get_memory())->set_shared_ptr(validOutputBuffer);
        } else {
            tmpOutputTensor = outputTensor;
        }

        this->inputTensors[0] = spliceResult;
        this->outputTensors[0] = tmpOutputTensor;
        FullyConnectedCPU::run();
        this->inputTensors[0] = inputTensor;
        this->outputTensors[0] = outputTensor;

        if (this->tdnn.activation_type != ACTIVATION_NULL) {
            ActivationParamSpec activationDesc;
            activationDesc.mode = this->tdnn.activation_type;
            activationDesc.value[0] = 0;
            CHECK_STATUS(
                activation(tmpOutputTensor, activationDesc, tmpOutputTensor, &this->archInfo));
        }
    }

    bool is_increasing_context()
    {
        bool ret = true;
        for (int i = 1; i < this->tdnn.num_context; i++) {
            if (this->tdnn.context[i] != this->tdnn.context[i - 1] + 1) {
                ret = false;
                break;
            }
        }
        return ret;
    }

    void get_context_min_max(int *context_min, int *context_max)
    {
        if (this->tdnn.num_context == 0) {
            *context_min = *context_max = 0;
        } else {
            *context_min = *context_max = this->tdnn.context[0];
        }
        for (int i = 0; i < this->tdnn.num_context; i++) {
            *context_min = UNI_MIN(*context_min, this->tdnn.context[i]);
            *context_max = UNI_MAX(*context_max, this->tdnn.context[i]);
        }
    }

    void generate_index()
    {
        if (this->slide && this->is_increasing_context()) {
            return;
        }
        int context_min, context_max;
        get_context_min_max(&context_min, &context_max);
        int index_num = this->outputFrameSize * this->tdnn.num_context;
        int i = 0;
        if (this->slide) {
            index_num = this->slide_size * this->tdnn.num_context;
            i = this->inputFrameSize - context_max - this->slide_size;
        }
        this->index = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U32, index_num));
        U32 *ptr = (U32 *)((CpuMemory *)this->index.get_memory())->get_ptr();
        int id = 0;
        for (; i < this->inputFrameSize; i++) {
            if (i + context_min < 0 || i + context_min >= this->inputFrameSize ||
                i + context_max < 0 || i + context_max >= this->inputFrameSize) {
                continue;
            }
            for (int j = 0; j < this->tdnn.num_context; j++, id++) {
                ptr[id] = i + this->tdnn.context[j];
            }
        }
        CHECK_REQUIREMENT(index_num == id);
    }

    void set_slide()
    {
        this->slide = false;
        if (this->tensorPos.size() == 2) {
            // output tensor is not reused
            if (this->tensorPos[1] == -1) {
                this->slide = true;
            }
        } else {
            UNI_ERROR_LOG("Tdnn's memory reuse parameter is invalid.\n");
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        if (inTensors.size() == 0 || inTensors[0] == nullptr) {
            return NULL_POINTER;
        }
        this->set_slide();
        TensorDesc inputDesc = inTensors[0]->get_desc();
        CHECK_REQUIREMENT(inputDesc.nDims == 3);
        this->numInput = inputDesc.dims[0] * this->tdnn.num_context;
        this->inputFrameSize = inputDesc.dims[1];
        int context_min, context_max;
        get_context_min_max(&context_min, &context_max);
        this->outputFrameSize = this->inputFrameSize - (context_max - context_min + 1) + 1;
        this->outputFrameSize = UNI_MAX(this->outputFrameSize, 0);
        int batch = inputDesc.dims[inputDesc.nDims - 1];
        if (this->slide && batch == 1) {
            this->mvm = true;
        } else {
            this->mvm = false;
        }
        TensorDesc outputDesc = inputDesc;
        outputDesc.dims[1] = outputFrameSize;
        outputDesc.dims[0] = this->tdnn.num_outputs;
        EE ret;
        if (outTensors.size() == 0 || outTensors[0] == nullptr) {
            ret = NULL_POINTER;
        } else {
            outTensors[0]->resize(outputDesc);
            ret = SUCCESS;
        }
        return ret;
    }

    TensorDesc get_splice_desc(TensorDesc inputDesc)
    {
        CHECK_REQUIREMENT(inputDesc.nDims == 3);
        int num;
        if (this->slide) {
            num = this->slide_size;
        } else {
            num = this->outputFrameSize;
        }
        return tensor3df(inputDesc.dt, inputDesc.df, inputDesc.dims[2], num, this->numInput);
    }

    U32 infer_tmp_memory_size() override
    {
        TensorDesc spliceDesc = this->get_splice_desc(this->inputTensors[0].get_desc());
        U32 bytes = 0;
        Tensor tmpInput;
        tmpInput.resize(desc_process(spliceDesc));
        CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(
            tmpInput, this->weightTensors[0], &bytes, &this->archInfo));
        bytes += tensorNumBytes(spliceDesc);
        return bytes;
    }

    EE transform_filter() override
    {
        this->generate_index();
        TensorDesc spliceDesc = this->get_splice_desc(this->inputTensors[0].get_desc());
        return FullyConnectedCPU::transform_filter(spliceDesc);
    }

#ifdef TDNN_SLIDE_OPT
    void set_input_tensors(std::vector<Tensor> &it) override
    {
        Operator::set_input_tensors(it);
        if (this->slide && it.size() > 0) {
            int context_min, context_max;
            get_context_min_max(&context_min, &context_max);
            int windowSize = context_max - context_min + 1;
            auto ptr = ((CpuMemory *)it[0].get_memory())->get_ptr();
            if (allWindowSize.find(ptr) != allWindowSize.end()) {
                windowSize = UNI_MAX(windowSize, allWindowSize[ptr]);
            }
            allWindowSize[ptr] = windowSize;
        }
    }
#endif
private:
    bool slide;
    int slide_size;
    TdnnParamSpec tdnn;
    int inputFrameSize;
    int outputFrameSize;
    Tensor index;
    static std::map<void *, int> allWindowSize;
};

#endif  // _TDNN_FULLY_CONNECTED_CPU_H
