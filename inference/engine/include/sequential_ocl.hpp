// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_MALI
#ifndef _SEQUENTIAL_OCL_HPP
#define _SEQUENTIAL_OCL_HPP

#include <cstring>
#include "sys.h"
#include "error.h"
#include "types.h"
#include <string>
#include "tensor.hpp"
#include "operator.hpp"
#include "cnn.h"
#include "op_type.h"
#include "tensor_desc.h"
#include "memory.hpp"
#include "weight_operator.hpp"
#include "pooling.hpp"
#include "convolution.hpp"
#include "bilateral_slice_apply.hpp"
#include "ocl/pooling_ocl.hpp"
#include "memory_ocl.hpp"
#include "ocl/convolution_ocl.hpp"
#include "ocl/bilateral_slice_apply_ocl.hpp"
#include "ocl/fully_connected_ocl.hpp"
#include "ocl/scale_ocl.hpp"

class SequentialOcl : public CNN {
public:
    SequentialOcl(AffinityPolicy affinityPolicy, DataType dt, std::string name)
        : CNN(affinityPolicy, dt, name)
    {
        input_output_same = false;
    }
    virtual ~SequentialOcl()
    {}

    EE ready(std::vector<TensorDesc> dims, std::shared_ptr<U8> modelPtr, U32 numOutput)
    {
        this->ops[0]->set_schedule(this->deviceInfo.schedule);
        input_output_same = this->ops[0]->can_input_output_the_same();
        CHECK_STATUS(this->infer_output_tensors_size(dims, numOutput));
        std::vector<Tensor> inTensors;
        std::vector<Tensor> outTensors;
        for (U32 i = 0; i < inputTensors.size(); i++) {
            inTensors.push_back(*inputTensors[i].get());
        }
        for (U32 i = 0; i < outputTensors.size(); i++) {
            outTensors.push_back(*outputTensors[i].get());
        }
        this->ops[0]->set_input_output_tensors(inTensors, outTensors);
        this->ops[0]->set_algorithm_map(this->algorithmMap);

        if (this->ops[0]->is_weight()) {
            if (this->ops[0]->get_type() == OT_Conv) {
                auto convOpPtr = dynamic_cast<Convolution *>(this->ops[0].get());
                auto weightOp = (WeightOperator *)convOpPtr;
                weightOp->set_hasBias(true);
                CHECK_STATUS(convOpPtr->init_weight_bias_from_model(&modelPtr));
                CHECK_STATUS(convOpPtr->infer_forward_algorithm(this->algorithmMap));
                CHECK_STATUS(convOpPtr->transform_filter());
            }
            if (this->ops[0]->get_type() == OT_FC) {
                auto fcOpPtr = dynamic_cast<FullyConnected *>(this->ops[0].get());
                auto weightOp = (WeightOperator *)fcOpPtr;
                weightOp->set_hasBias(true);
                CHECK_STATUS(fcOpPtr->init_weight_bias_from_model(&modelPtr));
                CHECK_STATUS(fcOpPtr->transform_filter());
            }
            if (this->ops[0]->get_type() == OT_Scale) {
                auto scaleOpPtr = dynamic_cast<Scale *>(this->ops[0].get());
                auto weightOp = (WeightOperator *)scaleOpPtr;
                weightOp->set_hasBias(true);
                CHECK_STATUS(scaleOpPtr->init_weight_bias_from_model(&modelPtr));
            }
        }
        this->infer_tmp_memory_size();
        this->assign_tmp_tensor();
        this->alloc_output_host_tensors(numOutput);
        return SUCCESS;
    }

    EE infer_output_tensors_size(std::map<std::string, TensorDesc>) override
    {
        return NOT_SUPPORTED;
    }

    void assign_output_tensor() override
    {}

    EE infer_output_tensors_size(std::vector<TensorDesc> dims, U32 outputTensorNum)
    {
        std::vector<Tensor *> inTensors;
        std::vector<Tensor *> outTensors;
        for (U32 i = 0; i < dims.size(); ++i) {
            std::shared_ptr<Tensor> tmpTensor(new Tensor(OCLMem));
            tmpTensor->resize(dims[i]);
            inputTensors.push_back(tmpTensor);
            inTensors.push_back(inputTensors[i].get());
        }
        for (U32 i = 0; i < outputTensorNum; ++i) {
            std::shared_ptr<Tensor> tmpTensor(new Tensor(OCLMem));
            outputTensors.push_back(tmpTensor);
            outTensors.push_back(outputTensors[i].get());
        }

        CHECK_STATUS(this->ops[0]->infer_output_tensors_size(inTensors, outTensors));
        for (auto p : inTensors) {
            p->alloc();
        }
        return SUCCESS;
    }

    EE infer_gclmem_descs(std::map<std::string, TensorDesc>)
    {
        return NOT_SUPPORTED;
    }

    void alloc_output_host_tensors(U32 outputTensorNum)
    {
        for (U32 i = 0; i < outputTensorNum; i++) {
            auto mem = (OclMemory *)outputTensors[i]->get_memory();
            mem->mapped_alloc();
        }
    }

    void infer_tmp_memory_size() override
    {
        maxTmpElements = 0;
        for (auto op : this->ops) {
            auto len = op->infer_tmp_memory_size();
            if (len > maxTmpElements) {
                maxTmpElements = len;
            }
        }
    }

    void assign_tmp_tensor() override
    {
        this->temp = Tensor(OCLMem);
        if (maxTmpElements) {
            temp.resize(tensor1d(DT_U8, maxTmpElements));
            temp.alloc();
        }
        for (auto op : this->ops) {
            op->set_tmp_memory(temp);
        }
    }

    void add(std::shared_ptr<Operator> op)
    {
        this->ops.push_back(op);
    }

    void mark_input_output()
    {
        if (this->deviceInfo.schedule == MALI) {
            U32 tmpBufSize = 0;
            for (U32 i = 0; i < inputTensors.size(); i++) {
                Tensor *inputTensor = inputTensors[i].get();
                TensorDesc desc = inputTensor->get_desc();
                U32 size = tensorNumBytes(desc);
                ArchInfo archInfo;
                archInfo.arch = MALI;
                tmpBufSize = (tmpBufSize < size) ? size : tmpBufSize;
            }

            if (tmpBufSize > maxTmpElements) {
                maxTmpElements = tmpBufSize;
            }
            temp.resize(tensor1d(DT_U8, maxTmpElements));
            temp.alloc();
        }
    }

    void set_input_tensors(std::vector<Tensor> modelInputTensors)
    {
        for (U32 i = 0; i < modelInputTensors.size(); i++) {
            auto hostMem = (CpuMemory *)modelInputTensors[i].get_memory();
            U8 *hostPtr = (U8 *)hostMem->get_ptr();
            TensorDesc hostDesc = modelInputTensors[i].get_desc();
            auto *mem = (OclMemory *)inputTensors[i]->get_memory();
            GCLMem_t input = (GCLMem_t)mem->get_ptr();
            auto *tmpmem = (OclMemory *)temp.get_memory();
            GCLMem_t tmp = (GCLMem_t)tmpmem->get_ptr();
            CHECK_STATUS(ocl_set_input(this->handle.get(), input, hostDesc, hostPtr, tmp, true));
        }
        gcl_finish(this->handle.get());
    }

    std::vector<std::shared_ptr<Tensor>> get_output_tensors()
    {
        return this->outputTensors;
    }

#ifdef _USE_MALI
#else
    EE ConvBiasAssignmentAndWeightTransform()
    {
        return SUCCESS;
    }

    EE FCBiasAssignmentAndWeight()
    {
        return SUCCESS;
    }
#endif

private:
    using Model::ready;
    U32 maxTmpElements;
    Tensor temp;
    std::vector<std::shared_ptr<Tensor>> inputTensors;
    std::vector<std::shared_ptr<Tensor>> outputTensors;
    bool input_output_same;
};
#endif
#endif
