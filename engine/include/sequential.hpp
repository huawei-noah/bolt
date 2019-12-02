// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SEQUENTIAL_HPP
#define _SEQUENTIAL_HPP

#include <cstring>
#include "sys.h"
#include "error.h"
#include "type.h"
#include <string>
#include "tensor.hpp"
#include "operator.hpp"
#include "convolution.hpp"
#include "fully_connected_eltwise.hpp"
#include "model.hpp"
#include "model_tools.h"
#include "tensor_desc.h"
#include "sequential.hpp"
#include "lstm.hpp"


template<Arch A>
class Sequential:public Model<A> {
public:
    Sequential(DataType dt, std::string name): Model<A>(dt, name) { }

    EE infer_output_tensors_size(Vec<TensorDesc> dims) override
    {
        if (dims.size() != 1) {
            return NOT_SUPPORTED;
        }
        Vec<TensorDesc> inDims = dims;
        Vec<TensorDesc> outDims(1);

        this->dimsOp = { dims };

        auto num = [](Vec<TensorDesc> dims) -> U32 {
            U32 ret = 0;
            for (auto d: dims) ret += tensorNumElements(d);
            return ret;
        };

        maxOutputElements = num(inDims);
        for (auto op: this->ops) {
            CHECK_STATUS_WITH_RETURN(op->infer_output_tensors_size(inDims, &outDims));
            dimsOp.push_back(outDims);
            auto numElements = num(outDims);
            if(maxOutputElements < numElements) maxOutputElements = numElements;
            inDims = outDims;
        }
        return SUCCESS;
    }


    void assign_output_tensor() override
    {
        auto firstPtr = (U8*)operator new(bytesOf(this->dt) * maxOutputElements);
        std::shared_ptr<U8> firstSharedPtr(firstPtr);
        auto secondPtr = (U8*)operator new(bytesOf(this->dt) * maxOutputElements);
        std::shared_ptr<U8> secondSharedPtr(secondPtr);
        for (auto i = 0U; i < this->ops.size(); i++) {
            auto op = this->ops[i];
            auto inDims = dimsOp[i];
            auto outDims = dimsOp[i+1];

            Vec<Tensor> inTensors;
            U32 index = 0;
            for (auto d: inDims) {
                inTensors.push_back(Tensor(d, std::shared_ptr<U8>(firstSharedPtr, (U8*)firstPtr + index*bytesOf(this->dt))));
                index += tensorNumElements(d);
            }

            Vec<Tensor> outTensors;
            index = 0;
            for (auto d: outDims) {
                outTensors.push_back(Tensor(d, std::shared_ptr<U8>(secondSharedPtr, (U8*)secondPtr + index*bytesOf(this->dt))));
                index += tensorNumElements(d);
            }

            op->set_input_output_tensors(inTensors, outTensors);

            std::swap(firstPtr, secondPtr); 
            std::swap(firstSharedPtr, secondSharedPtr);
        }
    }

    //TODO 0823
    EE ConvBiasAssignmentAndWeightTransform() {
        return SUCCESS;
    }

    //TODO 0823
    EE FCBiasAssignmentAndWeight() {
        return SUCCESS;
    }


    EE ready(Vec<TensorDesc> dims, std::shared_ptr<U8> modelPtr)
    {
        this->infer_output_tensors_size(dims);
        this->assign_output_tensor();

        U8* curPtr = modelPtr.get();
        for (auto op : this->ops) {
            if (op->is_weight()) {
                if (op->get_op_type() == OT_Conv) {
                    auto convOpPtr = dynamic_cast<Convolution<A>*>(op.get());
                    CHECK_STATUS_WITH_RETURN(convOpPtr->init_weight_bias_from_model(&curPtr));
                    CHECK_STATUS_WITH_RETURN(convOpPtr->infer_forward_algorithm());
                    CHECK_STATUS_WITH_RETURN(convOpPtr->transform_filter());
                } else if (op->get_op_type() == OT_FC) {
                    auto fcOpPtr = dynamic_cast<FullyConnectedEltwise<A>*>(op.get());
                    CHECK_STATUS_WITH_RETURN(fcOpPtr->init_weight_bias_from_model(&curPtr));
                    CHECK_STATUS_WITH_RETURN(fcOpPtr->transform_filter());
                } else if (op->get_op_type()==OT_LSTM) {
                    auto lstmOpPtr = dynamic_cast<Lstm<A>*>(op.get());
                    CHECK_STATUS_WITH_RETURN(lstmOpPtr->init_weight_bias_from_model(&curPtr));
                    CHECK_STATUS_WITH_RETURN(lstmOpPtr->transform_filter());
                }
            }
        }

        this->infer_tmp_memory_size();
        this->assign_tmp_tensor();
        return SUCCESS;
    }

    void infer_tmp_memory_size() override
    {
        tmpElements.clear();
        maxTmpElements = 0;

        for (auto op: this->ops) {
            auto len = op->infer_tmp_memory_size();
            tmpElements.push_back(len);
            if(len > maxTmpElements) maxTmpElements = len;
        }
    }

    void assign_tmp_tensor() override
    {
        auto secondPtrtr = std::shared_ptr<U8>((U8*)operator new(maxTmpElements));
        for (auto op: this->ops) {
            op->set_tmp_memory(maxTmpElements, secondPtrtr);
        }
    }

    void add(std::shared_ptr<Operator<A> > op)
    {
        this->ops.push_back(op);
    }

    Vec<Tensor> get_inputTensors()
    {
        auto op = this->ops[0].get();
        return op->get_input_tensors();
    }

    Vec<Tensor> get_output_tensors()
    {
        auto len = this->ops.size();
        auto op = this->ops[len-1].get();
        return op->get_output_tensors();
    }

    void set_input_tensors(Vec<Tensor> inputTensors)
    {
        auto op = this->ops[0].get();
        op->set_input_tensors(inputTensors);
    }

private:
    U32 maxOutputElements;
    Vec<Vec<TensorDesc> > dimsOp;
    U32 maxTmpElements;
    Vec<U32> tmpElements;
};
#endif


