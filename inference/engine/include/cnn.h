// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _CNN_H
#define _CNN_H

#include <string>
#include "model.hpp"
#include "memory_tracker.hpp"
#include "model_spec.h"

class CNN : public Model {
public:
    CNN()
    {}

    explicit CNN(AffinityPolicy affinityPolicy, DataType dt, std::string name)
        : Model(affinityPolicy, dt, name)
    {}

    virtual ~CNN() = default;

    CNN clone();

    void sort_operators_sequential(const ModelSpec *ms);

    void initialize_ops(const ModelSpec *ms);

    void ready(std::map<std::string, TensorDesc> inputDescMap) override;

    EE mark_input_output();

    std::map<std::string, TensorDesc> get_input_desc();

    std::map<std::string, std::shared_ptr<Tensor>> get_input();

    void set_input_by_assign(std::map<std::string, std::shared_ptr<U8>> modelTensorsInput);

    void set_input_by_copy(std::map<std::string, U8 *> modelTensorsInput);

    void run() override;

    std::map<std::string, TensorDesc> get_output_desc();

    std::map<std::string, std::shared_ptr<Tensor>> get_output();

    void reready(std::map<std::string, TensorDesc> inputDescMap);

    Tensor get_tensor_by_name(std::string tensorName);

    TensorDesc get_tensor_desc_by_name(std::string tensorName);

private:
    std::shared_ptr<Tensor> allocate_tensor(U32 size = 0);

    void add(std::shared_ptr<Operator> op,
        std::vector<std::string> &inputTensorsName,
        std::vector<std::string> &outputTensorsName);

    void infer_layout_desc();

    void update_op_tensors();

    void set_input_desc(std::map<std::string, TensorDesc> inputDescMap);

    void infer_tmp_memory_size() override;

    void assign_tmp_tensor() override;

    void check_memory_reuse_ratio();

    EE infer_output_tensors_size(std::map<std::string, TensorDesc> inputDescMap) override;

    void assign_output_tensor() override;

    void clean_tensorMap_desc();

private:
    std::map<std::string, std::shared_ptr<Tensor>> tensorMap;
    std::map<std::string, std::shared_ptr<Operator>> operatorMap;
    std::map<std::string, std::vector<std::vector<std::string>>> operatorTensorMap;

    std::set<std::string> weightOpOutputNames;
    std::map<std::string, std::shared_ptr<Tensor>> inputTensors;
    std::map<std::string, std::shared_ptr<Tensor>> outputTensors;
    std::vector<std::shared_ptr<Tensor>> storageMemory;
    Tensor tmpTensor;

    std::vector<std::string> sortedOps;

    MemoryTracker memoryTracker;
};
#endif
