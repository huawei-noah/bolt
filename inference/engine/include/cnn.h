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
#include <cstring>
#include "model.hpp"
#include "memory_tracker.hpp"
#ifdef _USE_MALI
#include "gcl_common.h"
#endif

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

    void reready(std::map<std::string, TensorDesc> inputDescMap);

    EE mark_input_output();

    void copy_to_named_input(std::string inputName, const U8 *data);

    void set_input_tensors_value(std::map<std::string, std::shared_ptr<U8>> modelTensorsInput);

    std::map<std::string, std::shared_ptr<Tensor>> get_inputs();

    std::map<std::string, std::shared_ptr<Tensor>> get_outputs();

    Tensor get_tensor_by_name(std::string tensorName);

    TensorDesc get_tensor_desc_by_name(std::string tensorName);

    std::vector<std::string> get_model_input_tensor_names();

    std::vector<TensorDesc> get_model_input_tensor_descs();

    std::vector<std::string> get_model_output_tensor_names();

    EE infer_output_tensors_size(std::map<std::string, TensorDesc> inputDescMap) override;

    void assign_output_tensor() override;

    void addOutputTensorNames(std::vector<std::string> outputTensorNames);

    void run() override;

#ifdef _USE_MALI
    void mali_prepare(bool reset);
#endif
private:
    std::shared_ptr<Tensor> allocate_tensor(U32 size = 0);

    void add(std::shared_ptr<Operator> op,
        std::vector<std::string> inputTensorsName,
        std::vector<std::string> outputTensorsName);

    void infer_layout_desc();

    void update_op_tensors();

    void set_input_tensors_desc(std::map<std::string, TensorDesc> inputDescMap);

    void infer_tmp_memory_size() override;

    void assign_tmp_tensor() override;

    void check_memory_reuse_ratio();

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

    std::vector<std::string> modelInputTensorNames;
    std::vector<TensorDesc> modelInputTensorDescs;
    std::vector<std::string> modelOutputTensorNames;
    MemoryTracker memoryTracker;
};
#endif
