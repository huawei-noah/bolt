// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>

#include "graph.h"
#include "tensor.hpp"
#include "node.h"
#include "flow.pb.h"

EE tinybertInferOutputSize(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    TensorDesc inputDesc = inputs.begin()->second->get_desc();
    outputs["intent_softmax"]->resize(tensor3df(DT_F32, DF_MTK, 1, 1, 65));
    outputs["slot_softmax"]->resize(tensor3df(DT_F32, DF_MTK, 1, inputDesc.dims[1], 45));
    return SUCCESS;
}

std::map<std::string, std::shared_ptr<Tensor>> inputOutput()
{
    const int length = 9;
    int words[length] = {101, 2224, 8224, 7341, 2000, 22149, 2000, 2899, 102};

    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    TensorDesc inputDesc = tensor2df(DT_U32, DF_NORMAL, 1, 9);
    tensors["tinybert_words"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["tinybert_words"]->resize(inputDesc);
    tensors["tinybert_words"]->alloc();
    memcpy(((CpuMemory *)tensors["tinybert_words"]->get_memory())->get_ptr(), words,
        tensorNumBytes(inputDesc));

    tensors["tinybert_positions"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["tinybert_positions"]->resize(inputDesc);
    tensors["tinybert_positions"]->alloc();
    tensors["tinybert_token_type"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["tinybert_token_type"]->resize(inputDesc);
    tensors["tinybert_token_type"]->alloc();
    unsigned int *positionPtr =
        (unsigned int *)((CpuMemory *)tensors["tinybert_positions"]->get_memory())->get_ptr();
    unsigned int *tokenTypePtr =
        (unsigned int *)((CpuMemory *)tensors["tinybert_token_type"]->get_memory())->get_ptr();
    for (int i = 0; i < length; i++) {
        positionPtr[i] = i;
        tokenTypePtr[i] = 0;
    }

    DataType dataType = DT_F32;
    tensors["intent_softmax"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["intent_softmax"]->resize(tensor3df(dataType, DF_MTK, 1, 1, 65));
    tensors["intent_softmax"]->alloc();

    tensors["slot_softmax"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["slot_softmax"]->resize(tensor3df(dataType, DF_MTK, 1, length, 45));
    tensors["slot_softmax"]->alloc();

    return tensors;
}

int main(int argc, char *argv[])
{
    flowRegisterFunction("tinybertInferOutputSize", tinybertInferOutputSize);
    std::string tinybertGraphPath = argv[1];
    std::map<std::string, std::shared_ptr<Tensor>> data = inputOutput();
    Graph<flow::GraphParameter, Node, Tensor> graph;
    graph.init(tinybertGraphPath);
    graph.ready(DT_F32, AFFINITY_CPU_HIGH_PERFORMANCE, -1);
    graph.run(data);
    return 0;
}
