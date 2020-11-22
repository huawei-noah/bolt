// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FLOW_INCLUDE_NODE_H_
#define FLOW_INCLUDE_NODE_H_

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "flow.pb.h"
#include "cnn.h"
#include "flow_function_factory.h"

class Node {
public:
    Node();

    ~Node();

    Node clone();

    void setNodeParameter(flow::NodeParameter nodeParameter);

    flow::NodeParameter getNodeParameter();

    EE inferOutputSize();

    void setPrecision(DataType precision);

    void initInference(AffinityPolicy affinityPolicy);

    unsigned int getTmpBufferSize();

    void setTmpBuffer(std::shared_ptr<Tensor> tmpTensor);

    EE ready();

    void setInput(std::map<std::string, std::shared_ptr<Tensor>> inputs);

    void setOutput(std::map<std::string, std::shared_ptr<Tensor>> outputs);

    void setRuntime(int cpuId, Arch arch);

    EE run();

private:
    DataType precision;
    flow::NodeParameter nodeParameter;
    std::map<std::string, std::shared_ptr<Tensor>> inputs;
    std::shared_ptr<Tensor> tmpTensor;
    std::map<std::string, std::shared_ptr<Tensor>> outputs;
    FlowFunction inferOutputSizeFunction;
    std::vector<std::string> inferOutputSizeParameter;
    FlowFunction preprocessFunction;
    std::vector<std::string> preprocessParameter;
    CNN boltModel;
    std::vector<std::string> inferenceParameter;
    FlowFunction postprocessFunction;
    std::vector<std::string> postprocessParameter;
};
#endif  // FLOW_INCLUDE_NODE_H_
