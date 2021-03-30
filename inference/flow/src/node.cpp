// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "node.h"
#include "inference.hpp"
#include "profiling.h"

Node::Node()
{}

Node::~Node()
{}

Node Node::clone()
{
    UNI_DEBUG_LOG("node %s clone start\n", this->nodeParameter.name().c_str());
    Node node = *this;
    node.boltModel = node.boltModel.clone();
    UNI_DEBUG_LOG("node %s clone end\n", this->nodeParameter.name().c_str());
    return node;
}

void Node::setNodeParameter(flow::NodeParameter nodeParameter)
{
    this->nodeParameter = nodeParameter;
    for (int i = 0; i < this->nodeParameter.infer_output_size_parameter_size(); i++) {
        this->inferOutputSizeParameter.push_back(this->nodeParameter.infer_output_size_parameter(i));
    }
    for (int i = 0; i < this->nodeParameter.preprocess_parameter_size(); i++) {
        this->preprocessParameter.push_back(this->nodeParameter.preprocess_parameter(i));
    }
    for (int i = 0; i < this->nodeParameter.inference_parameter_size(); i++) {
        this->inferenceParameter.push_back(this->nodeParameter.inference_parameter(i));
    }
    for (int i = 0; i < this->nodeParameter.postprocess_parameter_size(); i++) {
        this->postprocessParameter.push_back(this->nodeParameter.postprocess_parameter(i));
    }
}

flow::NodeParameter Node::getNodeParameter()
{
    return this->nodeParameter;
}

EE Node::inferOutputSize()
{
    std::string inferOutputSizeFunctionName =
        (this->inferOutputSizeParameter.size() > 0) ? this->inferOutputSizeParameter[0] : "NULL";
    UNI_DEBUG_LOG("node %s infer output size use %s begin\n", this->nodeParameter.name().c_str(),
        inferOutputSizeFunctionName.c_str());
    this->inferOutputSizeFunction = flowGetFunctionByName(inferOutputSizeFunctionName);
    EE ret = SUCCESS;
    if (this->inferOutputSizeFunction != NULL) {
        ret = this->inferOutputSizeFunction(
            this->inputs, this->tmpTensor, this->outputs, this->inferOutputSizeParameter);
    } else {
        std::map<std::string, TensorDesc> inferenceOutputDescs = this->boltModel.get_output_desc();
        for (auto iter : inferenceOutputDescs) {
            std::string name = iter.first;
            if (this->outputs.find(name) == this->outputs.end()) {
                this->outputs[name] = std::shared_ptr<Tensor>(new Tensor());
            }
            this->outputs[name]->resize(iter.second);
        }
    }
    UNI_DEBUG_LOG("node %s infer output size end\n", this->nodeParameter.name().c_str());
    return ret;
}

void Node::setPrecision(DataType precision)
{
    this->precision = precision;
}

unsigned int Node::getTmpBufferSize()
{
    return this->nodeParameter.tmp() * bytesOf(this->precision);
}

void Node::setTmpBuffer(std::shared_ptr<Tensor> tmpTensor)
{
    this->tmpTensor = tmpTensor;
}

void Node::initInference(AffinityPolicy affinityPolicy)
{
    if (this->inferenceParameter.size() == 0) {
        UNI_DEBUG_LOG("node %s has no inference\n", this->nodeParameter.name().c_str());
        return;
    }
    std::string modelPath = this->inferenceParameter[0];
    const char *algorithmMapPath = "./";
    if (this->inferenceParameter.size() > 1) {
        algorithmMapPath = this->inferenceParameter[1].c_str();
    }
    UNI_DEBUG_LOG("node %s init inference engine(precision:%d affinity:%d algorithm:%s) from %s\n",
        this->nodeParameter.name().c_str(), this->precision, affinityPolicy, algorithmMapPath,
        modelPath.c_str());
    ModelSpec ms;
    CHECK_STATUS(deserialize_model_from_file(modelPath.c_str(), &ms));
    CNN cnn(affinityPolicy, precision, ms.model_name);
    cnn.sort_operators_sequential(&ms);
    cnn.initialize_ops(&ms);
    cnn.loadAlgorithmMap(algorithmMapPath);
    std::map<std::string, TensorDesc> inputDescMap = extractInputDims(&ms);
    cnn.ready(inputDescMap);
    CHECK_STATUS(cnn.mark_input_output());
    cnn.saveAlgorithmMapToFile(algorithmMapPath);
    CHECK_STATUS(mt_destroy_model(&ms));
    this->boltModel = cnn;
    UNI_DEBUG_LOG("node %s init inference engine end\n", this->nodeParameter.name().c_str());
}

EE Node::ready()
{
    UNI_DEBUG_LOG("node %s ready begin\n", this->nodeParameter.name().c_str());
    std::string preprocessFunctionName =
        (this->preprocessParameter.size() > 0) ? this->preprocessParameter[0] : "NULL";
    std::string postprocessFunctionName =
        (this->postprocessParameter.size() > 0) ? this->postprocessParameter[0] : "NULL";
    this->preprocessFunction = flowGetFunctionByName(preprocessFunctionName);
    this->postprocessFunction = flowGetFunctionByName(postprocessFunctionName);
    UNI_DEBUG_LOG("node %s ready end\n", this->nodeParameter.name().c_str());
    return SUCCESS;
}

void Node::setInput(std::map<std::string, std::shared_ptr<Tensor>> inputs)
{
    this->inputs = inputs;
}

void Node::setOutput(std::map<std::string, std::shared_ptr<Tensor>> outputs)
{
    this->outputs = outputs;
}

void Node::setRuntime(int cpuId, Arch arch)
{
    UNI_DEBUG_LOG("node %s setRuntime(core:%d arch:%d)begin\n", this->nodeParameter.name().c_str(),
        cpuId, arch);
    if (this->inferenceParameter[0] != std::string("NULL") && cpuId >= 0) {
        this->boltModel.set_runtime_device(cpuId, arch);
    } else {
        UNI_DEBUG_LOG("currently not support to setRuntime for no inference node\n");
    }
    UNI_DEBUG_LOG("node %s setRuntime end\n", this->nodeParameter.name().c_str());
}

EE Node::run()
{
    std::string preprocessFunctionName =
        (this->preprocessParameter.size() > 0) ? this->preprocessParameter[0] : "NULL";
    std::string postprocessFunctionName =
        (this->postprocessParameter.size() > 0) ? this->postprocessParameter[0] : "NULL";
    std::string modelPath = (this->inferenceParameter.size() > 0) ? this->inferenceParameter[0]
                                                                  : "NULL";
    UNI_DEBUG_LOG("node %s run begin, preprocess use %s begin\n",
        this->nodeParameter.name().c_str(), preprocessFunctionName.c_str());

    // pre process part
    std::map<std::string, std::shared_ptr<Tensor>> preprocessOutputs = this->inputs;
    if (preprocessFunction == NULL) {
        UNI_DEBUG_LOG("node %s use default preprocess function(output is set to input)\n",
            this->nodeParameter.name().c_str());
    } else {
        if (modelPath != std::string("NULL")) {
            for (auto &iter : this->boltModel.get_input()) {
                preprocessOutputs[iter.first] = iter.second;
            }
        }
        for (auto &iter : this->outputs) {
            preprocessOutputs[iter.first] = iter.second;
        }
        preprocessFunction(
            this->inputs, this->tmpTensor, preprocessOutputs, this->preprocessParameter);
    }
    UNI_DEBUG_LOG("node %s preprocess end, inference use %s begin\n",
        this->nodeParameter.name().c_str(), this->inferenceParameter[0].c_str());

    std::map<std::string, std::shared_ptr<Tensor>> postprocessInputs = preprocessOutputs;
    if (postprocessFunction == NULL) {
        postprocessInputs = this->outputs;
    } else {
        for (auto &iter : this->boltModel.get_output()) {
            postprocessInputs[iter.first] = iter.second;
        }
    }
    // inference part
    if (modelPath != std::string("NULL")) {
        std::map<std::string, TensorDesc> inputDescs = this->boltModel.get_input_desc();
        for (auto &iter : inputDescs) {
            iter.second = preprocessOutputs[iter.first]->get_desc();
        }
        this->boltModel.reready(inputDescs);
        std::map<std::string, std::shared_ptr<U8>> inputs;
        for (auto &iter : inputDescs) {
            inputs[iter.first] =
                ((CpuMemory *)preprocessOutputs[iter.first]->get_memory())->get_shared_ptr();
        }
        this->boltModel.set_input_by_assign(inputs);
        double timeStart = ut_time_ms();
        this->boltModel.run();
        double timeEnd = ut_time_ms();
        UNI_PROFILE_INFO(this->nodeParameter.name().c_str(), "run", timeStart * 1000,
            (timeEnd - timeStart) * 1000);
        std::map<std::string, std::shared_ptr<Tensor>> inferenceResult =
            this->boltModel.get_output();
        for (auto &iter : inferenceResult) {
            std::string name = iter.first;
            if (postprocessInputs.find(name) != postprocessInputs.end()) {
                TensorDesc desc = inferenceResult[name]->get_desc();
                postprocessInputs[name]->resize(desc);
                void *src = ((CpuMemory *)inferenceResult[name]->get_memory())->get_ptr();
                void *dst = ((CpuMemory *)postprocessInputs[name]->get_memory())->get_ptr();
                if (src != dst) {
                    memcpy(dst, src, tensorNumBytes(desc));
                }
            }
        }
    } else {
        UNI_DEBUG_LOG("node %s use default inference function(output is set to input)\n",
            this->nodeParameter.name().c_str());
    }
    UNI_DEBUG_LOG("node %s inference end, postprocess use %s begin\n",
        this->nodeParameter.name().c_str(), postprocessFunctionName.c_str());

    // post process part
    if (this->postprocessFunction != NULL) {
        this->postprocessFunction(
            postprocessInputs, this->tmpTensor, this->outputs, this->postprocessParameter);
    } else {
        UNI_DEBUG_LOG("node %s use default postprocess function(output is set to input)\n",
            this->nodeParameter.name().c_str());
    }
    UNI_DEBUG_LOG("node %s postprocess end, run end\n", this->nodeParameter.name().c_str());
    return SUCCESS;
}
