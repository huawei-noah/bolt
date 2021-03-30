// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef UNI_INCLUDE_GRAPH_H_
#define UNI_INCLUDE_GRAPH_H_

#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <set>

#ifdef _USE_XCODE
#include "coded_stream.h"
#include "zero_copy_stream_impl.h"
#include "text_format.h"
#include "message.h"
#else
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#endif

#include "error.h"
#include "tensor_desc.h"
#include "thread_affinity.h"

template <class GraphParameter, class ComputeNode, class DataTensor>
class Graph {
public:
    Graph()
    {}

    ~Graph()
    {}

    Graph clone()
    {
        UNI_DEBUG_LOG("graph %s clone begin\n", this->name.c_str());
        Graph graph = *this;
        for (unsigned int i = 0; i < this->nodes.size(); i++) {
            graph.nodes[i] = this->nodes[i].clone();
        }
        CHECK_STATUS(graph.manageDataTensors());
        CHECK_STATUS(graph.manageTmpBuffer());
        UNI_DEBUG_LOG("graph %s clone end\n", this->name.c_str());
        return graph;
    }

    void init(std::string graphPath)
    {
        UNI_DEBUG_LOG("load and build graph from %s begin\n", graphPath.c_str());
        GraphParameter graphParameter;
        CHECK_REQUIREMENT(load(graphPath, (google::protobuf::Message *)(&graphParameter)));
        this->name = graphParameter.name();

        for (int i = 0; i < graphParameter.output_size(); i++) {
            this->outputs.insert(graphParameter.output(i));
        }
        for (int i = 0, index = 0; i < graphParameter.node_size(); i++) {
            ComputeNode node;
            auto nodeParameter = graphParameter.node(i);
            node.setNodeParameter(nodeParameter);
            if (nodeParameter.type() == std::string("Input")) {
                DataTensor *tensor = new DataTensor();
                tensor->resize(extractInputTensorDescFromNode(node));
                CHECK_REQUIREMENT(nodeParameter.output_size() == 1);
                this->tensors[nodeParameter.output(0)] = std::shared_ptr<DataTensor>(tensor);
                continue;
            }

            this->nodes.push_back(node);
            index++;
        }
        UNI_DEBUG_LOG("load and build graph from %s end\n", graphPath.c_str());
    }

    EE ready(DataType precision, AffinityPolicy affinityPolicy, int gpuId)
    {
        UNI_DEBUG_LOG("graph %s ready begin\n", this->name.c_str());
        CHECK_STATUS(managePrecision(precision));
        if (gpuId >= 0) {
            affinityPolicy = AFFINITY_GPU;
        }
        CHECK_STATUS(initInference(affinityPolicy));
        CHECK_STATUS(manageDataTensors());
        CHECK_STATUS(manageTmpBuffer());
        for (unsigned int i = 0; i < this->nodes.size(); i++) {
            this->nodes[i].ready();
        }
        UNI_DEBUG_LOG("graph %s ready end\n", this->name.c_str());
        return SUCCESS;
    }

    EE setRuntime(int cpuId, Arch arch)
    {
        UNI_DEBUG_LOG(
            "graph %s setRuntime(core:%d arch:%d) begin\n", this->name.c_str(), cpuId, arch);
        for (unsigned int i = 0; i < this->nodes.size(); i++) {
            this->nodes[i].setRuntime(cpuId, arch);
        }
        UNI_DEBUG_LOG("graph %s setRuntime end\n", this->name.c_str());
        return SUCCESS;
    }

    EE run(std::map<std::string, std::shared_ptr<DataTensor>> tensors)
    {
        UNI_DEBUG_LOG("graph %s run begin\n", this->name.c_str());
        CHECK_STATUS(setData(tensors));
        for (unsigned int i = 0; i < this->nodes.size(); i++) {
            this->nodes[i].run();
        }
        UNI_DEBUG_LOG("graph %s run end\n", this->name.c_str());
        return SUCCESS;
    }

private:
    std::string name;
    std::vector<ComputeNode> nodes;
    std::map<std::string, std::shared_ptr<DataTensor>> tensors;
    std::shared_ptr<DataTensor> tmpDataTensor;
    std::set<std::string> outputs;

    bool load(std::string graphPath, google::protobuf::Message *message)
    {
        std::ifstream fileStream(graphPath, std::ifstream::in);
        bool ret = false;
        if (fileStream.is_open()) {
            google::protobuf::io::IstreamInputStream input(&fileStream);
            ret = google::protobuf::TextFormat::Parse(&input, message);
            fileStream.close();
        } else {
            UNI_ERROR_LOG("can not load graph from %s\n", graphPath.c_str());
        }
        return ret;
    }

    TensorDesc extractInputTensorDescFromNode(ComputeNode node)
    {
        auto nodeParameter = node.getNodeParameter();
        std::map<std::string, DataType> types = {{"FLOAT32", DT_F32}, {"FLOAT16", DT_F16},
            {"UINT32", DT_U32}, {"INT8", DT_I8}, {"UINT8", DT_U8}};
        std::map<std::string, DataFormat> formats = {
            {"NCHW", DF_NCHW}, {"NCHWC8", DF_NCHWC8}, {"MTK", DF_MTK}, {"NORMAL", DF_NORMAL}};
        TensorDesc desc;
        if (types.find(nodeParameter.input_type()) != types.end()) {
            desc.dt = types[nodeParameter.input_type()];
        } else {
            UNI_ERROR_LOG(
                "graph unsupported input data type %s\n", nodeParameter.input_type().c_str());
        }
        if (formats.find(nodeParameter.input_format()) != formats.end()) {
            desc.df = formats[nodeParameter.input_format()];
        } else {
            UNI_ERROR_LOG(
                "graph unsupported input data format %s\n", nodeParameter.input_format().c_str());
        }
        desc.nDims = nodeParameter.input_dim_size();
        for (unsigned int i = 0; i < desc.nDims; i++) {
            desc.dims[i] = nodeParameter.input_dim(desc.nDims - 1 - i);
        }
        return desc;
    }

    EE inferOutputSize()
    {
        UNI_DEBUG_LOG("graph %s infer output size begin\n", this->name.c_str());
        for (unsigned int i = 0; i < this->nodes.size(); i++) {
            CHECK_STATUS(this->nodes[i].inferOutputSize());
        }
        UNI_DEBUG_LOG("graph %s infer output size end\n", this->name.c_str());
        return SUCCESS;
    }

    EE setNodeInputOutput()
    {
        UNI_DEBUG_LOG("graph %s set node input and output begin\n", this->name.c_str());
        for (unsigned int i = 0; i < this->nodes.size(); i++) {
            auto nodeParameter = this->nodes[i].getNodeParameter();
            std::map<std::string, std::shared_ptr<DataTensor>> nodeInputs, nodeOutputs;
            for (int j = 0; j < nodeParameter.input_size(); j++) {
                std::string nodeInputName = nodeParameter.input(j);
                nodeInputs[nodeInputName] = tensors[nodeInputName];
            }
            this->nodes[i].setInput(nodeInputs);

            for (int j = 0; j < nodeParameter.output_size(); j++) {
                std::string nodeOutputName = nodeParameter.output(j);
                nodeOutputs[nodeOutputName] = tensors[nodeOutputName];
            }
            this->nodes[i].setOutput(nodeOutputs);
        }
        CHECK_STATUS(inferOutputSize());
        UNI_DEBUG_LOG("graph %s set node input and output end\n", this->name.c_str());
        return SUCCESS;
    }

    EE manageDataTensors()
    {
        UNI_DEBUG_LOG("graph %s manage tensors begin\n", this->name.c_str());
        for (unsigned int i = 0; i < this->nodes.size(); i++) {
            auto nodeParameter = this->nodes[i].getNodeParameter();
            for (int j = 0; j < nodeParameter.output_size(); j++) {
                DataTensor *tensor = new DataTensor();
                std::string nodeOutputName = nodeParameter.output(j);
                this->tensors[nodeOutputName] = std::shared_ptr<DataTensor>(tensor);
            }
        }
        CHECK_STATUS(setNodeInputOutput());
        for (auto tensor : this->tensors) {
            if (this->outputs.find(tensor.first) == this->outputs.end()) {
                tensor.second->alloc();
            }
        }
        UNI_DEBUG_LOG("graph %s manage tensors end\n", this->name.c_str());
        return SUCCESS;
    }

    EE managePrecision(DataType dataType)
    {
        UNI_DEBUG_LOG("graph %s manage precision(%d) begin\n", this->name.c_str(), dataType);
        for (unsigned int i = 0; i < this->nodes.size(); i++) {
            this->nodes[i].setPrecision(dataType);
        }
        UNI_DEBUG_LOG("graph %s manage precision end\n", this->name.c_str());
        return SUCCESS;
    }

    EE initInference(AffinityPolicy affinityPolicy)
    {
        UNI_DEBUG_LOG("graph %s init inference(%d) begin\n", this->name.c_str(), affinityPolicy);
        for (unsigned int i = 0; i < this->nodes.size(); i++) {
            this->nodes[i].initInference(affinityPolicy);
        }
        UNI_DEBUG_LOG("graph %s init inference end\n", this->name.c_str());
        return SUCCESS;
    }

    unsigned int inferTmpBufferSize()
    {
        UNI_DEBUG_LOG("graph %s infer tmp buffer size begin\n", this->name.c_str());
        unsigned int maxTmpBufferSize = 0;
        for (unsigned int i = 0; i < this->nodes.size(); i++) {
            unsigned int tmpBufferSize = this->nodes[i].getTmpBufferSize();
            if (tmpBufferSize > maxTmpBufferSize) {
                maxTmpBufferSize = tmpBufferSize;
            }
        }
        UNI_DEBUG_LOG("graph %s infer tmp buffer size end\n", this->name.c_str());
        return maxTmpBufferSize;
    }

    EE manageTmpBuffer()
    {
        UNI_DEBUG_LOG("graph %s manage tmp buffer begin\n", this->name.c_str());
        unsigned int maxTmpBufferSize = inferTmpBufferSize();
        this->tmpDataTensor = std::shared_ptr<DataTensor>(new DataTensor());
        this->tmpDataTensor->resize(tensor1d(DT_U8, maxTmpBufferSize));
        for (unsigned int i = 0; i < this->nodes.size(); i++) {
            this->nodes[i].setTmpBuffer(this->tmpDataTensor);
        }
        UNI_DEBUG_LOG("graph %s manage tmp buffer end\n", this->name.c_str());
        return SUCCESS;
    }

    EE setData(std::map<std::string, std::shared_ptr<DataTensor>> tensors)
    {
        UNI_DEBUG_LOG("graph %s set data from upper begin\n", this->name.c_str());
        for (auto tensor : tensors) {
            if (this->tensors.find(tensor.first) != this->tensors.end()) {
                this->tensors[tensor.first] = tensor.second;
            } else {
                UNI_ERROR_LOG("graph %s can not find %s to set as input or output\n",
                    this->name.c_str(), tensor.first.c_str());
            }
        }
        CHECK_STATUS(setNodeInputOutput());
        UNI_DEBUG_LOG("graph %s set data from upper end\n", this->name.c_str());
        return SUCCESS;
    }
};
#endif  // UNI_INCLUDE_GRAPH_H_
