// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MINDSPOREADAPTEE
#define _H_MINDSPOREADAPTEE
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "mind_ir.pb.h"
#include "model_adaptee.h"

class MindsporeAdaptee : public ModelAdaptee {
public:
    MindsporeAdaptee()
    {}

    ~MindsporeAdaptee()
    {
        google::protobuf::ShutdownProtobufLibrary();
    }

protected:
    OperatorType convert_ms_type(const std::string &msNodeType)
    {
        if (msNodeType == "Conv2D") {
            return OT_Conv;
        } else if (msNodeType == "ReLU") {
            return OT_Relu;
        } else if (msNodeType == "MaxPool") {
            return OT_Pooling;
        } else if (msNodeType == "Reshape") {
            return OT_Reshape;
        } else if (msNodeType == "MatMul") {
            return OT_FC;
        } else if (msNodeType == "BiasAdd") {
            return OT_Scale;
        } else {
            UNI_ERROR_LOG("Unsupport this mindspore op");
        }
        return OT_None;
    }

    EE read_from_mindir_file(const char *path, google::protobuf::Message *message)
    {
        std::ifstream fs(path, std::ifstream::in | std::ifstream::binary);
        if (!fs.is_open()) {
            UNI_ERROR_LOG("Can not open mindir model file!\n");
        }

        google::protobuf::io::IstreamInputStream input(&fs);
        google::protobuf::io::CodedInputStream codedstr(&input);

        codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

        bool ret = message->ParseFromCodedStream(&codedstr);
        fs.close();
        if (!ret) {
            return NOT_SUPPORTED;
        }

        return SUCCESS;
    }

    EE parse_file(std::string modelDirectory, std::string modelFileName) override
    {
        std::string modelPath = modelDirectory + "/" + modelFileName + ".mindir";
        CHECK_STATUS(read_from_mindir_file(
            modelPath.c_str(), (google::protobuf::Message *)(&mindsporeModel)));
        mindsporeGraph = mindsporeModel.graph();
        return SUCCESS;
    }

    TensorDesc gen_desc_from_mindirTp(mind_ir::TensorProto tp)
    {
        TensorDesc desc;
        desc.dt = DT_F32;
        desc.nDims = tp.dims_size();
        if (desc.nDims == 4) {
            desc.df = DF_NCHW;
        } else if (desc.nDims == 3) {
            desc.df = DF_MTK;
        } else if (desc.nDims == 2) {
            desc.df = DF_NORMAL;
        } else {
            UNI_ERROR_LOG("Do not support this input, please check the model again.");
        }
        // reversed order to assign the dims
        for (int i = 0; i < tp.dims_size(); i++) {
            desc.dims[i] = tp.dims(tp.dims_size() - 1 - i);
        }
        return desc;
    }

    EE adapt_operators(ModelSpec *ms) override
    {
        str_copy(ms->model_name, mindsporeGraph.name().c_str(), mindsporeGraph.name().length());
        ms->dt = DT_F32;

        ms->num_inputs =
            mindsporeGraph.input_size();  // if some input belongs to const, need to filter
        ms->input_names = (I8 **)mt_malloc(ms->num_inputs * sizeof(I8 *));
        ms->input_dims = (TensorDesc *)mt_malloc(sizeof(TensorDesc) * ms->num_inputs);
        for (int i = 0; i < mindsporeGraph.input_size(); i++) {
            mind_ir::ValueInfoProto curInput = mindsporeGraph.input(i);
            std::string inputName = curInput.name();
            ms->input_names[i] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            str_copy(ms->input_names[i], inputName.c_str(), inputName.length());
            if (curInput.tensor_size() != 1) {
                UNI_ERROR_LOG("input extraction of mindsporeGraph failed!");
            } else {
                mind_ir::TensorProto curTensor = curInput.tensor(0);
                ms->input_dims[i] = gen_desc_from_mindirTp(curTensor);
            }
        }

        int dependIndex = -1;
        for (int i = 0; i < mindsporeGraph.node_size(); i++) {
            if (mindsporeGraph.node(i).op_type() == "Depend") {
                dependIndex = i;
                break;
            }
        }
        ms->num_outputs = mindsporeGraph.node(dependIndex).input_size() - 1;
        ms->output_names = (I8 **)mt_malloc(ms->num_outputs * sizeof(I8 *));
        for (int i = 0; i < ms->num_outputs; i++) {
            std::string curName = mindsporeGraph.node(dependIndex).input(i);
            ms->output_names[i] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            str_copy(ms->output_names[i], curName.c_str(), curName.length());
        }

        std::vector<OperatorSpec> operatorSpecVec;
        for (int i = 0; i < mindsporeGraph.node_size(); i++) {  // not all the nodes are valid
            mindsporeNode = mindsporeGraph.node(i);

            // mark constant info
            if (mindsporeNode.op_type() == "Constant") {
                std::string opOutput = mindsporeNode.output(0);
                constantIndex[opOutput] = i;
            }
            if (mindsporeNode.op_type() == "Load") {
                loadIndex[mindsporeNode.output(0)] = i;
            }
            if (weightsOperators.find(mindsporeNode.op_type()) != weightsOperators.end()) {
                weightsIndex.push_back(i);
            }

            if (deprecatedOperators.find(mindsporeNode.op_type()) != deprecatedOperators.end()) {
                UNI_DEBUG_LOG("[Deprecated]\n");
            } else {
                OperatorType opType = convert_ms_type(mindsporeNode.op_type());
                int curInputSize = mindsporeNode.input_size();
                if (weightsOperators.find(mindsporeNode.op_type()) != weightsOperators.end()) {
                    curInputSize = weightsOperators[mindsporeNode.op_type()];
                }
                if (mindsporeNode.op_type() == "Reshape") {
                    curInputSize = 1;
                }
                OperatorSpec os = mt_create_operator(mindsporeNode.name().c_str(), opType,
                    curInputSize, mindsporeNode.output_size());
                for (int j = 0; j < curInputSize; j++) {
                    str_copy(os.input_tensors_name[j], mindsporeNode.input(j).c_str(),
                        mindsporeNode.input(j).length());
                }
                for (int j = 0; j < mindsporeNode.output_size(); j++) {
                    str_copy(os.output_tensors_name[j], mindsporeNode.output(j).c_str(),
                        mindsporeNode.output(j).length());
                }
                adapt_operator(opType, &(os.ps));

                operatorSpecVec.push_back(os);
            }
        }

        ms->num_operator_specs = operatorSpecVec.size();
        ms->ops = (OperatorSpec *)mt_malloc(sizeof(OperatorSpec) * ms->num_operator_specs);
        UNI_MEMCPY(ms->ops, operatorSpecVec.data(), sizeof(OperatorSpec) * operatorSpecVec.size());
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            ms->ops[i].tensor_positions = nullptr;
            ms->ops[i].num_quant_feature = 0;
            ms->ops[i].feature_scale = nullptr;
        }
        return SUCCESS;
    }

    EE adapt_weights(ModelSpec *ms) override
    {
        std::map<std::string, int> paramIndex;
        for (int i = 0; i < mindsporeGraph.parameter_size(); i++) {
            paramIndex[mindsporeGraph.parameter(i).name()] = i;  // swap space for speed
        }

        std::vector<WeightSpec> ws;
        for (unsigned int i = 0; i < weightsIndex.size(); i++) {
            mind_ir::NodeProto curNode = mindsporeGraph.node(weightsIndex[i]);
            int curLoadOpIndex = loadIndex[curNode.input(1)];
            mind_ir::NodeProto curLoadNode = mindsporeGraph.node(curLoadOpIndex);
            if (paramIndex.find(curLoadNode.input(0)) == paramIndex.end()) {
                UNI_ERROR_LOG("Do not find valid param.\n");
            }
            int curParamIndex = paramIndex[curLoadNode.input(0)];
            mind_ir::TensorProto curTensorProto = mindsporeGraph.parameter(curParamIndex);
            int totalDim = 1;
            for (int j = 0; j < curTensorProto.dims_size(); j++) {
                totalDim *= curTensorProto.dims(j);
            }

            WeightSpec weightSpec;
            if (curNode.op_type() == "Conv2D" || curNode.op_type() == "MatMul") {
                weightSpec = mt_create_weight(
                    curNode.name().c_str(), DT_F32, totalDim * sizeof(float), 0, 0);

                if (curTensorProto.has_raw_data()) {
                    const std::string &rawData = curTensorProto.raw_data();
                    float *elementPtr = (float *)(rawData.data());
                    UNI_MEMCPY(weightSpec.weight, elementPtr, totalDim * sizeof(float));
                } else {
                    UNI_ERROR_LOG("Do not support the weight extraction.\n");
                }
            } else if (curNode.op_type() == "BiasAdd") {
                weightSpec = mt_create_weight(
                    curNode.name().c_str(), DT_F32, 0, totalDim * sizeof(float), 0);
                if (curTensorProto.has_raw_data()) {
                    const std::string &rawData = curTensorProto.raw_data();
                    float *elementPtr = (float *)(rawData.data());
                    UNI_MEMCPY(weightSpec.vec, elementPtr, totalDim * sizeof(float));
                } else {
                    UNI_ERROR_LOG("Do not support the vec extraction.\n");
                }
            }
            ws.push_back(weightSpec);
        }

        ms->num_weight_specs = ws.size();
        ms->ws = (WeightSpec *)mt_malloc(sizeof(WeightSpec) * ms->num_weight_specs);
        UNI_MEMCPY(ms->ws, ws.data(), sizeof(WeightSpec) * ws.size());
        return SUCCESS;
    }

    ParameterSpec adapt_Conv() override
    {
        ParameterSpec ps;
        ConvolutionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));

        p.kernel_t = 1;
        p.kernel_h = 1;
        p.kernel_w = 1;
        p.dilatedRate_t = 1;
        p.dilatedRate_h = 1;
        p.dilatedRate_w = 1;
        p.stride_t = 1;
        p.stride_h = 1;
        p.stride_w = 1;

        for (int i = 0; i < mindsporeNode.attribute_size(); i++) {
            mind_ir::AttributeProto curAttribute = mindsporeNode.attribute(i);
            if (curAttribute.name() == "kernel_size") {
                if (curAttribute.ints_size() != 2) {
                    UNI_ERROR_LOG("Do not support this conv kernel size\n");
                } else {
                    p.kernel_h = curAttribute.ints(0);
                    p.kernel_w = curAttribute.ints(1);
                }
            } else if (curAttribute.name() == "out_channel") {
                p.num_outputs = curAttribute.i();
                p.num_outputs_origin = p.num_outputs;
            } else if (curAttribute.name() == "pad" || curAttribute.name() == "pad_list") {
                if (curAttribute.ints_size() != 4) {  // TODO specify the order
                    p.pad_top = curAttribute.ints(0);
                    p.pad_left = curAttribute.ints(1);
                    p.pad_bottom = curAttribute.ints(2);
                    p.pad_right = curAttribute.ints(3);
                }
            } else if (curAttribute.name() == "groups" || curAttribute.name() == "group") {
                p.group = curAttribute.i();
            } else if (curAttribute.name() == "dilation") {
                p.dilatedRate_h = curAttribute.ints(0);
                p.dilatedRate_w = curAttribute.ints(1);
            }
        }

        if (p.group == p.num_outputs && p.group != 1) {
            p.convolution_type = CONVOLUTION_DEPTHWISE;
        } else {
            p.convolution_type = CONVOLUTION_POINTWISE;
        }

        p.dw_activation_type = ACTIVATION_NULL;
        p.pw_activation_type = ACTIVATION_NULL;
        ps.conv_spec = p;
        return ps;
    }

    ParameterSpec adapt_Relu() override
    {
        ParameterSpec ps;
        ReLUParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.neg_slope = 0.0;
        ps.relu_spec = p;
        return ps;
    }

    ParameterSpec adapt_Pooling() override
    {
        ParameterSpec ps;
        PoolingParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));

        if (mindsporeNode.op_type() == "MaxPool") {
            p.mode = POOLING_MAX;
        } else {
            p.mode = POOLING_MEAN;
        }
        p.round_mode = ROUND_FLOOR;

        for (int i = 0; i < mindsporeNode.attribute_size(); i++) {
            mind_ir::AttributeProto curAttribute = mindsporeNode.attribute(i);
            if (curAttribute.name() == "kernel_size") {
                p.kernel_t = 1;
                if (curAttribute.ints_size() == 2) {
                    p.kernel_h = curAttribute.ints(0);
                    p.kernel_w = curAttribute.ints(1);
                } else if (curAttribute.ints_size() == 4) {
                    p.kernel_h = curAttribute.ints(2);
                    p.kernel_w = curAttribute.ints(3);
                } else {
                    UNI_ERROR_LOG("Do not support this pooling kernel size.\n");
                }
            } else if (curAttribute.name() == "strides") {
                p.stride_t = 1;
                if (curAttribute.ints_size() == 2) {
                    p.stride_h = curAttribute.ints(0);
                    p.stride_w = curAttribute.ints(1);
                } else if (curAttribute.ints_size() == 4) {
                    p.stride_h = curAttribute.ints(2);
                    p.stride_w = curAttribute.ints(3);
                } else {
                    UNI_ERROR_LOG("Do not support this pooling stride size.\n");
                }
            }  // TODO pad
        }

        ps.pooling_spec = p;
        return ps;
    }

    ParameterSpec adapt_Reshape() override
    {
        // Locate the constant
        ParameterSpec ps;
        ReshapeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));

        if (constantIndex.find(mindsporeNode.input(1)) == constantIndex.end()) {
            UNI_ERROR_LOG("Do not find shape info\n");
        }
        mind_ir::NodeProto constantNode = mindsporeGraph.node(constantIndex[mindsporeNode.input(1)]);
        p.num_shape = constantNode.attribute(0).ints_size();
        for (int i = 0; i < constantNode.attribute(0).ints_size(); i++) {
            p.shape[i] = constantNode.attribute(0).ints(i);
        }
        p.axis = 0;
        p.num_axes = -1;
        ps.reshape_spec = p;
        return ps;
    }

    ParameterSpec adapt_Fc() override
    {
        ParameterSpec ps;
        FullyConnectedParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.num_outputs = -1;  // unsigned int, need to specify correctly
        mind_ir::AttributeProto fir_att = mindsporeNode.attribute(0);
        mind_ir::TensorProto tp = fir_att.tensors(0);
        p.num_outputs = tp.dims(1);
        p.num_slices = 1;
        p.slice_point[0] = p.num_outputs;
        ps.fc_spec = p;
        return ps;
    }

    ParameterSpec adapt_Scale() override
    {
        ParameterSpec ps;
        ScaleParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = -1;  // TODO
        ps.scale_spec = p;
        return ps;
    }

public:
    std::map<std::string, int> deprecatedOperators{
        {"Depend", 1}, {"UpdateState", 1}, {"MakeTuple", 1}, {"Load", 1}, {"Constant", 1}};
    std::map<std::string, int> weightsOperators{{"Conv2D", 1}, {"MatMul", 1}, {"BiasAdd", 1}};

private:
    mind_ir::ModelProto mindsporeModel;
    mind_ir::GraphProto mindsporeGraph;
    mind_ir::NodeProto mindsporeNode;

    std::map<std::string, mind_ir::TensorProto> parameters;
    std::map<std::string, int> constantIndex;
    std::map<std::string, int> loadIndex;
    std::vector<int> weightsIndex;
};
#endif
