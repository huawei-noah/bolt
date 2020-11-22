// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ONNXADAPTEE
#define _H_ONNXADAPTEE

#include <string>
#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "onnx.pb.h"

#include "converter.h"
#include "model_tools.h"
#include "model_adaptee.h"
#include "ut_util.h"

class OnnxAdaptee : public ModelAdaptee {
public:
    OnnxAdaptee(int removePreprocessOpNum_outside)
    {
        this->removePreprocessOpNum = removePreprocessOpNum_outside;
    }
    ~OnnxAdaptee()
    {}

protected:
    DataType get_weight_data_type(U32 weightLen, F32 *weight)
    {
        if (1 >= weightLen) {
            return DT_F32;
        }
        F32 val0 = 1;
        F32 val1 = 0;
        for (U32 i = 0; i < weightLen; i++) {
            F32 cur = weight[i];
            if (cur <= 0 && val0 <= 0 && cur != val0) {
                return DT_F32;
            }
            if (cur > 0 && val1 > 0 && cur != val1) {
                return DT_F32;
            }
            if (cur <= 0 && val0 > 0) {
                val0 = cur;
            }
            if (cur > 0 && val1 <= 0) {
                val1 = cur;
            }
        }
        if (val0 == 0) {
            return DT_BIN01;
        }
        CHECK_REQUIREMENT(0 == val0 + val1);
        return DT_BIN11;
    }

    std::vector<int> getOperatorWeightInputIndex(int weightOpIndex)
    {
        const onnx::NodeProto &weightNode = onnxGraph.node(weightOpIndex);
        std::vector<int> index;
        for (int i = 0; i < weightNode.input_size(); i++) {
            if (weights.end() != weights.find(weightNode.input(i))) {
                index.push_back(i);
            }
        }
        return index;
    }

    EE read_from_onnx_file(const char *path, google::protobuf::Message *message)
    {
        std::ifstream fs(path, std::ifstream::in | std::ifstream::binary);
        if (!fs.is_open()) {
            return NOT_FOUND;
        }

        google::protobuf::io::IstreamInputStream input(&fs);
        google::protobuf::io::CodedInputStream codedstr(&input);

        codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

        bool ret = message->ParseFromCodedStream(&codedstr);
        fs.close();

        return (ret) ? SUCCESS : NOT_SUPPORTED;
    }

    OperatorType convert_onnx_type(std::string inputType)
    {
        if (inputType == "Conv") {
            return OT_Conv;
        } else if (inputType == "BatchNormalization") {
            return OT_BatchNorm;
        } else if (inputType == "Sum" || inputType == "Add" || inputType == "Mul" ||
            inputType == "Div" || inputType == "Sub") {
            return OT_Eltwise;
        } else if (inputType == "Gemm") {
            return OT_FC;
        } else if (inputType == "AveragePool" || inputType == "MaxPool" ||
            inputType == "ReduceMean" || inputType == "GlobalAveragePool" ||
            inputType == "ReduceMax") {
            return OT_Pooling;
        } else if (inputType == "Relu" || inputType == "LeakyRelu") {
            return OT_Relu;
        } else if (inputType == "Softmax") {
            return OT_Softmax;
        } else if (inputType == "Concat") {
            return OT_Concat;
        } else if (inputType == "Pad") {
            return OT_Pad;
        } else if (inputType == "Max" || inputType == "Min" || inputType == "Clip") {
            return OT_Clip;
        } else if (inputType == "Reshape") {
            return OT_Reshape;
        } else if (inputType == "Squeeze") {
            return OT_Squeeze;
        } else if (inputType == "Transpose") {
            return OT_Transpose;
        } else if (inputType == "Gather") {
            return OT_Gather;
        } else if (inputType == "Unsqueeze") {
            return OT_Unsqueeze;
        } else if (inputType == "Resize" || inputType == "Upsample") {
            return OT_Resize;
        } else if (inputType == "Cast") {
            return OT_Cast;
        } else if (inputType == "Constant") {
            return OT_Constant;
        } else if (inputType == "MatMul") {
            return OT_MatMul;
        } else if (inputType == "Flatten") {
            return OT_Reshape;
        } else if (inputType == "ConvTranspose") {
            return OT_Deconvolution;
        } else if (inputType == "Tanh") {
            return OT_TanH;
        } else if (inputType == "LogSoftmax") {
            return OT_LogSoftmax;
        } else if (inputType == "Shape") {
            return OT_Shape;
        } else if (inputType == "Erf") {
            return OT_Erf;
        } else if (inputType == "Pow" || inputType == "Sqrt") {
            return OT_Power;
        } else if (inputType == "RNN" || inputType == "LSTM" || inputType == "GRU") {
            return OT_RNN;
        } else if (inputType == "ConstantOfShape") {
            return OT_ConstantOfShape;
        } else if (inputType == "SpaceToDepth") {
            return OT_Space2Depth;
        } else if (inputType == "DepthToSpace") {
            return OT_Depth2Space;
        } else if (inputType == "PRelu") {
            return OT_PRelu;
        } else if (inputType == "ArgMax") {
            return OT_ArgMax;
        } else if (inputType == "Tile") {
            return OT_Tile;
        } else if (inputType == "Sigmoid") {
            return OT_Sigmoid;
        } else if (inputType == "Slice") {
            return OT_Slice;
        } else if (inputType == "ReduceSum") {
            return OT_Reduction;
        } else if (inputType == "Split") {
            return OT_Slice;
        } else if (inputType == "Splice") {
            return OT_Splice;
        } else if (inputType == "Greater") {
            return OT_Greater;
        } else {
            UNI_ERROR_LOG(
                "encounter unsupported operator in onnx converter: %s\n", inputType.c_str());
        }
        return OT_None;
    }

    std::vector<int> get_node_vector_ints_attribute_by_name(
        const onnx::NodeProto &node, const char *key)
    {
        std::vector<int> result;
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = node.attribute(i);
            if (attribute.name() == key) {
                result.resize(attribute.ints_size());
                for (int j = 0; j < attribute.ints_size(); j++) {
                    result[j] = attribute.ints(j);
                }
                break;
            }
        }
        return result;
    }

    std::vector<F32> get_node_vector_float_tensor_attribute_by_name(
        const onnx::NodeProto &node, const char *key)
    {
        std::vector<F32> result;
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = node.attribute(i);
            if (attribute.name() == key) {
                CHECK_REQUIREMENT(4 == attribute.type());
                const onnx::TensorProto &tp = attribute.t();
                F32 *value;
                if (tp.has_raw_data()) {
                    const std::string &rawData = tp.raw_data();
                    value = (F32 *)(rawData.data());
                } else if (tp.data_type() == 1) {
                    value = (F32 *)(tp.float_data().data());
                } else {
                    UNI_WARNING_LOG("Constant not extracted\n");
                    return result;
                }

                result.resize(tp.dims(0));
                for (int j = 0; j < tp.dims(0); j++) {
                    result[j] = value[j];
                }
                break;
            }
        }
        return result;
    }

    int get_node_single_int_attribute_by_name(
        const onnx::NodeProto &node, const char *key, int defaultValue = 0)
    {
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = node.attribute(i);
            if (attribute.name() == key) {
                return attribute.i();
            }
        }
        return defaultValue;
    }

    std::string get_node_str_attribute_by_name(const onnx::NodeProto &node,
        const char *key,
        const std::string &defaultValue = std::string())
    {
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = node.attribute(i);
            if (attribute.name() == key) {
                return attribute.s();
            }
        }
        return defaultValue;
    }

    float get_node_float_attribute_by_name(
        const onnx::NodeProto &node, const char *key, float defaultValue = 0.f)
    {
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = node.attribute(i);
            if (attribute.name() == key) {
                return attribute.f();
            }
        }
        return defaultValue;
    }

    int get_data_size_from_tensor_proto(const onnx::TensorProto &tensorProto)
    {
        if (tensorProto.has_raw_data()) {
            const std::string &rawData = tensorProto.raw_data();
            int size = (int)rawData.size() / sizeof(float);
            return size;
        } else if (tensorProto.data_type() == 1) {
            return tensorProto.float_data_size();
        }
        return 0;
    }

    float *get_ptr_from_weight_obj(const onnx::TensorProto &tensorProto)
    {
        if (tensorProto.has_raw_data()) {
            const std::string &rawData = tensorProto.raw_data();
            float *paramPtr = (float *)(rawData.data());
            return paramPtr;
        } else if (tensorProto.data_type() == 1) {
            float *paramPtr = (float *)(tensorProto.float_data().data());
            return paramPtr;
        }
        return nullptr;
    }

    std::vector<int> get_reshapeInfo_from_tensorProto(const onnx::TensorProto &tp)
    {
        int size = 0;
        std::vector<int> shape;

        // int64
        if (tp.data_type() == 7 || tp.data_type() == 0) {
            const int64_t *shapeData = 0;
            if (tp.has_raw_data()) {
                shapeData = (const int64_t *)tp.raw_data().data();
                size = tp.raw_data().size() / 8;
            } else {
                shapeData = tp.int64_data().data();
                size = tp.int64_data_size();
            }

            for (int j = 0; j < size; j++) {
                shape.push_back(shapeData[j]);
            }
        } else if (tp.data_type() == 6) {  // int32
            const int32_t *shapeData = 0;
            if (tp.has_raw_data()) {
                shapeData = (const int32_t *)tp.raw_data().data();
                size = tp.raw_data().size() / 4;
            } else {
                shapeData = tp.int32_data().data();
                size = tp.int32_data_size();
            }

            for (int j = 0; j < size; j++) {
                shape.push_back(shapeData[j]);
            }
        } else {
            UNI_ERROR_LOG("UnSupport data type\n");
        }
        return shape;
    }

    float getSinFloat_from_tensorProto(const onnx::TensorProto &tp)
    {
        float value = 0;
        int exponentSize = get_data_size_from_tensor_proto(tp);
        if (tp.data_type() != 1 || exponentSize != 1) {
            UNI_ERROR_LOG("UnSupport this data type or the num of params exceeds 1.\n");
        } else {
            if (tp.has_raw_data()) {
                const std::string &raw_data = tp.raw_data();
                value = ((float *)raw_data.data())[0];
            } else {
                value = ((float *)tp.float_data().data())[0];
            }
        }
        return value;
    }

    EE parse_file(std::string dir, std::string mfn) override
    {
        std::string onnxSuffix = ".onnx";
        std::string onnxPath = dir + "/" + mfn + onnxSuffix;

        this->modelName = mfn;

        EE ret = read_from_onnx_file(onnxPath.c_str(), (google::protobuf::Message *)(&onnxModel));
        if (ret != SUCCESS) {
            UNI_ERROR_LOG("fail to load onnx model %s\n", onnxPath.c_str());
        }

        onnxGraph = onnxModel.graph();

        for (int i = 0; i < onnxGraph.initializer_size(); i++) {
            const onnx::TensorProto &initializer = onnxGraph.initializer(i);
            weights[initializer.name()] = initializer;
        }
        return ret;
    }

    EE adapt_operators(ModelSpec *ms) override
    {
        EE ret = SUCCESS;
        str_copy(ms->model_name, modelName.c_str(), modelName.length());
        ms->model_name[NAME_LEN - 1] = '\0';
        ms->dt = DT_F32;

        int onnxNodeCount = onnxGraph.node_size();
        int input_node_num = onnxGraph.input().size();
        int output_node_num = onnxGraph.output().size();
        if (input_node_num != 1) {
            UNI_WARNING_LOG("num of input node is not 1\n");
        }

        std::vector<std::string> exactly_input_names;
        std::vector<std::vector<int>> input_dimens;
        for (int i = 0; i < input_node_num; i++) {
            auto input_node = onnxGraph.input(i);
            std::string cur_input_name = input_node.name();
            if (weights.find(cur_input_name) != weights.end()) {
                continue;
            }
            exactly_input_names.push_back(cur_input_name);

            std::vector<int> dims_list;
            int node_dimension_size = input_node.type().tensor_type().shape().dim().size();
            if (node_dimension_size == 4) {
                // extraction for 4 dimension tensor
                int dim_0 = input_node.type().tensor_type().shape().dim(0).dim_value();
                if (dim_0 == 0) {
                    dims_list.push_back(1);
                } else {
                    dims_list.push_back(input_node.type().tensor_type().shape().dim(0).dim_value());
                }
                dims_list.push_back(input_node.type().tensor_type().shape().dim(1).dim_value());
                dims_list.push_back(input_node.type().tensor_type().shape().dim(2).dim_value());
                dims_list.push_back(input_node.type().tensor_type().shape().dim(3).dim_value());
            } else if (node_dimension_size == 3 || node_dimension_size == 2) {
                for (int j = 0; j < node_dimension_size; j++) {
                    dims_list.push_back(input_node.type().tensor_type().shape().dim(j).dim_value());
                }
            } else {
                UNI_WARNING_LOG("not support input dimension!\n");
            }
            input_dimens.push_back(dims_list);
        }

        input_node_num = exactly_input_names.size();
        ms->num_inputs = input_node_num;
        ms->input_names = (I8 **)mt_new_storage(ms->num_inputs * sizeof(I8 *));
        if (exactly_input_names.size() == 1) {
            const onnx::NodeProto &theFirstNode = onnxGraph.node(removePreprocessOpNum);
            for (int k = 0; k < theFirstNode.input_size(); k++) {
                if (weights.find(theFirstNode.input(k)) != weights.end()) {
                    continue;
                } else {
                    std::string modelInputName = theFirstNode.input(k);
                    exactly_input_names[0] = modelInputName;
                    break;
                }
            }
        }

        for (int k = 0; k < input_node_num; k++) {
            ms->input_names[k] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            str_copy(ms->input_names[k], exactly_input_names[k].c_str(),
                exactly_input_names[k].length());
        }
        ms->input_dims = (TensorDesc *)mt_new_storage(sizeof(TensorDesc) * ms->num_inputs);
        for (int i = 0; i < ms->num_inputs; i++) {
            int curInputDimSize = input_dimens[i].size();
            TensorDesc input_desc;
            if (curInputDimSize == 4) {
                input_desc = tensor4d(DT_F32, input_dimens[i][0], input_dimens[i][1],
                    input_dimens[i][2], input_dimens[i][3]);
            } else if (curInputDimSize == 3) {
                input_desc = ms->input_dims[i] = tensor3df(
                    DT_F32, DF_MTK, input_dimens[i][0], input_dimens[i][1], input_dimens[i][2]);
            } else if (curInputDimSize == 2) {
                input_desc = ms->input_dims[i] =
                    tensor2df(DT_F32, DF_NORMAL, input_dimens[i][0], input_dimens[i][1]);
            } else {
                UNI_WARNING_LOG("not support input dimension!\n");
            }
            ms->input_dims[i] = input_desc;
        }

        ms->num_outputs = output_node_num;
        ms->output_names = (I8 **)mt_new_storage(ms->num_outputs * sizeof(I8 *));
        for (int k = 0; k < output_node_num; k++) {
            ms->output_names[k] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            str_copy(ms->output_names[k], onnxGraph.output(k).name().c_str(),
                onnxGraph.output(k).name().length());
        }

        int bnOpNum = 0;
        int constantOpNum = 0;
        for (int i = 0; i < onnxNodeCount; i++) {
            const onnx::NodeProto &tmpNode = onnxGraph.node(i);
            if (tmpNode.op_type() == "BatchNormalization") {
                bnOpNum++;
            } else if (tmpNode.op_type() == "Constant") {
                if (i >= removePreprocessOpNum) {
                    constantOpNum++;
                }
            }
        }

        ms->num_operator_specs = onnxNodeCount + bnOpNum - constantOpNum -
            removePreprocessOpNum;  // appending space for scale op
        OperatorSpec *opsPtr =
            (OperatorSpec *)mt_new_storage(sizeof(OperatorSpec) * ms->num_operator_specs);
        ms->ops = opsPtr;
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            ms->ops[i].tensor_positions = nullptr;
            ms->ops[i].num_quant_feature = 0;
            ms->ops[i].feature_scale = nullptr;
        }

        // Some models transformed from TF store weight and bias as Constant OP
        int numUnseenConstants = 0;
        nodeIndex = 0;
        for (int i = 0; i < removePreprocessOpNum; i++) {
            this->node = onnxGraph.node(nodeIndex);
            this->op = node.op_type();
            if (op == "Constant") {
                handle_Constant();
                numUnseenConstants++;
            }
            nodeIndex++;
        }
        if (0 != numUnseenConstants) {
            UNI_INFO_LOG("%d OPs are skipped, and %d of them are Constant OP.\n",
                removePreprocessOpNum, numUnseenConstants);
        }

        nodeIndex = removePreprocessOpNum;
        int opIndex = 0;
        for (int i = removePreprocessOpNum; i < onnxNodeCount; i++) {
            this->node = onnxGraph.node(nodeIndex);
            this->op = node.op_type();
            if (op == "Constant") {
                handle_Constant();
                nodeIndex++;
                continue;
            }
            std::string opName = node.name();
            if (opName.empty()) {
                opName = node.output(0);
            }
            int opInputNum = (int)node.input_size();
            opFinalInputNum = opInputNum;
            std::vector<std::string> inputNames;
            std::vector<std::string> op_weight_objs;
            for (int j = 0; j < opInputNum; j++) {
                const std::string &input_name = node.input(j);
                if (weights.find(input_name) != weights.end()) {
                    opFinalInputNum--;
                    op_weight_objs.push_back(input_name);
                } else {
                    inputNames.push_back(input_name);
                    if (op == "Max" || op == "Min") {
                        opFinalInputNum = 1;
                        break;
                    }
                }
            }
            int opOutputNum = (int)node.output_size();
            std::vector<std::string> outputNames;
            for (int j = 0; j < opOutputNum; j++) {
                const std::string &output_name = node.output(j);
                outputNames.push_back(output_name);
            }

            str_copy(opsPtr[opIndex].name, opName.c_str(), opName.length());
            OperatorType opType = convert_onnx_type(op);

            // op type correction
            if (op == "MatMul" && opFinalInputNum == 1) {
                opType = OT_FC;
            }

            opsPtr[opIndex].type = opType;
            opsPtr[opIndex].num_inputs = opFinalInputNum;
            opsPtr[opIndex].input_tensors_name =
                (I8 **)mt_new_storage(opsPtr[opIndex].num_inputs * sizeof(I8 *));
            for (U32 j = 0; j < opsPtr[opIndex].num_inputs; j++) {
                opsPtr[opIndex].input_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(opsPtr[opIndex].input_tensors_name[j], inputNames[j].c_str(),
                    inputNames[j].length());
            }
            opsPtr[opIndex].num_outputs = opOutputNum;
            opsPtr[opIndex].output_tensors_name =
                (I8 **)mt_new_storage(opsPtr[opIndex].num_outputs * sizeof(I8 *));
            for (U32 j = 0; j < opsPtr[opIndex].num_outputs; j++) {
                opsPtr[opIndex].output_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(opsPtr[opIndex].output_tensors_name[j], outputNames[j].c_str(),
                    outputNames[j].length());
            }

            if ((op == "Add" || op == "Mul" || op == "Div") && opFinalInputNum == 1) {
                weightOpIndexLists.push_back(nodeIndex);
                opsPtr[opIndex].type = OT_Scale;
                initialization_zero(&(opsPtr[opIndex].ps), sizeof(opsPtr[opIndex].ps));
                opsPtr[opIndex].ps.scale_spec.axis = 1;
            } else if (op == "Transpose" && opFinalInputNum == 0) {
                weightOpIndexLists.push_back(nodeIndex);
            } else {
                if (op == "Gather") {
                    if (weights.find(node.input(0)) != weights.end()) {
                        weightOpIndexLists.push_back(nodeIndex);
                        if (weights.find(node.input(1)) != weights.end()) {  // both provided
                            opsPtr[opIndex].type = OT_SharedWeight;
                            opType = OT_SharedWeight;
                        } else {
                            opsPtr[opIndex].type = OT_Embedding;
                            opType = OT_Embedding;
                        }
                    } else if (weights.find(node.input(1)) != weights.end()) {
                        opType = OT_Slice;
                        opsPtr[opIndex].type = OT_Slice;
                        initialization_zero(&(opsPtr[opIndex].ps), sizeof(opsPtr[opIndex].ps));
                        opsPtr[opIndex].ps.slice_spec.slice_points[0] = 1;
                        opsPtr[opIndex].ps.slice_spec.slice_size = 1;
                        opsPtr[opIndex].ps.slice_spec.axis = 1;
                        opsPtr[opIndex].num_outputs = 2;
                        free(opsPtr[opIndex].output_tensors_name[0]);
                        free(opsPtr[opIndex].output_tensors_name);
                        opsPtr[opIndex].output_tensors_name =
                            (I8 **)mt_new_storage(opsPtr[opIndex].num_outputs * sizeof(I8 *));
                        opsPtr[opIndex].output_tensors_name[0] =
                            (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                        str_copy(opsPtr[opIndex].output_tensors_name[0], outputNames[0].c_str(),
                            outputNames[0].length());
                        opsPtr[opIndex].output_tensors_name[1] =
                            (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                        std::string reduntStr = "DropOut_Str";
                        str_copy(opsPtr[opIndex].output_tensors_name[1], reduntStr.c_str(),
                            reduntStr.length());
                    }
                }

                ParameterSpec curPs;
                ret = adapt_operator(opType, &curPs);
                CHECK_STATUS(ret);
                opsPtr[opIndex].ps = curPs;

                if (opType == OT_BatchNorm) {
                    std::string scaleInputName = outputNames[0];
                    std::string scaleOpName = "scale_" + opName;
                    opIndex++;
                    str_copy(opsPtr[opIndex].name, scaleOpName.c_str(), scaleOpName.length());
                    opsPtr[opIndex].type = OT_Scale;
                    opsPtr[opIndex].ps.scale_spec.axis = 1;
                    opsPtr[opIndex].num_inputs = 1;
                    opsPtr[opIndex].input_tensors_name = (I8 **)mt_new_storage(sizeof(I8 *));
                    opsPtr[opIndex].input_tensors_name[0] =
                        (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[opIndex].input_tensors_name[0], scaleInputName.c_str(),
                        scaleInputName.length());
                    opsPtr[opIndex].num_outputs = 1;
                    opsPtr[opIndex].output_tensors_name = (I8 **)mt_new_storage(sizeof(I8 *));
                    opsPtr[opIndex].output_tensors_name[0] =
                        (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[opIndex].output_tensors_name[0], scaleInputName.c_str(),
                        scaleInputName.length());

                    ParameterSpec scalePs;
                    ret = adapt_operator(opsPtr[opIndex].type, &scalePs);
                    CHECK_STATUS(ret);
                    opsPtr[opIndex].ps = scalePs;
                }
            }

            nodeIndex++;
            opIndex++;
        }
        ms->num_weight_specs = weightOpIndexLists.size() + bnOpNum;
        return ret;
    }

    EE adapt_weights(ModelSpec *ms) override
    {
        EE ret = SUCCESS;
        WeightSpec *wsPtr = (WeightSpec *)mt_new_storage(sizeof(WeightSpec) * ms->num_weight_specs);
        for (int j = 0; j < ms->num_weight_specs; j++) {
            wsPtr[j].num_quant_scale = 0;
            wsPtr[j].weight_scale = nullptr;
        }
        ms->ws = wsPtr;
        int weightOpIndexIndeed = 0;
        for (U32 i = 0; i < (U32)ms->num_weight_specs; i++) {
            int weightOpIndex = weightOpIndexLists[weightOpIndexIndeed];
            const onnx::NodeProto &weightNode = onnxGraph.node(weightOpIndex);
            std::string weightOpName = weightNode.name();
            if (weightOpName.empty()) {
                weightOpName = weightNode.output(0);
            }
            const std::string &weightOpType = weightNode.op_type();

            if (weightOpType == "Conv" || weightOpType == "ConvTranspose") {
                // to check that if any op has bias
                int convInputNum =
                    weightNode.input_size();  // if convInputNum == 3, means has bias , otherwise , do not have bias

                const onnx::TensorProto &convWeightTp = weights[weightNode.input(1)];

                int convWeightNum = get_data_size_from_tensor_proto(convWeightTp);
                float *convWeightParamPtr = get_ptr_from_weight_obj(convWeightTp);
                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());

                // traverse weight elements to see whether it is bnn convolution
                wsPtr[i].mdt = get_weight_data_type(convWeightNum, convWeightParamPtr);
                wsPtr[i].bytes_of_weight =
                    convWeightNum * sizeof(float);  // Please do not change to bytesOf(mdt)
                wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, convWeightParamPtr, wsPtr[i].bytes_of_weight);

                int convBiasNum = 0;
                float *convBiasParamPtr = nullptr;
                if (convInputNum == 3) {
                    const onnx::TensorProto &convBiasTp = weights[weightNode.input(2)];
                    convBiasNum = get_data_size_from_tensor_proto(convBiasTp);
                    convBiasParamPtr = get_ptr_from_weight_obj(convBiasTp);
                    wsPtr[i].bytes_of_vec = convBiasNum * sizeof(float);
                    if (DT_BIN11 == wsPtr[i].mdt || DT_BIN01 == wsPtr[i].mdt) {
                        wsPtr[i].bytes_of_vec *=
                            2;  // BNN conv must have a scale vector and a bias vector, so that it can fuse with BN
                    }
                    wsPtr[i].vec = (U8 *)mt_new_storage(wsPtr[i].bytes_of_vec);
                    if (DT_BIN11 == wsPtr[i].mdt || DT_BIN01 == wsPtr[i].mdt) {
                        U32 vecBytes = convBiasNum * sizeof(float);
                        F32 *scale = (F32 *)wsPtr[i].vec;
                        for (I32 j = 0; j < convBiasNum; j++) {
                            scale[j] = 1.0;
                        }
                        memcpy(wsPtr[i].vec + vecBytes, convBiasParamPtr,
                            vecBytes);  // Copy bias (if any) to the second half for BNN
                    } else {
                        memcpy(wsPtr[i].vec, convBiasParamPtr, wsPtr[i].bytes_of_vec);
                    }
                } else {
                    wsPtr[i].bytes_of_vec = 0;
                    wsPtr[i].vec = nullptr;
                }
            } else if (weightOpType == "Gemm") {
                // attention: fc op weight bias order is different from conv op
                const onnx::TensorProto &fcWeightTp = weights[weightNode.input(1)];
                const onnx::TensorProto &fcBiasTp = weights[weightNode.input(2)];
                int fcWeightNum = get_data_size_from_tensor_proto(fcWeightTp);
                int fcBiasNum = get_data_size_from_tensor_proto(fcBiasTp);
                float *fcWeightParamPtr = get_ptr_from_weight_obj(fcWeightTp);
                float *fcBiasParamPtr = get_ptr_from_weight_obj(fcBiasTp);
                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = fcWeightNum * sizeof(float);
                wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, fcWeightParamPtr, wsPtr[i].bytes_of_weight);
                wsPtr[i].bytes_of_vec = fcBiasNum * sizeof(float);
                wsPtr[i].vec = (U8 *)mt_new_storage(wsPtr[i].bytes_of_vec);
                memcpy(wsPtr[i].vec, fcBiasParamPtr, wsPtr[i].bytes_of_vec);
            } else if (weightOpType == "BatchNormalization") {
                const onnx::TensorProto &scale = weights[weightNode.input(1)];
                const onnx::TensorProto &bias = weights[weightNode.input(2)];
                const onnx::TensorProto &mean = weights[weightNode.input(3)];
                const onnx::TensorProto &var = weights[weightNode.input(4)];

                float *meanPtr = get_ptr_from_weight_obj(mean);
                int bnMeanNum = get_data_size_from_tensor_proto(mean);
                float *varPtr = get_ptr_from_weight_obj(var);
                int bnVarNum = get_data_size_from_tensor_proto(var);

                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = bnMeanNum * sizeof(float);
                wsPtr[i].bytes_of_vec = bnVarNum * sizeof(float);

                wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, meanPtr, wsPtr[i].bytes_of_weight);
                wsPtr[i].vec = (U8 *)mt_new_storage(wsPtr[i].bytes_of_vec);
                memcpy(wsPtr[i].vec, varPtr, wsPtr[i].bytes_of_vec);

                // for scale
                std::string scaleWeightOpName = "scale_" + weightOpName;
                i = i + 1;
                float *scalePtr = get_ptr_from_weight_obj(scale);
                int scaleWeightNum = get_data_size_from_tensor_proto(scale);
                float *biasPtr = get_ptr_from_weight_obj(bias);
                int scaleBiasNum = get_data_size_from_tensor_proto(bias);

                str_copy(wsPtr[i].op_name, scaleWeightOpName.c_str(), scaleWeightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = scaleWeightNum * sizeof(float);
                wsPtr[i].bytes_of_vec = scaleBiasNum * sizeof(float);

                wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, scalePtr, wsPtr[i].bytes_of_weight);
                wsPtr[i].vec = (U8 *)mt_new_storage(wsPtr[i].bytes_of_vec);
                memcpy(wsPtr[i].vec, biasPtr, wsPtr[i].bytes_of_vec);
            } else if (weightOpType == "Add") {
                const onnx::TensorProto &bias = weights[weightNode.input(1)];
                float *bias_ptr = get_ptr_from_weight_obj(bias);
                int bias_num = get_data_size_from_tensor_proto(bias);

                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = 0;
                wsPtr[i].bytes_of_vec = bias_num * sizeof(float);
                wsPtr[i].weight = nullptr;
                wsPtr[i].vec = (U8 *)mt_new_storage(wsPtr[i].bytes_of_vec);
                memcpy(wsPtr[i].vec, bias_ptr, wsPtr[i].bytes_of_vec);
            } else if (weightOpType == "Mul") {
                auto indices = getOperatorWeightInputIndex(weightOpIndex);
                CHECK_REQUIREMENT(0 != indices.size());
                const onnx::TensorProto &weight = weights[weightNode.input(indices[0])];
                float *weight_ptr = get_ptr_from_weight_obj(weight);
                int weight_num = get_data_size_from_tensor_proto(weight);

                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = weight_num * sizeof(float);
                wsPtr[i].bytes_of_vec = 0;
                wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, weight_ptr, wsPtr[i].bytes_of_weight);
                wsPtr[i].vec = nullptr;
            } else if (weightOpType == "MatMul" || weightOpType == "PRelu") {
                const onnx::TensorProto &weight = weights[weightNode.input(1)];
                float *weight_ptr = get_ptr_from_weight_obj(weight);
                int weight_num = get_data_size_from_tensor_proto(weight);

                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = weight_num * sizeof(float);
                wsPtr[i].bytes_of_vec = 0;
                wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);
                int row = weight.dims(0);
                int column = weight.dims(1);
                for (int m = 0, index = 0; m < column; m++) {
                    for (int n = 0; n < row; n++, index += sizeof(float)) {
                        memcpy(wsPtr[i].weight + index, weight_ptr + n * column + m, sizeof(float));
                    }
                }
                wsPtr[i].vec = nullptr;
            } else if (weightOpType == "Div") {
                const onnx::TensorProto &weight = weights[weightNode.input(1)];
                float *weight_ptr = get_ptr_from_weight_obj(weight);
                int weight_num = get_data_size_from_tensor_proto(weight);

                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = weight_num * sizeof(float);
                wsPtr[i].bytes_of_vec = 0;
                wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);
                F32 *scale = (F32 *)wsPtr[i].weight;
                memcpy(scale, weight_ptr, wsPtr[i].bytes_of_weight);
                for (int j = 0; j < weight_num; j++) {
                    scale[j] = 1 / scale[j];
                }
                wsPtr[i].vec = nullptr;
            } else if (weightOpType == "Transpose") {
                const onnx::TensorProto &weight = weights[weightNode.input(0)];
                float *weight_ptr = get_ptr_from_weight_obj(weight);
                int weight_num = get_data_size_from_tensor_proto(weight);

                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = weight_num * sizeof(float);
                // For the time being, use bytes_of_vec to record the horizontal length of weight
                wsPtr[i].bytes_of_vec = weight.dims(0);
                wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, weight_ptr, wsPtr[i].bytes_of_weight);
                wsPtr[i].vec = nullptr;
            } else if (weightOpType == "LSTM") {
                const onnx::TensorProto &W = weights[weightNode.input(1)];
                const onnx::TensorProto &R = weights[weightNode.input(2)];
                const onnx::TensorProto &B = weights[weightNode.input(3)];

                float *W_ptr = get_ptr_from_weight_obj(W);
                float *R_ptr = get_ptr_from_weight_obj(R);
                float *B_ptr = get_ptr_from_weight_obj(B);

                int W_dim_size = W.dims_size();
                int R_dim_size = R.dims_size();
                int iter_times = 1;
                std::vector<int> W_dims_vec;
                std::vector<int> R_dims_vec;
                if (W_dim_size != R_dim_size) {
                    UNI_ERROR_LOG("not support onnx LSTM W != R\n");
                } else {
                    for (int k = 0; k < W_dim_size - 1; k++) {
                        W_dims_vec.push_back(W.dims(k));
                        R_dims_vec.push_back(R.dims(k));
                        iter_times *= W.dims(k);
                    }
                }
                int W_con_dim_size = W.dims(W_dim_size - 1);
                int R_con_dim_size = R.dims(R_dim_size - 1);
                int W_weight_num = get_data_size_from_tensor_proto(W);
                int R_weight_num = get_data_size_from_tensor_proto(R);
                int B_weight_num = get_data_size_from_tensor_proto(B);

                wsPtr[i].mdt = DT_F32;
                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].bytes_of_weight = (W_weight_num + R_weight_num) * sizeof(float);
                wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);
                int begin_index = 0;
                for (int k = 0; k < iter_times; k++) {
                    memcpy(((float *)wsPtr[i].weight) + begin_index, W_ptr + k * W_con_dim_size,
                        W_con_dim_size * sizeof(float));
                    memcpy(((float *)wsPtr[i].weight) + begin_index + W_con_dim_size,
                        R_ptr + k * R_con_dim_size, R_con_dim_size * sizeof(float));
                    begin_index += (W_con_dim_size + R_con_dim_size);
                }
                wsPtr[i].bytes_of_vec = B_weight_num * sizeof(float);
                wsPtr[i].vec = (U8 *)mt_new_storage(wsPtr[i].bytes_of_vec);
                memcpy(wsPtr[i].vec, B_ptr, wsPtr[i].bytes_of_vec);
            } else if (weightOpType == "Gather") {
                auto weightTp = weights[weightNode.input(0)];
                int weightNum = get_data_size_from_tensor_proto(weightTp);
                float *weightParamPtr = get_ptr_from_weight_obj(weightTp);
                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = weightNum * sizeof(float);
                wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, weightParamPtr, wsPtr[i].bytes_of_weight);
                wsPtr[i].bytes_of_vec = 0;
                wsPtr[i].vec = nullptr;
            } else if (weightOpType == "Splice") {
                std::vector<int> indices =
                    get_node_vector_ints_attribute_by_name(weightNode, "forward_indexes");
                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_U32;
                wsPtr[i].bytes_of_weight = indices.size() * sizeof(U32);
                wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, indices.data(), wsPtr[i].bytes_of_weight);
                wsPtr[i].bytes_of_vec = 0;
                wsPtr[i].vec = nullptr;
            }

            weightOpIndexIndeed++;
        }
        return ret;
    }

    ParameterSpec adapt_SharedWeight() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        const onnx::TensorProto& data = weights[node.input(0)];
        const onnx::TensorProto& ind = weights[node.input(1)];
        SharedWeightParamSpec sharedWeightPs;
        sharedWeightPs.desc.nDims = 3;
        sharedWeightPs.desc.dims[2] = 1;
        sharedWeightPs.desc.dims[1] = ind.dims(1);
        sharedWeightPs.desc.dims[0] = data.dims(1);
        sharedWeightPs.desc.df = DF_NORMAL;
        sharedWeightPs.desc.dt = DT_F32;
        UNI_DEBUG_LOG("SharedWeight: %s\n" ,tensorDesc2Str(sharedWeightPs.desc).c_str());
        curPs.shared_weight_spec = sharedWeightPs;
        return curPs;
    }

    ParameterSpec adapt_Reshape() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        ReshapeParamSpec reshapePs;
        initialization_zero(&reshapePs, sizeof(reshapePs));
        std::vector<int> reshapeInfo;
        if (this->op == "Flatten") {
            int axis = get_node_single_int_attribute_by_name(node, "axis", 1);
            for (int i = 0; i < axis; i++) {
                reshapeInfo.push_back(0);
            }
            reshapeInfo.push_back(-1);
        } else {
            if (node.input_size() == 1) {
                reshapeInfo = get_node_vector_ints_attribute_by_name(node, "shape");
            } else {
                reshapeInfo = get_reshapeInfo_from_tensorProto(
                    weights[node.input(1)]);  // tp:weights[node.input(1)]
            }
        }
        reshapePs.shape_size = reshapeInfo.size();
        memcpy(reshapePs.shape_dims, reshapeInfo.data(), reshapePs.shape_size * sizeof(I32));
        curPs.reshape_spec = reshapePs;
        return curPs;
    }

    ParameterSpec adapt_Resize() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        ResizeParamSpec resizePs;
        initialization_zero(&resizePs, sizeof(resizePs));
        std::string mode = get_node_str_attribute_by_name(node, "mode", "linear");
        str_copy(resizePs.mode, mode.c_str(), mode.length());
        resizePs.num_scales = 0;
        resizePs.num_sizes = 0;
        std::string scalesIndex = "";
        std::string sizesIndex = "";
        if (node.op_type() == "Resize") {
            scalesIndex = node.input(2);
            if (node.input_size() == 4) {
                sizesIndex = node.input(3);
            }
        } else if (node.op_type() == "Upsample") {
            scalesIndex = node.input(1);
        } else {
            UNI_ERROR_LOG("unsupported resize op name %s\n", node.op_type().c_str());
        }
        if (scalesIndex != "") {
            const onnx::TensorProto &scales = weights[scalesIndex];
            if (scales.dims(0) > 0) {
                CHECK_REQUIREMENT(scales.dims(0) == 4);
                resizePs.num_scales = scales.dims(0);
                F32 *scalesPtr = nullptr;
                if (scales.has_raw_data()) {
                    const std::string &rawData = scales.raw_data();
                    scalesPtr = (F32 *)(rawData.data());
                } else if (scales.data_type() == 1) {
                    scalesPtr = (F32 *)(scales.float_data().data());
                } else {
                    UNI_ERROR_LOG("Resize extract scales failed\n");
                }
                memcpy(resizePs.scales, scalesPtr, resizePs.num_scales * bytesOf(DT_F32));
            }
        }
        if (sizesIndex != "") {
            const onnx::TensorProto &sizes = weights[sizesIndex];
            if (sizes.dims(0) > 0) {
                CHECK_REQUIREMENT(sizes.dims(0) == 4);
                if (sizes.has_raw_data()) {
                    const std::string &rawData = sizes.raw_data();
                    I64 *sizesPtr = (I64 *)(rawData.data());
                    resizePs.sizes[0] = sizesPtr[2];
                    resizePs.sizes[1] = sizesPtr[3];
                } else if (sizes.data_type() == 1) {
                    resizePs.sizes[0] = sizes.int64_data(2);
                    resizePs.sizes[1] = sizes.int64_data(3);
                } else {
                    UNI_ERROR_LOG("Resize extract sizes failed\n");
                }
                resizePs.num_sizes = 2;
            }
        }
        curPs.resize_spec = resizePs;
        return curPs;
    }

    ParameterSpec adapt_Transpose() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        TransposeParamSpec transposePs;
        initialization_zero(&transposePs, sizeof(transposePs));
        std::vector<int> transpose_info = get_node_vector_ints_attribute_by_name(node, "perm");
        transposePs.trans_size = transpose_info.size();
        memcpy(transposePs.trans_dims, transpose_info.data(), transposePs.trans_size * sizeof(U32));
        curPs.transpose_spec = transposePs;
        return curPs;
    }

    ParameterSpec adapt_Clip() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        ClipParamSpec clipParam;
        initialization_zero(&clipParam, sizeof(clipParam));
        if (op == "Max") {
            clipParam.min = 0;
            clipParam.max = UNI_F16_MAX;
        } else if (op == "Min") {
            clipParam.min = -UNI_F16_MAX;
            clipParam.max = 1;
        } else {  // op == "Clip"
            if (node.input_size() == 1) {
                clipParam.min = get_node_float_attribute_by_name(node, "min", -UNI_F16_MAX);
                clipParam.max = get_node_float_attribute_by_name(node, "max", UNI_F16_MAX);
            } else {
                auto minTp = weights[node.input(1)];
                auto maxTp = weights[node.input(2)];
                clipParam.min = getSinFloat_from_tensorProto(minTp);
                clipParam.max = getSinFloat_from_tensorProto(maxTp);
            }
        }
        curPs.clip_spec = clipParam;
        return curPs;
    }

    ParameterSpec adapt_Conv() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        ConvolutionParamSpec cps;
        initialization_zero(&cps, sizeof(cps));
        std::vector<int> kernelShape = get_node_vector_ints_attribute_by_name(node, "kernel_shape");
        std::vector<int> dilations = get_node_vector_ints_attribute_by_name(node, "dilations");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(node, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(node, "pads");
        int group = get_node_single_int_attribute_by_name(node, "group", 1);

        const onnx::TensorProto &weight = weights[node.input(1)];
        cps.num_outputs = weight.dims(0);
        cps.num_outputs_origin = cps.num_outputs;
        cps.kernel_t = 1;
        cps.stride_t = 1;
        cps.padding_before = 0;
        cps.padding_after = 0;
        cps.dilatedRate_t = 1;
        if (kernelShape.size() == 2) {
            cps.kernel_h = kernelShape[0];
            cps.kernel_w = kernelShape[1];
        } else if (kernelShape.size() == 1) {
            cps.kernel_h = kernelShape[0];
            cps.kernel_w = 1;
        } else {
            UNI_ERROR_LOG("convolution: kernel_size unknown\n");
        }

        if (dilations.size() == 2) {
            cps.dilatedRate_h = dilations[0];
            cps.dilatedRate_w = dilations[1];
        } else if (dilations.size() == 1) {
            cps.dilatedRate_h = dilations[0];
            cps.dilatedRate_w = 1;
        } else {
            UNI_WARNING_LOG("convolution: dilation unknown. Default to 1\n");
            cps.dilatedRate_h = 1;
            cps.dilatedRate_w = 1;
        }

        if (strides.size() == 2) {
            cps.stride_h = strides[0];
            cps.stride_w = strides[1];
        } else if (strides.size() == 1) {
            cps.stride_h = strides[0];
            cps.stride_w = 1;
        } else {
            UNI_ERROR_LOG("convolution: stride unknown\n");
        }

        if (pads.size() == 4) {
            if (cps.kernel_h == cps.kernel_w && (pads[0] != pads[2] || pads[1] != pads[3])) {
                cps.padding_top = UNI_MAX(pads[0], pads[2]);
                cps.padding_bottom = UNI_MAX(pads[0], pads[2]);
                cps.padding_left = UNI_MAX(pads[1], pads[3]);
                cps.padding_right = UNI_MAX(pads[1], pads[3]);
            } else {
                cps.padding_top = pads[0];
                cps.padding_left = pads[1];
                cps.padding_bottom = pads[2];
                cps.padding_right = pads[3];
            }
        } else if (pads.size() == 2) {
            cps.padding_top = pads[0];
            cps.padding_bottom = pads[1];
            cps.padding_left = 0;
            cps.padding_right = 0;
        } else {
            UNI_ERROR_LOG("deconvolution: pad unknown\n");
        }

        cps.group = group;
        if (cps.group != 1 && cps.group == cps.num_outputs) {
            cps.convolution_type = Convolution_Depthwise;
        } else {
            if (cps.dilatedRate_h > 1 || cps.dilatedRate_w > 1) {
                cps.convolution_type = Convolution_Dilation;
            } else {
                cps.convolution_type = Convolution_Pointwise;
            }
        }

        cps.dw_activation_type = ACTIVATION_NULL;
        cps.pw_activation_type = ACTIVATION_NULL;
        curPs.conv_spec = cps;
        return curPs;
    }

    ParameterSpec adapt_Deconvolution() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        ConvolutionParamSpec cps;
        initialization_zero(&cps, sizeof(cps));
        std::vector<int> kernelShape = get_node_vector_ints_attribute_by_name(node, "kernel_shape");
        std::vector<int> dilations = get_node_vector_ints_attribute_by_name(node, "dilations");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(node, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(node, "pads");
        int group = get_node_single_int_attribute_by_name(node, "group", 1);

        const onnx::TensorProto &weight = weights[node.input(1)];
        cps.num_outputs = weight.dims(1);
        cps.kernel_t = 1;
        cps.stride_t = 1;
        cps.padding_before = 0;
        cps.padding_after = 0;
        cps.dilatedRate_t = 1;
        if (kernelShape.size() == 2) {
            cps.kernel_h = kernelShape[0];
            cps.kernel_w = kernelShape[1];
        } else if (kernelShape.size() == 1) {
            cps.kernel_h = kernelShape[0];
            cps.kernel_w = 1;
        } else {
            UNI_ERROR_LOG("deconvolution: kernel_size unknown\n");
        }

        if (dilations.size() == 2) {
            cps.dilatedRate_h = dilations[0];
            cps.dilatedRate_w = dilations[1];
        } else if (dilations.size() == 1) {
            cps.dilatedRate_h = dilations[0];
            cps.dilatedRate_w = 1;
        } else {
            UNI_ERROR_LOG("deconvolution: dilation unknown\n");
        }

        if (strides.size() == 2) {
            cps.stride_h = strides[0];
            cps.stride_w = strides[1];
        } else if (strides.size() == 1) {
            cps.stride_h = strides[0];
            cps.stride_w = 1;
        } else {
            UNI_ERROR_LOG("deconvolution: stride unknown\n");
        }

        if (pads.size() == 4) {
            cps.padding_top = pads[0];
            cps.padding_left = pads[1];
            cps.padding_bottom = pads[2];
            cps.padding_right = pads[3];
        } else if (pads.size() == 2) {
            cps.padding_top = pads[0];
            cps.padding_bottom = pads[1];
            cps.padding_left = 0;
            cps.padding_right = 0;
        } else {
            UNI_ERROR_LOG("deconvolution: pad unknown\n");
        }

        cps.group = group;
        if (1 == group) {
            cps.convolution_type = Convolution_Deconvolution;
        } else {
            cps.convolution_type = Convolution_Depthwise_Deconvolution;
            cps.num_outputs = weight.dims(0);
        }
        cps.num_outputs_origin = cps.num_outputs;
        cps.dw_activation_type = ACTIVATION_NULL;
        cps.pw_activation_type = ACTIVATION_NULL;
        curPs.conv_spec = cps;
        return curPs;
    }

    ParameterSpec adapt_Pooling() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        PoolingParamSpec pps;
        initialization_zero(&pps, sizeof(pps));
        std::string autoPad = get_node_str_attribute_by_name(node, "auto_pad");  // deprecated
        std::vector<int> kernelShape = get_node_vector_ints_attribute_by_name(node, "kernel_shape");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(node, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(node, "pads");

        pps.kernel_t = 1;
        pps.stride_t = 1;
        pps.padding_before = 0;
        pps.padding_after = 0;
        if (op == "AveragePool" || op == "ReduceMean" || op == "GlobalAveragePool") {
            pps.mode = POOLING_MEAN;
        } else {
            pps.mode = POOLING_MAX;
        }

        if (autoPad == "SAME_UPPER") {
            pps.rm = CEIL;
        } else {
            pps.rm = FLOOR;
        }

        if (kernelShape.size() == 2) {
            pps.kernel_h = kernelShape[0];
            pps.kernel_w = kernelShape[1];
        } else {
            pps.kernel_h = 0;
            pps.kernel_w = 0;
            UNI_INFO_LOG("pooling: kernel_size unknown. This could be global pooling.\n");
        }

        if (strides.size() == 2) {
            pps.stride_h = strides[0];
            pps.stride_w = strides[1];
        } else {
            pps.stride_h = 1;
            pps.stride_w = 1;
            UNI_INFO_LOG("pooling: stride unknown. This could be global pooling.\n");
        }

        if (pads.size() == 4) {
            pps.padding_top = pads[0];
            pps.padding_bottom = pads[2];
            pps.padding_left = pads[1];
            pps.padding_right = pads[3];
        } else {
            pps.padding_top = 0;
            pps.padding_bottom = 0;
            pps.padding_left = 0;
            pps.padding_right = 0;
        }
        curPs.pooling_spec = pps;
        return curPs;
    }

    ParameterSpec adapt_MatMul() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        MatMulParamSpec matmulPs;
        initialization_zero(&matmulPs, sizeof(matmulPs));
        matmulPs.transpose_a = false;
        matmulPs.transpose_b = false;
        curPs.matmul_spec = matmulPs;
        return curPs;
    }

    ParameterSpec adapt_Fc() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        FullyConnectedParamSpec fcParamSpec;
        initialization_zero(&fcParamSpec, sizeof(fcParamSpec));
        fcParamSpec.num_outputs = -1;

        if (op == "MatMul") {
            const onnx::TensorProto &matmulTp = weights[node.input(1)];
            if (matmulTp.dims_size() == 2) {
                fcParamSpec.num_outputs = matmulTp.dims(1);
            } else {
                UNI_ERROR_LOG("onnx model adaptor not support matmul\n");
            }
        } else {
            float alpha = get_node_float_attribute_by_name(node, "alpha", 1.f);
            float beta = get_node_float_attribute_by_name(node, "beta", 1.f);
            int transA = get_node_single_int_attribute_by_name(node, "transA", 0);
            int transB = get_node_single_int_attribute_by_name(node, "transB", 0);
            if (alpha == 1.f && beta == 1.f) {
                if (transA == 0 && transB == 1) {
                    const onnx::TensorProto &C = weights[node.input(2)];
                    int num_output = get_data_size_from_tensor_proto(C);
                    fcParamSpec.num_outputs = num_output;
                }
            } else {
                UNI_ERROR_LOG("onnx model adaptor fully connect layer num_output is unkown\n");
            }
        }
        fcParamSpec.num_slices = 1;
        fcParamSpec.slice_point[0] = fcParamSpec.num_outputs;
        curPs.fc_spec = fcParamSpec;
        return curPs;
    }

    ParameterSpec adapt_BatchNorm() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        BatchNormParamSpec bnPs;
        initialization_zero(&bnPs, sizeof(bnPs));
        bnPs.eps = get_node_float_attribute_by_name(node, "epsilon", 1e-5f);
        bnPs.axis = 1;
        bnPs.gama = 1;
        bnPs.momentum = get_node_float_attribute_by_name(node, "momentum", 0.9);
        curPs.bn_spec = bnPs;
        return curPs;
    }

    ParameterSpec adapt_Eltwise() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        EltwiseParamSpec eps;
        initialization_zero(&eps, sizeof(eps));
        if (op == "Add") {
            eps.elt_mode = ELTWISE_SUM;
            eps.elt_sum_spec.coeff_size = 2;
            for (I32 j = 0; j < eps.elt_sum_spec.coeff_size; j++) {
                eps.elt_sum_spec.coeff_values[j] = 1.0;
            }
        } else if (op == "Mul") {
            eps.elt_mode = ELTWISE_PROD;
        } else if (op == "Sub") {
            eps.elt_mode = ELTWISE_SUB;
        } else if (op == "Div") {
            eps.elt_mode = ELTWISE_DIV;
        } else {
            CHECK_STATUS(NOT_IMPLEMENTED);
        }
        eps.activation_type = ACTIVATION_NULL;
        curPs.eltwise_spec = eps;
        return curPs;
    }

    void handle_Constant()
    {
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = node.attribute(i);
            if (attribute.name() == "value") {
                CHECK_REQUIREMENT(4 == attribute.type());
                const onnx::TensorProto &tp = attribute.t();
                weights[node.output(0)] = tp;
                break;
            }
        }
    }

    ParameterSpec adapt_Pad() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        PadParamSpec padPs;
        initialization_zero(&padPs, sizeof(padPs));
        std::string padModeStr = get_node_str_attribute_by_name(node, "mode");
        std::vector<int> padVec = get_node_vector_ints_attribute_by_name(node, "pads");
        F32 padValue = get_node_float_attribute_by_name(node, "value", 0.f);
        if (padModeStr == "constant") {
            padPs.pad_mode = Pad_Constant;
        } else if (padModeStr == "edge") {
            padPs.pad_mode = Pad_Edge;
        } else if (padModeStr == "reflect") {
            padPs.pad_mode = Pad_Reflect;
        } else {
            UNI_ERROR_LOG("unknown pad mode: %s\n", padModeStr.c_str());
        }

        padPs.before = 0;
        padPs.after = 0;
        U32 padSize = padVec.size();
        if (padSize == 8) {  // NCHW
            padPs.top = padVec[2];
            padPs.left = padVec[3];
            padPs.bottom = padVec[6];
            padPs.right = padVec[7];
        } else if (padSize == 6) {  // NCH
            padPs.top = padVec[2];
            padPs.left = 0;
            padPs.bottom = padVec[5];
            padPs.right = 0;
        } else if (padSize == 4) {  // HW
            padPs.top = padVec[0];
            padPs.left = padVec[1];
            padPs.bottom = padVec[2];
            padPs.right = padVec[3];
        } else {
            UNI_ERROR_LOG("unsupported pad length\n");
        }
        padPs.constant_value = padValue;
        curPs.pad_spec = padPs;
        return curPs;
    }

    ParameterSpec adapt_Gather() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        GatherParamSpec gps;
        initialization_zero(&gps, sizeof(gps));
        gps.gather_axis = get_node_single_int_attribute_by_name(node, "axis", 0);
        curPs.gather_spec = gps;
        return curPs;
    }

    ParameterSpec adapt_Slice() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        SliceParamSpec slice_ps;
        initialization_zero(&slice_ps, sizeof(slice_ps));
        if (op == "Gather") {
            ParameterSpec gather_ps = adapt_Gather();
            slice_ps.slice_points[0] = 1;
            slice_ps.slice_size = 1;
            slice_ps.axis = gather_ps.gather_spec.gather_axis;
        } else if (op == "Slice") {
            std::vector<int> startsInfo = get_node_vector_ints_attribute_by_name(node, "starts");
            CHECK_REQUIREMENT(0 == startsInfo[0]);  // Support this only case
            std::vector<int> endsInfo = get_node_vector_ints_attribute_by_name(node, "ends");
            std::vector<int> axesInfo = get_node_vector_ints_attribute_by_name(node, "axes");
            slice_ps.slice_points[0] = endsInfo[0];
            slice_ps.slice_size = 1;
            slice_ps.axis = axesInfo[0];
        } else if (op == "Split") {
            std::vector<int> splitInfo = get_node_vector_ints_attribute_by_name(node, "split");
            slice_ps.axis = get_node_single_int_attribute_by_name(node, "axis", 0);
            if (0 == splitInfo.size()) {  // Split equally by default. Set all slice_points to 0
                slice_ps.slice_size = (int)node.output_size();
                memset(slice_ps.slice_points, 0, slice_ps.slice_size * sizeof(I32));
            } else {
                slice_ps.slice_size = splitInfo.size();
                slice_ps.slice_points[0] = splitInfo[0];
                for (U32 i = 1; i < slice_ps.slice_size; i++) {
                    slice_ps.slice_points[i] = slice_ps.slice_points[i - 1] + splitInfo[i];
                }
            }
        }
        curPs.slice_spec = slice_ps;
        return curPs;
    }

    ParameterSpec adapt_Embedding() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        EmbedParamSpec embed_ps;
        initialization_zero(&embed_ps, sizeof(embed_ps));
        std::string embed_weight_name = node.input(0);
        auto tensor_proto = weights[embed_weight_name];
        int size_of_dims = tensor_proto.dims_size();
        if (size_of_dims != 2) {
            UNI_ERROR_LOG("unsupported onnx embedding parameter\n");
        }
        embed_ps.input_dim = tensor_proto.dims(0);
        embed_ps.num_output = tensor_proto.dims(1);
        embed_ps.bias_term = false;
        embed_ps.transpose = false;
        curPs.embed_spec = embed_ps;
        return curPs;
    }

    ParameterSpec adapt_Squeeze() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        SqueezeParamSpec squeezePs;
        initialization_zero(&squeezePs, sizeof(squeezePs));
        std::vector<int> squeezeAxes = get_node_vector_ints_attribute_by_name(node, "axes");
        squeezePs.axes_num = squeezeAxes.size();
        for (int squeeze_i = 0; squeeze_i < (int)squeezeAxes.size(); squeeze_i++) {
            squeezePs.axes[squeeze_i] = squeezeAxes[squeeze_i];
        }
        curPs.squeeze_spec = squeezePs;
        return curPs;
    }

    ParameterSpec adapt_Unsqueeze() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        UnsqueezeParamSpec unsqueezePs;
        initialization_zero(&unsqueezePs, sizeof(unsqueezePs));
        std::vector<int> unsqueezeAxes = get_node_vector_ints_attribute_by_name(node, "axes");
        unsqueezePs.axes_num = unsqueezeAxes.size();
        for (int unsqueeze_i = 0; unsqueeze_i < (int)unsqueezeAxes.size(); unsqueeze_i++) {
            unsqueezePs.axes[unsqueeze_i] = unsqueezeAxes[unsqueeze_i];
        }
        curPs.unsqueeze_spec = unsqueezePs;
        return curPs;
    }

    ParameterSpec adapt_Cast() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        CastParamSpec castPs;
        initialization_zero(&castPs, sizeof(castPs));
        int cast_to = get_node_single_int_attribute_by_name(node, "to", 0);
        if (cast_to == 1) {
            castPs.castPrecision = ToFloat;
        } else if (cast_to == 5 || cast_to == 6 || cast_to == 7) {
            castPs.castPrecision = ToInt;
        } else {
            castPs.castPrecision = KeepPrecision;
        }
        curPs.cast_spec = castPs;
        return curPs;
    }

    ParameterSpec adapt_Concat() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        ConcatParamSpec concatPs;
        initialization_zero(&concatPs, sizeof(concatPs));
        concatPs.axis = get_node_single_int_attribute_by_name(node, "axis", 1);
        curPs.concat_spec = concatPs;
        return curPs;
    }

    ParameterSpec adapt_Softmax() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        SoftmaxParamSpec softmaxPs;
        initialization_zero(&softmaxPs, sizeof(softmaxPs));
        softmaxPs.axis = get_node_single_int_attribute_by_name(node, "axis", 1);
        curPs.softmax_spec = softmaxPs;
        return curPs;
    }

    ParameterSpec adapt_Relu() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        ReLUParamSpec reluPs;
        initialization_zero(&reluPs, sizeof(reluPs));
        reluPs.neg_slope = get_node_float_attribute_by_name(node, "alpha", 0.0);
        curPs.relu_spec = reluPs;
        return curPs;
    }

    ParameterSpec adapt_RNN() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        RNNParamSpec rnnPs;
        initialization_zero(&rnnPs, sizeof(rnnPs));
        if (this->op == "RNN") {
            rnnPs.mode = RNN_RNN;
        } else if (this->op == "LSTM") {
            rnnPs.mode = RNN_LSTM;
        } else if (this->op == "GRU") {
            int linear_before_reset =
                get_node_single_int_attribute_by_name(node, "linear_before_reset", 0);
            if (linear_before_reset == 0) {
                rnnPs.mode = RNN_GRU;
            } else {
                rnnPs.mode = RNN_GRU_LBR;
            }
        } else {
            UNI_ERROR_LOG("not support %s currently\n", this->op.c_str());
        }
        rnnPs.numOutput = get_node_single_int_attribute_by_name(node, "hidden_size", 1);
        rnnPs.biDirection =
            get_node_str_attribute_by_name(node, "direction", "forward") == "bidirectional" ? true
                                                                                            : false;
        rnnPs.steps = 0;
        rnnPs.numProjection = 0;
        rnnPs.zoneoutCell = 0;
        rnnPs.zoneoutOutput = 0;
        rnnPs.forgetBias = 1.0;
        rnnPs.activationMode = ACTIVATION_TANH;
        curPs.rnn_spec = rnnPs;
        return curPs;
    }

    ParameterSpec adapt_Power() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        PowerParamSpec powerPs;
        initialization_zero(&powerPs, sizeof(powerPs));
        powerPs.scale = 1;
        powerPs.shift = 0;
        if (this->op == "Pow") {
            auto tp = weights[node.input(1)];
            powerPs.power = getSinFloat_from_tensorProto(tp);
        } else if (this->op == "Sqrt") {
            powerPs.power = 0.5;
        } else {
            UNI_ERROR_LOG("onnx model read failed in adapt_Power for %s\n", this->op.c_str());
        }
        curPs.power_spec = powerPs;
        return curPs;
    }

    ParameterSpec adapt_Scale() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        ScaleParamSpec scale_ps;
        initialization_zero(&scale_ps, sizeof(scale_ps));
        scale_ps.axis = 1;
        curPs.scale_spec = scale_ps;
        return curPs;
    }

    ParameterSpec adapt_Space2Depth() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        Space2DepthParamSpec s2dPs;
        initialization_zero(&s2dPs, sizeof(s2dPs));
        s2dPs.blockSize = get_node_single_int_attribute_by_name(node, "blocksize", 1);
        curPs.space2depth_spec = s2dPs;
        return curPs;
    }

    ParameterSpec adapt_Depth2Space() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        Depth2SpaceParamSpec d2sPs;
        initialization_zero(&d2sPs, sizeof(d2sPs));
        d2sPs.blockSize = get_node_single_int_attribute_by_name(node, "blocksize", 1);
        std::string d2s_mode = get_node_str_attribute_by_name(node, "mode", "DCR");
        str_copy(d2sPs.reMode, d2s_mode.c_str(), d2s_mode.length(), 8);
        curPs.depth2space_spec = d2sPs;
        return curPs;
    }

    ParameterSpec adapt_Reduction() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        ReductionParamSpec rsPs;
        initialization_zero(&rsPs, sizeof(rsPs));
        std::vector<int> axesInfo =
            get_node_vector_ints_attribute_by_name(node, "axes");  // default one element
        int keepdimsInfo = get_node_single_int_attribute_by_name(node, "keepdims", 0);
        rsPs.axes[0] = axesInfo[0];
        rsPs.axes_num = 1;
        rsPs.keep_dim = keepdimsInfo == 0 ? false : true;
        rsPs.coeff = 1.0;
        if (op == "ReduceSum") {
            rsPs.reduction_mode = REDUCTION_SUM;
        } else {
            rsPs.reduction_mode = REDUCTION_MEAN;
        }
        curPs.reduction_spec = rsPs;
        return curPs;
    }

    ParameterSpec adapt_ArgMax() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        ArgMaxParamSpec amPs;
        initialization_zero(&amPs, sizeof(amPs));
        amPs.axis = get_node_single_int_attribute_by_name(node, "axis", -1);
        curPs.argmax_spec = amPs;
        return curPs;
    }

    ParameterSpec adapt_PRelu() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        return curPs;
    }

    ParameterSpec adapt_Tile() override
    {
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        TileParamSpec tilePs;
        initialization_zero(&tilePs, sizeof(tilePs));
        std::vector<int> tileInfo = get_reshapeInfo_from_tensorProto(weights[node.input(1)]);
        if (tileInfo.size() > 0 && tileInfo.size() <= 8) {
            tilePs.dimsSize = tileInfo.size();
        } else {
            UNI_ERROR_LOG("not support this mode tile currently\n");
        }
        for (U32 i = 0; i < tileInfo.size(); i++) {
            tilePs.repeatsInfo[i] = tileInfo[i];
        }
        curPs.tile_spec = tilePs;
        return curPs;
    }

    ParameterSpec adapt_Splice() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        initialization_zero(&curPs, sizeof(curPs));
        SpliceParamSpec splicePs;
        initialization_zero(&splicePs, sizeof(splicePs));
        splicePs.outputDim = get_node_single_int_attribute_by_name(node, "output_dim", 600);
        std::vector<int> indices = get_node_vector_ints_attribute_by_name(node, "forward_indexes");
        splicePs.numIndices = indices.size();
        curPs.splice_spec = splicePs;
        return curPs;
    }

private:
    std::string op;
    std::string modelName;
    int removePreprocessOpNum;
    TensorDesc inputDesc;
    onnx::ModelProto onnxModel;
    onnx::GraphProto onnxGraph;
    onnx::NodeProto node;
    std::map<std::string, onnx::TensorProto> weights;
    int nodeIndex;
    std::vector<int> weightOpIndexLists;
    int opFinalInputNum;
};
#endif
