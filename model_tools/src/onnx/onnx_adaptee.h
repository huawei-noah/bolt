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

#include <iostream>
#include <string>
#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "onnx.pb.h"

#include "model_adaptee.h"

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
        char *environmentSetting = getenv("BOLT_BNN");
        bool useBNN =
            (environmentSetting != NULL && std::string(environmentSetting) == std::string("OFF"))
            ? false
            : true;
        if (1 >= weightLen || !useBNN) {
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
            UNI_ERROR_LOG("can not open onnx model file %s.\n", path);
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
        } else if (inputType == "BatchNormalization" || inputType == "BatchNorm") {
            return OT_BatchNorm;
        } else if (inputType == "Sum" || inputType == "Add" || inputType == "Mul" ||
            inputType == "Div" || inputType == "Sub") {
            std::vector<int> indexes = getOperatorWeightInputIndex(this->nodeIndex);
            if (indexes.size() == 0) {
                return OT_Eltwise;
            } else {
                CHECK_REQUIREMENT(indexes.size() == 1);
                const onnx::TensorProto &weightTp = weights[this->node.input(indexes[0])];
                int weightNum = get_data_size_from_tensor_proto(weightTp);
                if (weightNum == 1) {
                    return OT_Power;
                } else {
                    return OT_Scale;
                }
            }
        } else if (inputType == "Gemm" || inputType == "Linear") {
            return OT_FC;
        } else if (inputType == "AveragePool" || inputType == "MaxPool" ||
            inputType == "GlobalAveragePool") {
            return OT_Pooling;
        } else if (inputType == "ReduceMean" || inputType == "ReduceMax") {
            std::vector<int> axesInfo = get_node_vector_ints_attribute_by_name(node, "axes");
            int keepdimsInfo = get_node_single_int_attribute_by_name(node, "keepdims", 0);
            if (axesInfo.size() == 2 && axesInfo[0] == 2 && axesInfo[1] == 3 && keepdimsInfo == 1) {
                return OT_Pooling;
            }
            return OT_Reduction;
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
        } else if (inputType == "RNN" || inputType == "GRU" || inputType == "LSTM" ||
            inputType == "Scan") {
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
            return OT_TfSlice;
        } else if (inputType == "ReduceSum" || inputType == "ReduceMin") {
            return OT_Reduction;
        } else if (inputType == "Split") {
            return OT_Slice;
        } else if (inputType == "Splice") {
            return OT_Splice;
        } else if (inputType == "Greater") {
            return OT_Greater;
        } else if (inputType == "Where") {
            return OT_Where;
        } else if (inputType == "SoftPlus") {
            return OT_SoftPlus;
        } else if (inputType == "Exp") {
            return OT_Exp;
        } else if (inputType == "NoOp") {
            return OT_Split;
        } else if (inputType == "Tdnn") {
            return OT_Tdnn;
        } else if (inputType == "Dropout") {
            return OT_Dropout;
        } else if (inputType == "Scale") {
            return OT_Power;
        } else if (inputType == "TopK") {
            return OT_TopK;
        } else if (inputType == "Equal") {
            return OT_Equal;
        } else if (inputType == "Sign") {
            return OT_Sign;
        } else if (inputType == "TFL_HARD_SWISH") {
            return OT_HSwish;
        } else {
            UNI_ERROR_LOG("operator name:%s type:%s not supported.\n", this->node.name().c_str(),
                inputType.c_str());
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
                    result[j] = UNI_MIN(attribute.ints(j), INT_MAX);
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
                U8 *value;
                if (tp.has_raw_data()) {
                    const std::string &rawData = tp.raw_data();
                    value = (U8 *)(rawData.data());
                } else if (tp.data_type() == onnx::TensorProto::FLOAT) {
                    value = (U8 *)(tp.float_data().data());
                } else {
                    UNI_ERROR_LOG("can not process operator name:%s %s type attribute.\n",
                        this->node.name().c_str(), onnx_data_type_string(tp.data_type()).c_str());
                }

                result.resize(tp.dims(0));
                memcpy(result.data(), value, tp.dims(0) * sizeof(float));
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
                return UNI_MIN(attribute.i(), INT_MAX);
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

    std::string onnx_data_type_string(int num)
    {
        const google::protobuf::EnumDescriptor *descriptor =
            onnx::TensorProto::DataType_descriptor();
        return descriptor->FindValueByNumber(num)->name();
    }

    int get_data_size_from_tensor_proto(const onnx::TensorProto &tensorProto)
    {
        int size = 0;
        if (tensorProto.has_raw_data()) {
            const std::string &rawData = tensorProto.raw_data();
            if (tensorProto.data_type() == onnx::TensorProto::BOOL) {
                size = (int)rawData.size() / sizeof(bool);
            } else if (tensorProto.data_type() == onnx::TensorProto::INT64) {
                size = (int)rawData.size() / sizeof(int64_t);
            } else if (tensorProto.data_type() == onnx::TensorProto::INT32) {
                size = (int)rawData.size() / sizeof(int);
            } else if (tensorProto.data_type() == onnx::TensorProto::FLOAT) {
                size = (int)rawData.size() / sizeof(float);
            } else {
                UNI_ERROR_LOG("can not process onnx converter name:%s %s type raw tensor.\n",
                    this->node.name().c_str(),
                    onnx_data_type_string(tensorProto.data_type()).c_str());
            }
        } else if (tensorProto.data_type() == onnx::TensorProto::FLOAT) {
            size = tensorProto.float_data_size();
        } else {
            UNI_ERROR_LOG("can not process operator name:%s %s type tensor.\n",
                this->node.name().c_str(), onnx_data_type_string(tensorProto.data_type()).c_str());
        }
        return size;
    }

    TensorDesc genDescFromTp(const onnx::TensorProto &tp)
    {
        DataType dt;
        if (onnx::TensorProto::FLOAT == tp.data_type() ||
            onnx::TensorProto::DOUBLE == tp.data_type()) {
            dt = DT_F32;
        } else if (onnx::TensorProto::INT64 == tp.data_type() ||
            onnx::TensorProto::INT32 == tp.data_type()) {
            dt = DT_I32;
        } else if (onnx::TensorProto::FLOAT16 == tp.data_type()) {
            dt = DT_F16;
        } else {
            UNI_ERROR_LOG("can not process operator name:%s %s type tensor desc.\n",
                this->node.name().c_str(), onnx_data_type_string(tp.data_type()).c_str());
        }
        TensorDesc desc;
        desc.nDims = tp.dims_size();
        desc.dt = dt;
        desc.df = getTensorDefaultDataFormat(desc.nDims);
        for (U32 j = 0; j < desc.nDims; j++) {
            desc.dims[desc.nDims - 1 - j] = tp.dims(j);
        }
        return desc;
    }

    bool *get_bool_ptr_from_tensor_proto(const onnx::TensorProto &tensorProto)
    {
        bool *ptr = nullptr;
        if (tensorProto.has_raw_data()) {
            const std::string &rawData = tensorProto.raw_data();
            ptr = (bool *)rawData.data();
        } else {
            UNI_ERROR_LOG("can not process operator name:%s %s type non-raw bool tensor.\n",
                this->node.name().c_str(), onnx_data_type_string(tensorProto.data_type()).c_str());
        }
        return ptr;
    }

    U8 *get_ptr_from_weight_obj(const onnx::TensorProto &tensorProto)
    {
        U8 *ptr = nullptr;
        if (tensorProto.has_raw_data()) {
            const std::string &rawData = tensorProto.raw_data();
            ptr = (U8 *)rawData.data();
        } else if (tensorProto.data_type() == onnx::TensorProto::FLOAT) {
            ptr = (U8 *)tensorProto.float_data().data();
        } else {
            UNI_ERROR_LOG("can not process operator name:%s %s type weight.\n",
                this->node.name().c_str(), onnx_data_type_string(tensorProto.data_type()).c_str());
        }
        return ptr;
    }

    std::vector<int> get_int_vec_from_tensorProto(const onnx::TensorProto &tp)
    {
        int size = 0;
        std::vector<int> shape;

        if (tp.data_type() == onnx::TensorProto::INT64 ||
            tp.data_type() == onnx::TensorProto::UNDEFINED) {
            U8 *shapeData = 0;
            if (tp.has_raw_data()) {
                shapeData = (U8 *)tp.raw_data().data();
                size = tp.raw_data().size() / 8;
            } else {
                shapeData = (U8 *)tp.int64_data().data();
                size = tp.int64_data_size();
            }
            shape.resize(size);
            for (int j = 0; j < size; j++) {
                int64_t value;
                memcpy(&value, shapeData + j * sizeof(int64_t), sizeof(int64_t));
                shape[j] = UNI_MIN(value, INT_MAX);
            }
        } else if (tp.data_type() == onnx::TensorProto::INT32) {
            U8 *shapeData = nullptr;
            if (tp.has_raw_data()) {
                shapeData = (U8 *)tp.raw_data().data();
                size = tp.raw_data().size() / 4;
            } else {
                shapeData = (U8 *)tp.int32_data().data();
                size = tp.int32_data_size();
            }
            shape.resize(size);
            memcpy(shape.data(), shapeData, sizeof(int32_t) * size);
        } else {
            UNI_ERROR_LOG("can not process operator name:%s %s type tensor.\n",
                this->node.name().c_str(), onnx_data_type_string(tp.data_type()).c_str());
        }
        return shape;
    }

    float getSinFloat_from_tensorProto(const onnx::TensorProto &tp)
    {
        float value = 0;
        int exponentSize = get_data_size_from_tensor_proto(tp);
        if (tp.data_type() != onnx::TensorProto::FLOAT || exponentSize != 1) {
            UNI_ERROR_LOG("can not process operator name:%s %d-%s type tensor.\n",
                this->node.name().c_str(), exponentSize,
                onnx_data_type_string(tp.data_type()).c_str());
        } else {
            if (tp.has_raw_data()) {
                const std::string &raw_data = tp.raw_data();
                memcpy(&value, raw_data.data(), sizeof(float));
            } else {
                memcpy(&value, tp.float_data().data(), sizeof(float));
            }
        }
        return value;
    }

    void memcpy_trans2d(void *dest, void *src, int N, int K)
    {
        for (int r = 0, index = 0; r < N; r++) {
            for (int c = 0; c < K; c++, index += sizeof(float)) {
                memcpy((U8 *)dest + index, (U8 *)src + (c * N + r) * sizeof(float), sizeof(float));
            }
        }
    }

    bool is_multi_dim(onnx::TensorProto &wps)
    {
        int multidim = 0;
        for (int idx = 0; idx < wps.dims_size(); ++idx) {
            if (wps.dims(idx) > 1) {
                ++multidim;
            }
        }
        return (multidim > 1);
    }

    EE parse_file(std::string dir, std::string mfn) override
    {
        std::string onnxSuffix = ".onnx";
        std::string onnxPath = dir + "/" + mfn + onnxSuffix;

        this->modelName = mfn;

        EE ret = read_from_onnx_file(onnxPath.c_str(), (google::protobuf::Message *)(&onnxModel));
        if (ret != SUCCESS) {
            UNI_ERROR_LOG("can not read onnx model file %s.\n", onnxPath.c_str());
        }

        onnxGraph = onnxModel.graph();

        for (int i = 0; i < onnxGraph.initializer_size(); i++) {
            const onnx::TensorProto &initializer = onnxGraph.initializer(i);
            weights[initializer.name()] = initializer;
        }
        return ret;
    }

    std::string crop_name(const std::string &name)
    {
        std::string ret;
        if (name.length() < NAME_LEN) {
            ret = name;
        } else if (this->croppingNames.find(name) != this->croppingNames.end()) {
            ret = this->croppingNames[name];
        } else {
            ret = "brief_" + std::to_string(this->croppingNames.size());
            this->croppingNames[name] = ret;
        }
        return ret;
    }

    EE adapt_operators(ModelSpec *ms) override
    {
        EE ret = SUCCESS;
        str_copy(ms->model_name, modelName.c_str(), modelName.length());
        ms->dt = DT_F32;

        ms->num_inputs = 0;
        for (int i = 0; i < onnxGraph.input().size(); i++) {
            auto input_node = onnxGraph.input(i);
            auto input_name = input_node.name();
            if (weights.find(input_name) != weights.end()) {
                continue;
            }
            ms->num_inputs++;
        }
        ms->input_names = (I8 **)mt_new_storage(ms->num_inputs * sizeof(I8 *));
        ms->input_dims = (TensorDesc *)mt_new_storage(sizeof(TensorDesc) * ms->num_inputs);
        for (int i = 0, index = 0; i < onnxGraph.input().size(); i++) {
            auto input_node = onnxGraph.input(i);
            auto input_name = input_node.name();
            if (weights.find(input_name) != weights.end()) {
                continue;
            }
            ms->input_names[index] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            str_copy(ms->input_names[index], input_name.c_str(), input_name.length());

            auto type = input_node.type().tensor_type().elem_type();
            switch (type) {
                case onnx::TensorProto::INT64:
                case onnx::TensorProto::INT32:
                    ms->input_dims[index].dt = DT_I32;
                    break;
                case onnx::TensorProto::UINT64:
                case onnx::TensorProto::UINT32:
                    ms->input_dims[index].dt = DT_U32;
                    break;
                case onnx::TensorProto::DOUBLE:
                case onnx::TensorProto::FLOAT:
                case onnx::TensorProto::FLOAT16:
                case onnx::TensorProto::BFLOAT16:
                    ms->input_dims[index].dt = DT_F32;
                    break;
                default:
                    UNI_ERROR_LOG(
                        "can not process %s type input.\n", onnx_data_type_string(type).c_str());
                    break;
            }
            ms->input_dims[index].nDims = input_node.type().tensor_type().shape().dim().size();
            ms->input_dims[index].df = getTensorDefaultDataFormat(ms->input_dims[index].nDims);
            for (U32 j = 0; j < ms->input_dims[index].nDims; j++) {
                ms->input_dims[index].dims[ms->input_dims[index].nDims - 1 - j] =
                    input_node.type().tensor_type().shape().dim(j).dim_value();
            }
            // batch must > 0
            for (U32 j = 0; j < ms->input_dims[index].nDims; j++) {
                if (ms->input_dims[index].dims[j] == 0) {
                    ms->input_dims[index].dims[j] = 1;
                }
            }
            index++;
        }

        ms->num_outputs = onnxGraph.output().size();
        ms->output_names = (I8 **)mt_new_storage(ms->num_outputs * sizeof(I8 *));
        for (int k = 0; k < onnxGraph.output().size(); k++) {
            ms->output_names[k] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            str_copy(ms->output_names[k], onnxGraph.output(k).name().c_str(),
                onnxGraph.output(k).name().length());
        }

        int bnOpNum = 0;
        int constantOpNum = 0;
        int onnxNodeCount = onnxGraph.node_size();
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

        // appending space for scale op
        ms->num_operator_specs = onnxNodeCount + bnOpNum - constantOpNum - removePreprocessOpNum;
        this->operatorSpecVec = std::vector<OperatorSpec>(ms->num_operator_specs);

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
            UNI_INFO_LOG("first %d operators are removed, and %d of them are Constant operator.\n",
                removePreprocessOpNum, numUnseenConstants);
        }

        nodeIndex = removePreprocessOpNum;
        int opIndex = 0;
        for (int i = removePreprocessOpNum; i < onnxNodeCount; i++) {
            this->node = onnxGraph.node(nodeIndex);
            UNI_DEBUG_LOG("process operator name:%s parameter.\n", this->node.name().c_str());
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
            opName = this->crop_name(opName);

            int opInputNum = (int)node.input_size();
            opFinalInputNum = opInputNum;
            std::vector<std::string> inputNames;
            std::vector<std::string> op_weight_objs;
            for (int j = 0; j < opInputNum; j++) {
                const std::string &input_name = node.input(j);
                if (weights.find(input_name) != weights.end()) {
                    opFinalInputNum--;
                    op_weight_objs.push_back(input_name);
                } else if (input_name == "") {
                    opFinalInputNum--;
                } else {
                    inputNames.push_back(input_name);
                    if (op == "Max" || op == "Min") {
                        opFinalInputNum = 1;
                        break;
                    }
                }
            }
            // op input names correction
            if (op == "Add" || op == "Mul") {
                if (weights.find(node.input(1)) != weights.end() &&
                    is_multi_dim(weights[node.input(1)])) {
                    opFinalInputNum++;
                    inputNames.push_back(node.input(1));
                }
            }

            if (op == "Concat" &&  node.input_size() > 1 && weights.find(node.input(1)) != weights.end()) {
                opFinalInputNum++;
                inputNames.push_back(node.input(1));
            }

            int opOutputNum = (int)node.output_size();
            std::vector<std::string> outputNames;
            for (int j = 0; j < opOutputNum; j++) {
                const std::string &output_name = node.output(j);
                outputNames.push_back(output_name);
            }

            str_copy(operatorSpecVec[opIndex].name, opName.c_str(), opName.length());
            OperatorType opType = convert_onnx_type(op);

            // op type correction
            if (op == "MatMul" && opFinalInputNum == 1) {
                opType = OT_FC;
            } else if ((op == "Add" || op == "Sub" || op == "Mul" || op == "Div")) {
                if (opFinalInputNum == 1 && node.input_size() == 2) {
                    onnx::TensorProto tp;
                    if (weights.find(node.input(0)) != weights.end()) {
                        tp = weights[node.input(0)];
                    } else if (weights.find(node.input(1)) != weights.end()) {
                        tp = weights[node.input(1)];
                    } else {
                        UNI_ERROR_LOG("can not map operator name:%s type:%s to Power.\n",
                            this->node.name().c_str(), op.c_str());
                    }
                    if (get_data_size_from_tensor_proto(tp) == 1) {
                        opType = OT_Power;
                        operatorSpecVec[opIndex].type = OT_Power;
                    }
                } else if (opFinalInputNum == 2 && weights.find(node.input(1)) != weights.end() &&
                    is_multi_dim(weights[node.input(1)])) {
                    opType = OT_Eltwise;
                    operatorSpecVec[opIndex].type = OT_Eltwise;
                }
            }

            // input names order correction
            if (op == "Scan" || op == "Gather") {
                for (int k = 0; k < (int)(inputNames.size() / 2); k++) {
                    std::string frontStr = inputNames[k];
                    inputNames[k] = inputNames[inputNames.size() - 1 - k];
                    inputNames[inputNames.size() - 1 - k] = frontStr;
                }
                if (op == "Scan") {
                    if (outputNames.size() >= 2) {
                        std::string firOutput = outputNames[0];
                        std::string lastOutput = outputNames[outputNames.size() - 1];
                        outputNames.clear();
                        outputNames.push_back(lastOutput);
                        outputNames.push_back(firOutput);
                    }
                    opOutputNum = outputNames.size();
                }
            }

            operatorSpecVec[opIndex].type = opType;
            operatorSpecVec[opIndex].num_inputs = opFinalInputNum;
            operatorSpecVec[opIndex].input_tensors_name =
                (I8 **)mt_new_storage(operatorSpecVec[opIndex].num_inputs * sizeof(I8 *));
            for (U32 j = 0; j < operatorSpecVec[opIndex].num_inputs; j++) {
                inputNames[j] = this->crop_name(inputNames[j]);
                operatorSpecVec[opIndex].input_tensors_name[j] =
                    (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(operatorSpecVec[opIndex].input_tensors_name[j], inputNames[j].c_str(),
                    inputNames[j].length());
            }
            operatorSpecVec[opIndex].num_outputs = opOutputNum;
            operatorSpecVec[opIndex].output_tensors_name =
                (I8 **)mt_new_storage(operatorSpecVec[opIndex].num_outputs * sizeof(I8 *));
            for (U32 j = 0; j < operatorSpecVec[opIndex].num_outputs; j++) {
                outputNames[j] = this->crop_name(outputNames[j]);
                operatorSpecVec[opIndex].output_tensors_name[j] =
                    (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(operatorSpecVec[opIndex].output_tensors_name[j], outputNames[j].c_str(),
                    outputNames[j].length());
            }

            if (op == "Equal" || op == "Linear") {
                weightOpIndexLists.push_back(nodeIndex);
            }

            if ((op == "Add" || op == "Sub" || op == "Mul" || op == "Div") &&
                opFinalInputNum == 1 && opType != OT_Power) {
                weightOpIndexLists.push_back(nodeIndex);
                operatorSpecVec[opIndex].type = OT_Scale;
                opType = OT_Scale;
                memset(&(operatorSpecVec[opIndex].ps), 0, sizeof(operatorSpecVec[opIndex].ps));
                operatorSpecVec[opIndex].ps.scale_spec.axis = 1;
            }

            if (op == "Transpose" && opFinalInputNum == 0) {
                weightOpIndexLists.push_back(nodeIndex);
            } else {
                if (op == "Gather") {
                    if (node.input_size() == 2 && weights.find(node.input(0)) == weights.end() &&
                        weights.find(node.input(1)) == weights.end()) {
                        operatorSpecVec[opIndex].type = OT_Embedding;
                        opType = OT_Embedding;
                    } else if (weights.find(node.input(0)) != weights.end()) {
                        weightOpIndexLists.push_back(nodeIndex);
                        if (weights.find(node.input(1)) != weights.end()) {  // both provided
                            operatorSpecVec[opIndex].type = OT_SharedWeight;
                            opType = OT_SharedWeight;
                        } else {
                            operatorSpecVec[opIndex].type = OT_Embedding;
                            opType = OT_Embedding;
                        }
                    } else if (weights.find(node.input(1)) != weights.end()) {
                        opType = OT_Slice;
                        operatorSpecVec[opIndex].type = OT_Slice;
                        memset(
                            &(operatorSpecVec[opIndex].ps), 0, sizeof(operatorSpecVec[opIndex].ps));
                        operatorSpecVec[opIndex].ps.slice_spec.slice_points[0] = 1;
                        operatorSpecVec[opIndex].ps.slice_spec.slice_size = 1;
                        operatorSpecVec[opIndex].ps.slice_spec.axis = 1;
                        operatorSpecVec[opIndex].num_outputs = 2;
                        free(operatorSpecVec[opIndex].output_tensors_name[0]);
                        free(operatorSpecVec[opIndex].output_tensors_name);
                        operatorSpecVec[opIndex].output_tensors_name = (I8 **)mt_new_storage(
                            operatorSpecVec[opIndex].num_outputs * sizeof(I8 *));
                        operatorSpecVec[opIndex].output_tensors_name[0] =
                            (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                        str_copy(operatorSpecVec[opIndex].output_tensors_name[0],
                            outputNames[0].c_str(), outputNames[0].length());
                        operatorSpecVec[opIndex].output_tensors_name[1] =
                            (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                        std::string reduntStr = "DropOut_Str";
                        str_copy(operatorSpecVec[opIndex].output_tensors_name[1], reduntStr.c_str(),
                            reduntStr.length());
                    }
                }

                ParameterSpec curPs;
                CHECK_STATUS(adapt_operator(opType, &curPs));
                operatorSpecVec[opIndex].ps = curPs;

                if (opType == OT_BatchNorm && this->op == "BatchNormalization") {
                    std::string scaleInputName = outputNames[0];
                    std::string scaleOpName = "scale_" + opName;
                    opIndex++;
                    str_copy(
                        operatorSpecVec[opIndex].name, scaleOpName.c_str(), scaleOpName.length());
                    operatorSpecVec[opIndex].type = OT_Scale;
                    operatorSpecVec[opIndex].num_inputs = 1;
                    operatorSpecVec[opIndex].input_tensors_name =
                        (I8 **)mt_new_storage(sizeof(I8 *));
                    operatorSpecVec[opIndex].input_tensors_name[0] =
                        (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                    str_copy(operatorSpecVec[opIndex].input_tensors_name[0], scaleInputName.c_str(),
                        scaleInputName.length());
                    operatorSpecVec[opIndex].num_outputs = 1;
                    operatorSpecVec[opIndex].output_tensors_name =
                        (I8 **)mt_new_storage(sizeof(I8 *));
                    operatorSpecVec[opIndex].output_tensors_name[0] =
                        (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                    str_copy(operatorSpecVec[opIndex].output_tensors_name[0],
                        scaleInputName.c_str(), scaleInputName.length());

                    ParameterSpec scalePs;
                    CHECK_STATUS(adapt_operator(operatorSpecVec[opIndex].type, &scalePs));
                    operatorSpecVec[opIndex].ps = scalePs;
                }
            }

            nodeIndex++;
            opIndex++;
        }

        // insert sharedWeightOp
        ms->num_weight_specs = weightOpIndexLists.size() + bnOpNum + insertSharedWeight.size();
        this->weightSpecVec = std::vector<WeightSpec>(weightOpIndexLists.size() + bnOpNum);
        for (auto iter = insertSharedWeight.begin(); iter != insertSharedWeight.end(); iter++) {
            OperatorSpec tmpOps = mt_create_operator(iter->first.c_str(), OT_SharedWeight, 0, 1);
            str_copy(tmpOps.output_tensors_name[0], iter->first.c_str(), iter->first.length());
            SharedWeightParamSpec sharedWeightPs;
            const auto &weightTp = weights[iter->first];
            std::vector<int> weightShape;
            for (int k = 0; k < weightTp.dims_size(); k++) {
                weightShape.push_back(weightTp.dims(k));
            }
            sharedWeightPs.desc.dt = DT_F32;
            sharedWeightPs.desc.nDims = UNI_MAX(1, weightShape.size());
            sharedWeightPs.desc.df = getTensorDefaultDataFormat(sharedWeightPs.desc.nDims);
            sharedWeightPs.desc.dims[0] = 1;
            for (U32 j = 0; j < sharedWeightPs.desc.nDims; j++) {
                sharedWeightPs.desc.dims[sharedWeightPs.desc.nDims - 1 - j] = weightShape[j];
            }
            WeightSpec weightSpec = mt_create_weight(
                iter->first.c_str(), DT_F32, tensorNumBytes(sharedWeightPs.desc), 0, 0);
            U8 *weightData = get_ptr_from_weight_obj(weightTp);
            // memcpy(weightSpec.weight, weightData, tensorNumBytes(sharedWeightPs.desc));
            if (weightTp.data_type() == onnx::TensorProto::INT64) {
                std::vector<int> intVec = get_int_vec_from_tensorProto(weightTp);
                std::vector<float> floatVec;
                for (U32 k = 0; k < intVec.size(); k++) {
                    floatVec.push_back(intVec[k] * 1.0);
                }
                memcpy(
                    weightSpec.weight, (U8 *)(&(floatVec[0])), tensorNumBytes(sharedWeightPs.desc));
            } else {
                memcpy(weightSpec.weight, weightData, tensorNumBytes(sharedWeightPs.desc));
            }
            tmpOps.ps.shared_weight_spec = sharedWeightPs;
            this->sharedWeightOps.push_back(tmpOps);
            this->sharedWeightWps.push_back(weightSpec);
        }
        ms->num_operator_specs = this->sharedWeightOps.size() + this->operatorSpecVec.size();
        ms->ops = (OperatorSpec *)mt_new_storage(sizeof(OperatorSpec) * ms->num_operator_specs);
        std::vector<OperatorSpec> combineOpsVec;
        combineOpsVec.insert(combineOpsVec.end(), sharedWeightOps.begin(), sharedWeightOps.end());
        combineOpsVec.insert(combineOpsVec.end(), operatorSpecVec.begin(), operatorSpecVec.end());
        memcpy(ms->ops, combineOpsVec.data(), sizeof(OperatorSpec) * ms->num_operator_specs);
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            ms->ops[i].tensor_positions = nullptr;
            ms->ops[i].num_quant_feature = 0;
            ms->ops[i].feature_scale = nullptr;
        }
        return ret;
    }

    EE adapt_weights(ModelSpec *ms) override
    {
        EE ret = SUCCESS;
        // WeightSpec *weightSpecVec = (WeightSpec *)mt_new_storage(sizeof(WeightSpec) * ms->num_weight_specs);
        int weightOpIndexIndeed = 0;
        for (U32 i = 0; i < (U32)(ms->num_weight_specs - this->insertSharedWeight.size()); i++) {
            int weightOpIndex = weightOpIndexLists[weightOpIndexIndeed];
            this->node = onnxGraph.node(weightOpIndex);
            UNI_DEBUG_LOG("process operator name:%s weight.\n", this->node.name().c_str());
            const onnx::NodeProto &weightNode = onnxGraph.node(weightOpIndex);
            std::string weightOpName = weightNode.name();
            if (weightOpName.empty()) {
                weightOpName = weightNode.output(0);
            }
            if (croppingNames.find(weightOpName) != croppingNames.end()) {
                weightOpName = croppingNames[weightOpName];
            }
            const std::string &weightOpType = weightNode.op_type();
            auto indices = getOperatorWeightInputIndex(weightOpIndex);

            if (weightOpType == "Conv" || weightOpType == "ConvTranspose") {
                // to check that if any op has bias
                int convInputNum =
                    weightNode.input_size();  // if convInputNum == 3, means has bias , otherwise , do not have bias

                const onnx::TensorProto &convWeightTp = weights[weightNode.input(1)];

                int convWeightNum = get_data_size_from_tensor_proto(convWeightTp);
                U8 *convWeightParamPtr = get_ptr_from_weight_obj(convWeightTp);
                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());

                weightSpecVec[i].bytes_of_weight =
                    convWeightNum * sizeof(float);  // Please do not change to bytesOf(mdt)
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                memcpy(
                    weightSpecVec[i].weight, convWeightParamPtr, weightSpecVec[i].bytes_of_weight);
                // traverse weight elements to see whether it is bnn convolution
                weightSpecVec[i].mdt =
                    get_weight_data_type(convWeightNum, (F32 *)weightSpecVec[i].weight);

                int convBiasNum = 0;
                U8 *convBiasParamPtr = nullptr;
                if (convInputNum == 3) {
                    const onnx::TensorProto &convBiasTp = weights[weightNode.input(2)];
                    convBiasNum = get_data_size_from_tensor_proto(convBiasTp);
                    convBiasParamPtr = get_ptr_from_weight_obj(convBiasTp);
                    weightSpecVec[i].bytes_of_vec = convBiasNum * sizeof(float);
                    if (DT_BIN11 == weightSpecVec[i].mdt || DT_BIN01 == weightSpecVec[i].mdt) {
                        weightSpecVec[i].bytes_of_vec *=
                            2;  // BNN conv must have a scale vector and a bias vector, so that it can fuse with BN
                    }
                    weightSpecVec[i].vec = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_vec);
                    if (DT_BIN11 == weightSpecVec[i].mdt || DT_BIN01 == weightSpecVec[i].mdt) {
                        U32 vecBytes = convBiasNum * sizeof(float);
                        F32 *scale = (F32 *)weightSpecVec[i].vec;
                        for (I32 j = 0; j < convBiasNum; j++) {
                            scale[j] = 1.0;
                        }
                        memcpy(weightSpecVec[i].vec + vecBytes, convBiasParamPtr,
                            vecBytes);  // Copy bias (if any) to the second half for BNN
                    } else {
                        memcpy(
                            weightSpecVec[i].vec, convBiasParamPtr, weightSpecVec[i].bytes_of_vec);
                    }
                } else {
                    weightSpecVec[i].bytes_of_vec = 0;
                    weightSpecVec[i].vec = nullptr;
                }
            } else if (weightOpType == "Gemm" || weightOpType == "Linear") {
                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].mdt = DT_F32;
                int transB = 1;
                if (weightOpType == "Gemm") {
                    const onnx::TensorProto &fcBiasTp = weights[weightNode.input(2)];
                    int fcBiasNum = get_data_size_from_tensor_proto(fcBiasTp);
                    U8 *fcBiasParamPtr = get_ptr_from_weight_obj(fcBiasTp);
                    weightSpecVec[i].bytes_of_vec = fcBiasNum * sizeof(float);
                    weightSpecVec[i].vec = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_vec);
                    memcpy(weightSpecVec[i].vec, fcBiasParamPtr, weightSpecVec[i].bytes_of_vec);
                    transB = get_node_single_int_attribute_by_name(weightNode, "transB", 0);
                } else {
                    weightSpecVec[i].bytes_of_vec = 0;
                    weightSpecVec[i].vec = nullptr;
                }
                const onnx::TensorProto &fcWeightTp = weights[weightNode.input(1)];
                int fcWeightNum = get_data_size_from_tensor_proto(fcWeightTp);
                U8 *fcWeightParamPtr = get_ptr_from_weight_obj(fcWeightTp);
                weightSpecVec[i].bytes_of_weight = fcWeightNum * sizeof(float);
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                if (transB) {
                    memcpy(weightSpecVec[i].weight, fcWeightParamPtr, fcWeightNum * sizeof(float));
                } else {
                    memcpy_trans2d(weightSpecVec[i].weight, fcWeightParamPtr,
                        (int)fcWeightTp.dims(1), (int)fcWeightTp.dims(0));
                }
            } else if (weightOpType == "BatchNormalization") {
                const onnx::TensorProto &scale = weights[weightNode.input(1)];
                const onnx::TensorProto &bias = weights[weightNode.input(2)];
                const onnx::TensorProto &mean = weights[weightNode.input(3)];
                const onnx::TensorProto &var = weights[weightNode.input(4)];

                U8 *meanPtr = get_ptr_from_weight_obj(mean);
                int bnMeanNum = get_data_size_from_tensor_proto(mean);
                U8 *varPtr = get_ptr_from_weight_obj(var);
                int bnVarNum = get_data_size_from_tensor_proto(var);

                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].mdt = DT_F32;
                weightSpecVec[i].bytes_of_weight = bnMeanNum * sizeof(float);
                weightSpecVec[i].bytes_of_vec = bnVarNum * sizeof(float);

                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                memcpy(weightSpecVec[i].weight, meanPtr, weightSpecVec[i].bytes_of_weight);
                weightSpecVec[i].vec = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_vec);
                memcpy(weightSpecVec[i].vec, varPtr, weightSpecVec[i].bytes_of_vec);

                // for scale
                std::string scaleWeightOpName = "scale_" + weightOpName;
                i = i + 1;
                U8 *scalePtr = get_ptr_from_weight_obj(scale);
                int scaleWeightNum = get_data_size_from_tensor_proto(scale);
                U8 *biasPtr = get_ptr_from_weight_obj(bias);
                int scaleBiasNum = get_data_size_from_tensor_proto(bias);

                str_copy(weightSpecVec[i].op_name, scaleWeightOpName.c_str(),
                    scaleWeightOpName.length());
                weightSpecVec[i].mdt = DT_F32;
                weightSpecVec[i].bytes_of_weight = scaleWeightNum * sizeof(float);
                weightSpecVec[i].bytes_of_vec = scaleBiasNum * sizeof(float);

                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                memcpy(weightSpecVec[i].weight, scalePtr, weightSpecVec[i].bytes_of_weight);
                weightSpecVec[i].vec = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_vec);
                memcpy(weightSpecVec[i].vec, biasPtr, weightSpecVec[i].bytes_of_vec);
            } else if (weightOpType == "BatchNorm") {
                const onnx::TensorProto &mean = weights[weightNode.input(1)];
                const onnx::TensorProto &var = weights[weightNode.input(2)];
                U8 *meanPtr = get_ptr_from_weight_obj(mean);
                int bnMeanNum = get_data_size_from_tensor_proto(mean);
                U8 *varPtr = get_ptr_from_weight_obj(var);
                int bnVarNum = get_data_size_from_tensor_proto(var);
                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].mdt = DT_F32;
                weightSpecVec[i].bytes_of_weight = bnMeanNum * sizeof(float);
                weightSpecVec[i].bytes_of_vec = bnVarNum * sizeof(float);
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                memcpy(weightSpecVec[i].weight, meanPtr, weightSpecVec[i].bytes_of_weight);
                weightSpecVec[i].vec = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_vec);
                memcpy(weightSpecVec[i].vec, varPtr, weightSpecVec[i].bytes_of_vec);
            } else if (weightOpType == "Tdnn") {
                const onnx::TensorProto &weight = weights[weightNode.input(2)];
                const onnx::TensorProto &bias = weights[weightNode.input(3)];
                U8 *weightPtr = get_ptr_from_weight_obj(weight);
                int weightNum = get_data_size_from_tensor_proto(weight);
                U8 *biasPtr = get_ptr_from_weight_obj(bias);
                int biasNum = get_data_size_from_tensor_proto(bias);
                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].mdt = DT_F32;
                weightSpecVec[i].bytes_of_weight = weightNum * sizeof(float);
                weightSpecVec[i].bytes_of_vec = biasNum * sizeof(float);
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                memcpy(weightSpecVec[i].weight, weightPtr, weightSpecVec[i].bytes_of_weight);
                weightSpecVec[i].vec = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_vec);
                memcpy(weightSpecVec[i].vec, biasPtr, weightSpecVec[i].bytes_of_vec);
            } else if (weightOpType == "Add") {
                CHECK_REQUIREMENT(0 != indices.size());
                const onnx::TensorProto &bias = weights[weightNode.input(indices[0])];
                U8 *bias_ptr = get_ptr_from_weight_obj(bias);
                int bias_num = get_data_size_from_tensor_proto(bias);

                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].mdt = DT_F32;
                weightSpecVec[i].bytes_of_weight = 0;
                weightSpecVec[i].weight = nullptr;
                weightSpecVec[i].bytes_of_vec = bias_num * sizeof(float);
                weightSpecVec[i].vec = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_vec);
                memcpy(weightSpecVec[i].vec, bias_ptr, weightSpecVec[i].bytes_of_vec);
            } else if (weightOpType == "Mul") {
                CHECK_REQUIREMENT(0 != indices.size());
                const onnx::TensorProto &weight = weights[weightNode.input(indices[0])];
                U8 *weight_ptr = get_ptr_from_weight_obj(weight);
                int weight_num = get_data_size_from_tensor_proto(weight);

                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].mdt = DT_F32;
                weightSpecVec[i].bytes_of_weight = weight_num * sizeof(float);
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                memcpy(weightSpecVec[i].weight, weight_ptr, weightSpecVec[i].bytes_of_weight);
                weightSpecVec[i].bytes_of_vec = 0;
                weightSpecVec[i].vec = nullptr;
            } else if (weightOpType == "MatMul" || weightOpType == "PRelu") {
                const onnx::TensorProto &weight = weights[weightNode.input(1)];
                U8 *weight_ptr = get_ptr_from_weight_obj(weight);
                int weight_num = get_data_size_from_tensor_proto(weight);

                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].mdt = DT_F32;
                weightSpecVec[i].bytes_of_weight = weight_num * sizeof(float);
                weightSpecVec[i].bytes_of_vec = 0;
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                int row = weight.dims(0);
                int column = weight.dims(1);
                for (int m = 0, index = 0; m < column; m++) {
                    for (int n = 0; n < row; n++, index += sizeof(float)) {
                        memcpy(weightSpecVec[i].weight + index,
                            weight_ptr + (n * column + m) * sizeof(float), sizeof(float));
                    }
                }
                weightSpecVec[i].vec = nullptr;
            } else if (weightOpType == "Div") {
                const onnx::TensorProto &weight = weights[weightNode.input(1)];
                U8 *weight_ptr = get_ptr_from_weight_obj(weight);
                int weight_num = get_data_size_from_tensor_proto(weight);

                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].mdt = DT_F32;
                weightSpecVec[i].bytes_of_weight = weight_num * sizeof(float);
                weightSpecVec[i].bytes_of_vec = 0;
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                F32 *scale = (F32 *)weightSpecVec[i].weight;
                memcpy(scale, weight_ptr, weightSpecVec[i].bytes_of_weight);
                for (int j = 0; j < weight_num; j++) {
                    scale[j] = 1 / scale[j];
                }
                weightSpecVec[i].vec = nullptr;
            } else if (weightOpType == "Transpose") {
                const onnx::TensorProto &weight = weights[weightNode.input(0)];
                U8 *weight_ptr = get_ptr_from_weight_obj(weight);
                int weight_num = get_data_size_from_tensor_proto(weight);

                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].mdt = DT_F32;
                weightSpecVec[i].bytes_of_weight = weight_num * sizeof(float);
                // For the time being, use bytes_of_vec to record the horizontal length of weight
                weightSpecVec[i].bytes_of_vec = weight.dims(0);
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                memcpy(weightSpecVec[i].weight, weight_ptr, weightSpecVec[i].bytes_of_weight);
                weightSpecVec[i].vec = nullptr;
            } else if (weightOpType == "RNN" || weightOpType == "GRU" || weightOpType == "LSTM") {
                const onnx::TensorProto &W = weights[weightNode.input(1)];
                const onnx::TensorProto &R = weights[weightNode.input(2)];
                const onnx::TensorProto &B = weights[weightNode.input(3)];
                if (W.dims_size() != 3 || R.dims_size() != 3) {
                    UNI_ERROR_LOG("can not process operator name:%s type:%s W.dims_size() != "
                                  "3 || R.dims_size() != 3.\n",
                        this->node.name().c_str(), weightOpType.c_str());
                }
                if (W.dims(0) != R.dims(0) || W.dims(1) != R.dims(1)) {
                    UNI_ERROR_LOG("can not process operator name:%s type:%s W.dims(0) != "
                                  "R.dims(0) || W.dims(1) != "
                                  "R.dims(1).\n",
                        this->node.name().c_str(), weightOpType.c_str());
                }
                int biasNum = (B.dims_size() == 0) ? 0 : 1;
                for (int j = 0; j < B.dims_size(); j++) {
                    biasNum *= B.dims(j);
                }
                // reorganize bias
                if (biasNum % 2 != 0) {
                    UNI_ERROR_LOG("can not process operator name:%s type:%s bias.\n",
                        this->node.name().c_str(), weightOpType.c_str());
                }
                biasNum /= 2;
                int gates = 0;
                std::vector<int> order;
                bool gru_lbr = false;
                if (weightOpType == "RNN") {
                    gates = 1;
                    order = {0};
                } else if (weightOpType == "GRU") {
                    gates = 3;
                    order = {0, 1, 2};
                    if (0 !=
                        get_node_single_int_attribute_by_name(weightNode, "linear_before_reset", 0)) {
                        gru_lbr = true;
                        biasNum += biasNum / gates;
                    }
                } else if (weightOpType == "LSTM") {
                    gates = 4;
                    order = {0, 3, 2, 1};
                } else {
                    UNI_ERROR_LOG("can not process operator name:%s type:%s.\n",
                        this->node.name().c_str(), weightOpType.c_str());
                }
                U8 *W_ptr = get_ptr_from_weight_obj(W);
                U8 *R_ptr = get_ptr_from_weight_obj(R);
                U8 *B_ptr = get_ptr_from_weight_obj(B);
                weightSpecVec[i].mdt = DT_F32;
                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].bytes_of_weight =
                    (W.dims(0) * W.dims(1) * (W.dims(2) + R.dims(2))) * sizeof(float);
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                weightSpecVec[i].bytes_of_vec = biasNum * sizeof(float);
                weightSpecVec[i].vec = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_vec);
                int hidden = W.dims(1) / gates;
                U8 *weightPtr = weightSpecVec[i].weight;
                F32 *biasPtr = (F32 *)weightSpecVec[i].vec;
                // loop direction
                for (int j = 0; j < W.dims(0); j++) {
                    // loop LSTM(iofc), GRU(zrh), RNN(g)
                    for (int m = 0; m < gates; m++) {
                        int k = order[m];
                        for (int n = 0; n < hidden; n++) {
                            memcpy(weightPtr,
                                W_ptr + ((j * gates + k) * hidden + n) * W.dims(2) * sizeof(float),
                                W.dims(2) * sizeof(float));
                            weightPtr += W.dims(2) * sizeof(float);
                            memcpy(weightPtr,
                                R_ptr + ((j * gates + k) * hidden + n) * R.dims(2) * sizeof(float),
                                R.dims(2) * sizeof(float));
                            weightPtr += R.dims(2) * sizeof(float);

                            if (biasNum > 0) {
                                float W_B, R_B;
                                memcpy(&W_B,
                                    B_ptr + (((j * 2) * gates + k) * hidden + n) * sizeof(float),
                                    sizeof(float));
                                memcpy(&R_B,
                                    B_ptr + (((j * 2 + 1) * gates + k) * hidden + n) * sizeof(float),
                                    sizeof(float));
                                // not to preprocess LBR GRU's h gates bias
                                if (gru_lbr && m == gates - 1) {
                                    *biasPtr = W_B;
                                    *(biasPtr + hidden) = R_B;
                                } else {
                                    *biasPtr = W_B + R_B;
                                }
                                biasPtr++;
                            }
                        }
                    }
                    if (gru_lbr) {
                        biasPtr += hidden;
                    }
                }
            } else if (weightOpType == "Gather") {
                auto weightTp = weights[weightNode.input(0)];
                int weightNum = get_data_size_from_tensor_proto(weightTp);
                U8 *weightParamPtr = get_ptr_from_weight_obj(weightTp);
                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].mdt = DT_F32;
                weightSpecVec[i].bytes_of_weight = weightNum * sizeof(float);
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                memcpy(weightSpecVec[i].weight, weightParamPtr, weightSpecVec[i].bytes_of_weight);
                weightSpecVec[i].bytes_of_vec = 0;
                weightSpecVec[i].vec = nullptr;
            } else if (weightOpType == "Splice") {
                std::vector<int> indices =
                    get_node_vector_ints_attribute_by_name(weightNode, "forward_indexes");
                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].mdt = DT_U32;
                weightSpecVec[i].bytes_of_weight = indices.size() * sizeof(U32);
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                memcpy(weightSpecVec[i].weight, indices.data(), weightSpecVec[i].bytes_of_weight);
                weightSpecVec[i].bytes_of_vec = 0;
                weightSpecVec[i].vec = nullptr;
            } else if (weightOpType == "Where") {
                bool *conditionTpPtr = nullptr;
                int conditionTpSize = 0;
                std::vector<float> conditionVec;
                if (weights.find(weightNode.input(0)) != weights.end()) {
                    auto conditionTp = weights[weightNode.input(0)];
                    conditionTpPtr = (bool *)(get_ptr_from_weight_obj(conditionTp));
                    conditionTpSize = get_data_size_from_tensor_proto(conditionTp);
                    for (int i = 0; i < conditionTpSize; i++) {
                        float curCon = (conditionTpPtr[i] == true) ? 1.0 : 0.0;
                        conditionVec.push_back(curCon);
                    }
                }
                U8 *yPtr = nullptr;
                int yTpSize = 0;
                if (weights.find(weightNode.input(2)) != weights.end()) {
                    auto yTp = weights[weightNode.input(2)];
                    yPtr = get_ptr_from_weight_obj(yTp);
                    yTpSize = get_data_size_from_tensor_proto(yTp);
                }
                weightSpecVec[i].mdt = DT_F32;
                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].bytes_of_weight = conditionTpSize * sizeof(float);
                if (weightSpecVec[i].bytes_of_weight == 0) {
                    weightSpecVec[i].weight = nullptr;
                } else {
                    weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                    memcpy(weightSpecVec[i].weight, conditionVec.data(),
                        weightSpecVec[i].bytes_of_weight);
                }
                weightSpecVec[i].bytes_of_vec = yTpSize * sizeof(float);
                if (weightSpecVec[i].bytes_of_vec == 0) {
                    weightSpecVec[i].vec = nullptr;
                } else {
                    weightSpecVec[i].vec = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_vec);
                    memcpy(weightSpecVec[i].vec, yPtr, weightSpecVec[i].bytes_of_vec);
                }
            } else if (weightOpType == "Linear") {
                auto linearWeightTp = weights[node.input(1)];
                int weightSize = get_data_size_from_tensor_proto(linearWeightTp);
                U8 *tpPtr = get_ptr_from_weight_obj(linearWeightTp);
                weightSpecVec[i].mdt = DT_F32;
                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].bytes_of_weight = weightSize * sizeof(float);
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                memcpy(weightSpecVec[i].weight, tpPtr, weightSpecVec[i].bytes_of_weight);
                weightSpecVec[i].bytes_of_vec = 0;
                weightSpecVec[i].vec = nullptr;
            } else if (weightOpType == "Equal") {
                auto cmpTp = weights[weightNode.input(1)];
                int cmpTpSize = get_data_size_from_tensor_proto(cmpTp);
                if (cmpTp.data_type() == onnx::TensorProto::FLOAT) {
                    weightSpecVec[i].mdt = DT_F32;
                } else if (cmpTp.data_type() == onnx::TensorProto::INT32) {
                    weightSpecVec[i].mdt = DT_I32;
                } else {
                    UNI_ERROR_LOG("can not process operator name:%s %s type Equal.\n",
                        this->node.name().c_str(), onnx_data_type_string(cmpTp.data_type()).c_str());
                }
                U8 *cmpPtr = (U8 *)get_ptr_from_weight_obj(cmpTp);
                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                weightSpecVec[i].bytes_of_weight = cmpTpSize * sizeof(float);
                weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                memcpy(weightSpecVec[i].weight, cmpPtr, weightSpecVec[i].bytes_of_weight);
                weightSpecVec[i].bytes_of_vec = 0;
                weightSpecVec[i].vec = nullptr;
            } else if (weightOpType == "Scan") {
                onnx::GraphProto gp;
                for (int k = 0; k < weightNode.attribute_size(); k++) {
                    const onnx::AttributeProto &attribute = weightNode.attribute(k);
                    if (attribute.name() == "body") {
                        gp = attribute.g();
                        break;
                    }
                }

                // extract the weight from scan tp
                std::map<std::string, std::vector<onnx::TensorProto>> weightMap;
                std::vector<onnx::TensorProto> tps;
                weightMap["Gemm"] = tps;
                weightMap["MatMul"] = tps;
                std::map<std::string, std::vector<int>> transMap;
                std::vector<int> trans;
                transMap["Gemm"] = trans;
                transMap["MatMul"] = trans;
                for (int j = 0; j < gp.node_size(); j++) {
                    auto curNode = gp.node(j);
                    if (curNode.op_type() == "Gemm" || curNode.op_type() == "MatMul") {
                        for (int k = 0; k < (int)curNode.input_size(); k++) {
                            if (weights.find(curNode.input(k)) != weights.end()) {
                                auto hidWeightTp = weights[curNode.input(k)];
                                if (get_data_size_from_tensor_proto(hidWeightTp) == 0) {
                                    continue;
                                } else {
                                    weightMap[curNode.op_type()].push_back(hidWeightTp);
                                }
                            }
                        }
                        int noTransB = 1;
                        noTransB = get_node_single_int_attribute_by_name(curNode, "transB", 0);
                        transMap[curNode.op_type()].push_back(noTransB);
                    }
                }

                // initial empty desc
                TensorDesc empDesc;
                empDesc.nDims = 0;
                TensorDesc wDesc1 = genDescFromTp(weightMap["Gemm"][0]);
                TensorDesc bDesc1;
                if (weightMap["Gemm"].size() > 1) {
                    bDesc1 = genDescFromTp(weightMap["Gemm"][1]);
                } else {
                    bDesc1 = empDesc;
                }
                TensorDesc wDesc2;
                if (weightMap["MatMul"].size() > 0) {
                    wDesc2 = genDescFromTp(weightMap["MatMul"][0]);
                } else {
                    wDesc2 = empDesc;
                }
                TensorDesc bDesc2;
                if (weightMap["MatMul"].size() > 1) {
                    bDesc2 = genDescFromTp(weightMap["MatMul"][1]);
                } else {
                    bDesc2 = empDesc;
                }

                weightSpecVec[i].mdt = DT_F32;
                str_copy(weightSpecVec[i].op_name, weightOpName.c_str(), weightOpName.length());
                int wBytes = tensorNumElements(wDesc1) + tensorNumElements(wDesc2);
                weightSpecVec[i].bytes_of_weight = wBytes * sizeof(float);
                if (weightSpecVec[i].bytes_of_weight == 0) {
                    weightSpecVec[i].weight = nullptr;
                } else {
                    weightSpecVec[i].weight = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_weight);
                    int wOffSet = 0;
                    if (tensorNumElements(wDesc1) > 0) {
                        U8 *tmpWPtr1 = get_ptr_from_weight_obj(weightMap["Gemm"][0]);
                        if (transMap["Gemm"][0]) {
                            memcpy(&((weightSpecVec[i].weight)[wOffSet]), tmpWPtr1,
                                tensorNumBytes(wDesc1));
                        } else {
                            memcpy_trans2d(&((weightSpecVec[i].weight)[wOffSet]), tmpWPtr1,
                                (int)weightMap["Gemm"][0].dims(1),
                                (int)weightMap["Gemm"][0].dims(0));
                        }
                        wOffSet += tensorNumBytes(wDesc1);
                    }
                    if (tensorNumElements(wDesc2) > 0) {
                        U8 *tmpWPtr2 = get_ptr_from_weight_obj(weightMap["MatMul"][0]);
                        if (transMap["MatMul"][0]) {
                            memcpy(&((weightSpecVec[i].weight)[wOffSet]), tmpWPtr2,
                                tensorNumBytes(wDesc2));
                        } else {
                            memcpy_trans2d(&((weightSpecVec[i].weight)[wOffSet]), tmpWPtr2,
                                (int)weightMap["MatMul"][0].dims(1),
                                (int)weightMap["MatMul"][0].dims(0));
                        }
                    }
                }

                int bBytes = tensorNumElements(bDesc1) + tensorNumElements(bDesc2);
                weightSpecVec[i].bytes_of_vec = bBytes * sizeof(float);
                if (weightSpecVec[i].bytes_of_vec == 0) {
                    weightSpecVec[i].vec = nullptr;
                } else {
                    weightSpecVec[i].vec = (U8 *)mt_new_storage(weightSpecVec[i].bytes_of_vec);
                    int bOffSet = 0;
                    if (tensorNumElements(bDesc1) > 0) {
                        U8 *tmpBPtr1 = get_ptr_from_weight_obj(weightMap["Gemm"][1]);
                        memcpy(&((weightSpecVec[i].vec)[bOffSet]), tmpBPtr1, tensorNumBytes(bDesc1));
                        bOffSet += tensorNumBytes(bDesc1);
                    }
                    if (tensorNumElements(bDesc2) > 0) {
                        U8 *tmpBPtr2 = get_ptr_from_weight_obj(weightMap["MatMul"][1]);
                        memcpy(&((weightSpecVec[i].vec)[bOffSet]), tmpBPtr2, tensorNumBytes(bDesc2));
                    }
                }
            }
            weightOpIndexIndeed++;
        }
        ms->num_weight_specs = this->sharedWeightWps.size() + this->weightSpecVec.size();
        ms->ws = (WeightSpec *)mt_new_storage(sizeof(WeightSpec) * ms->num_weight_specs);
        std::vector<WeightSpec> combineWpsVec;
        combineWpsVec.insert(combineWpsVec.end(), sharedWeightWps.begin(), sharedWeightWps.end());
        combineWpsVec.insert(combineWpsVec.end(), weightSpecVec.begin(), weightSpecVec.end());
        memcpy(ms->ws, combineWpsVec.data(), sizeof(WeightSpec) * ms->num_weight_specs);
        for (I32 i = 0; i < ms->num_weight_specs; i++) {
            ms->ws[i].num_quant_scale = 0;
            ms->ws[i].weight_scale = nullptr;
        }
        return ret;
    }

    ParameterSpec adapt_SharedWeight() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        const onnx::TensorProto &data = weights[node.input(0)];
        const onnx::TensorProto &ind = weights[node.input(1)];
        SharedWeightParamSpec sharedWeightPs;
        sharedWeightPs.desc.nDims = 3;
        sharedWeightPs.desc.dims[2] = 1;
        sharedWeightPs.desc.dims[1] = ind.dims(1);
        sharedWeightPs.desc.dims[0] = data.dims(1);
        sharedWeightPs.desc.df = DF_NORMAL;
        sharedWeightPs.desc.dt = DT_F32;
        curPs.shared_weight_spec = sharedWeightPs;
        return curPs;
    }

    ParameterSpec adapt_Reshape() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ReshapeParamSpec reshapePs;
        memset(&reshapePs, 0, sizeof(reshapePs));
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
                reshapeInfo = get_int_vec_from_tensorProto(
                    weights[node.input(1)]);  // tp:weights[node.input(1)]
            }
        }
        reshapePs.shape_size = reshapeInfo.size();
        memcpy(reshapePs.shape_dims, reshapeInfo.data(), reshapePs.shape_size * sizeof(I32));
        reshapePs.axis = 0;
        reshapePs.num_axes = -1;
        curPs.reshape_spec = reshapePs;
        return curPs;
    }

    ParameterSpec adapt_Resize() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ResizeParamSpec resizePs;
        memset(&resizePs, 0, sizeof(resizePs));
        resizePs.num_scales = 0;
        resizePs.num_sizes = 0;
        std::string scalesIndex = "";
        std::string sizesIndex = "";
        if (this->op == "Resize") {
            for (int i = 0; i < node.input_size(); i++) {
                if (weights.find(node.input(i)) != weights.end()) {
                    auto curTp = weights[node.input(i)];
                    if (curTp.data_type() == onnx::TensorProto::FLOAT) {
                        scalesIndex = node.input(i);
                    } else if (curTp.data_type() == onnx::TensorProto::INT64) {
                        sizesIndex = node.input(i);
                    } else {
                        UNI_ERROR_LOG("can not process operator name:%s %s type attributes.\n",
                            this->node.name().c_str(),
                            onnx_data_type_string(curTp.data_type()).c_str());
                    }
                }
            }
        } else if (this->op == "Upsample") {
            scalesIndex = node.input(1);
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Resize.\n",
                this->node.name().c_str(), op.c_str());
        }
        if (scalesIndex != "") {
            const onnx::TensorProto &scales = weights[scalesIndex];
            if (scales.dims(0) == 0 || scales.dims(0) == 4) {
                resizePs.num_scales = scales.dims(0);
                U8 *ptr = get_ptr_from_weight_obj(scales);
                memcpy(resizePs.scales, ptr, resizePs.num_scales * bytesOf(DT_F32));
            } else {
                UNI_ERROR_LOG("can not get scale information from operator name:%s type:%s.\n",
                    this->node.name().c_str(), op.c_str());
            }
        }
        if (sizesIndex != "") {
            const onnx::TensorProto &sizes = weights[sizesIndex];
            if (sizes.dims(0) == 0) {
            } else if (sizes.dims(0) == 4) {
                std::vector<int> ptr = get_int_vec_from_tensorProto(sizes);
                resizePs.num_sizes = 2;
                resizePs.sizes[0] = ptr[2];
                resizePs.sizes[1] = ptr[3];
            } else {
                UNI_ERROR_LOG("can not get resize information from operator name:%s "
                              "type:%s.\n",
                    this->node.name().c_str(), op.c_str());
            }
        }

        std::string mode = get_node_str_attribute_by_name(node, "mode", "nearest");
        std::string coordinate_transformation_mode =
            get_node_str_attribute_by_name(node, "coordinate_transformation_mode", "half_pixel");
        std::string nearest_mode =
            get_node_str_attribute_by_name(node, "nearest_mode", "round_prefer_floor");

        if (mode.compare("linear") == 0) {
            resizePs.mode = LINEAR;
        } else {
            resizePs.mode = NEAREST;
        }

        if (coordinate_transformation_mode.compare("align_corners") == 0) {
            resizePs.trans_mode = ALIGN_CORNERS;
        } else {
            resizePs.trans_mode = HALF_PIXEL;
        }

        if (nearest_mode.compare("round_prefer_floor") == 0) {
            resizePs.round_mode = ROUND_PREFER_FLOOR;
        } else if (nearest_mode.compare("round_prefer_ceil") == 0) {
            resizePs.round_mode = ROUND_PREFER_CEIL;
        } else if (nearest_mode.compare("floor") == 0) {
            resizePs.round_mode = FLOOR;
        } else {
            resizePs.round_mode = CEIL;
        }

        curPs.resize_spec = resizePs;
        return curPs;
    }

    ParameterSpec adapt_Transpose() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        TransposeParamSpec transposePs;
        memset(&transposePs, 0, sizeof(transposePs));
        std::vector<int> transpose_info = get_node_vector_ints_attribute_by_name(node, "perm");
        transposePs.trans_size = transpose_info.size();
        memcpy(transposePs.trans_dims, transpose_info.data(), transposePs.trans_size * sizeof(U32));
        curPs.transpose_spec = transposePs;
        return curPs;
    }

    ParameterSpec adapt_Clip() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ClipParamSpec clipParam;
        memset(&clipParam, 0, sizeof(clipParam));
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
        memset(&curPs, 0, sizeof(curPs));
        ConvolutionParamSpec cps;
        memset(&cps, 0, sizeof(cps));
        std::vector<int> kernelShape = get_node_vector_ints_attribute_by_name(node, "kernel_shape");
        std::vector<int> dilations = get_node_vector_ints_attribute_by_name(node, "dilations");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(node, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(node, "pads");
        int group = get_node_single_int_attribute_by_name(node, "group", 1);

        const onnx::TensorProto &weight = weights[node.input(1)];
        cps.num_outputs = weight.dims(0);
        cps.num_outputs_origin = cps.num_outputs;
        cps.kernel_t = 1;
        cps.kernel_h = 1;
        cps.kernel_w = 1;
        if (kernelShape.size() == 3) {
            cps.kernel_t = kernelShape[0];
            cps.kernel_h = kernelShape[1];
            cps.kernel_w = kernelShape[2];
        } else if (kernelShape.size() == 2) {
            cps.kernel_h = kernelShape[0];
            cps.kernel_w = kernelShape[1];
        } else if (kernelShape.size() == 1) {
            cps.kernel_h = kernelShape[0];
        }

        cps.dilatedRate_t = 1;
        cps.dilatedRate_h = 1;
        cps.dilatedRate_w = 1;
        if (dilations.size() == 3) {
            cps.dilatedRate_t = dilations[0];
            cps.dilatedRate_h = dilations[1];
            cps.dilatedRate_w = dilations[2];
        } else if (dilations.size() == 2) {
            cps.dilatedRate_h = dilations[0];
            cps.dilatedRate_w = dilations[1];
        } else if (dilations.size() == 1) {
            cps.dilatedRate_h = dilations[0];
        }

        cps.stride_t = 1;
        cps.stride_h = 1;
        cps.stride_w = 1;
        if (strides.size() == 3) {
            cps.stride_t = strides[0];
            cps.stride_h = strides[1];
            cps.stride_w = strides[2];
        } else if (strides.size() == 2) {
            cps.stride_h = strides[0];
            cps.stride_w = strides[1];
        } else if (strides.size() == 1) {
            cps.stride_h = strides[0];
        }

        cps.padding_before = 0;
        cps.padding_top = 0;
        cps.padding_left = 0;
        cps.padding_after = 0;
        cps.padding_bottom = 0;
        cps.padding_right = 0;
        if (pads.size() == 6) {
            cps.padding_before = pads[0];
            cps.padding_top = pads[1];
            cps.padding_left = pads[2];
            cps.padding_after = pads[3];
            cps.padding_bottom = pads[4];
            cps.padding_right = pads[5];
        } else if (pads.size() == 4) {
            cps.padding_top = pads[0];
            cps.padding_left = pads[1];
            cps.padding_bottom = pads[2];
            cps.padding_right = pads[3];
        } else if (pads.size() == 2) {
            cps.padding_top = pads[0];
            cps.padding_bottom = pads[1];
        }

        cps.group = group;
        if (cps.group != 1 && cps.group == cps.num_outputs) {
            cps.convolution_type = Convolution_Depthwise;
        } else {
            if (cps.dilatedRate_t > 1 || cps.dilatedRate_h > 1 || cps.dilatedRate_w > 1) {
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
        memset(&curPs, 0, sizeof(curPs));
        ConvolutionParamSpec cps;
        memset(&cps, 0, sizeof(cps));
        std::vector<int> kernelShape = get_node_vector_ints_attribute_by_name(node, "kernel_shape");
        std::vector<int> dilations = get_node_vector_ints_attribute_by_name(node, "dilations");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(node, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(node, "pads");
        int group = get_node_single_int_attribute_by_name(node, "group", 1);

        const onnx::TensorProto &weight = weights[node.input(1)];
        cps.num_outputs = weight.dims(1);
        cps.kernel_t = 1;
        cps.kernel_h = 1;
        cps.kernel_w = 1;
        if (kernelShape.size() == 2) {
            cps.kernel_h = kernelShape[0];
            cps.kernel_w = kernelShape[1];
        } else if (kernelShape.size() == 1) {
            cps.kernel_h = kernelShape[0];
        }

        cps.dilatedRate_t = 1;
        cps.dilatedRate_h = 1;
        cps.dilatedRate_w = 1;
        if (dilations.size() == 2) {
            cps.dilatedRate_h = dilations[0];
            cps.dilatedRate_w = dilations[1];
        } else if (dilations.size() == 1) {
            cps.dilatedRate_h = dilations[0];
        }

        cps.stride_t = 1;
        cps.stride_h = 1;
        cps.stride_w = 1;
        if (strides.size() == 2) {
            cps.stride_h = strides[0];
            cps.stride_w = strides[1];
        } else if (strides.size() == 1) {
            cps.stride_h = strides[0];
        }

        cps.padding_before = 0;
        cps.padding_after = 0;
        cps.padding_top = 0;
        cps.padding_bottom = 0;
        cps.padding_left = 0;
        cps.padding_right = 0;
        cps.rm = CEIL;
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
        memset(&curPs, 0, sizeof(curPs));
        PoolingParamSpec pps;
        memset(&pps, 0, sizeof(pps));
        std::string autoPad = get_node_str_attribute_by_name(node, "auto_pad");  // deprecated
        std::vector<int> kernelShape = get_node_vector_ints_attribute_by_name(node, "kernel_shape");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(node, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(node, "pads");

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

        pps.kernel_t = 0;
        pps.kernel_h = 0;
        pps.kernel_w = 0;
        if (kernelShape.size() == 3) {
            pps.kernel_t = kernelShape[0];
            pps.kernel_h = kernelShape[1];
            pps.kernel_w = kernelShape[2];
        } else if (kernelShape.size() == 2) {
            pps.kernel_t = 1;
            pps.kernel_h = kernelShape[0];
            pps.kernel_w = kernelShape[1];
        } else if (kernelShape.size() == 1) {
            pps.kernel_t = 1;
            pps.kernel_h = kernelShape[0];
            pps.kernel_w = 1;
        }

        pps.stride_t = 1;
        pps.stride_h = 1;
        pps.stride_w = 1;
        if (strides.size() == 3) {
            pps.stride_t = strides[0];
            pps.stride_h = strides[1];
            pps.stride_w = strides[2];
        } else if (strides.size() == 2) {
            pps.stride_h = strides[0];
            pps.stride_w = strides[1];
        } else if (strides.size() == 1) {
            pps.stride_h = strides[0];
        }

        pps.padding_before = 0;
        pps.padding_top = 0;
        pps.padding_left = 0;
        pps.padding_after = 0;
        pps.padding_bottom = 0;
        pps.padding_right = 0;
        if (pads.size() == 6) {
            pps.padding_before = pads[0];
            pps.padding_top = pads[1];
            pps.padding_left = pads[2];
            pps.padding_after = pads[3];
            pps.padding_bottom = pads[4];
            pps.padding_right = pads[5];
        } else if (pads.size() == 4) {
            pps.padding_top = pads[0];
            pps.padding_left = pads[1];
            pps.padding_bottom = pads[2];
            pps.padding_right = pads[3];
        } else if (pads.size() == 2) {
            pps.padding_top = pads[0];
            pps.padding_bottom = pads[1];
        }
        curPs.pooling_spec = pps;
        return curPs;
    }

    ParameterSpec adapt_MatMul() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        MatMulParamSpec matmulPs;
        memset(&matmulPs, 0, sizeof(matmulPs));
        matmulPs.transpose_a = false;
        matmulPs.transpose_b = false;
        curPs.matmul_spec = matmulPs;
        return curPs;
    }

    ParameterSpec adapt_Fc() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        FullyConnectedParamSpec fcParamSpec;
        memset(&fcParamSpec, 0, sizeof(fcParamSpec));
        fcParamSpec.num_outputs = -1;

        if (op == "MatMul") {
            const onnx::TensorProto &matmulTp = weights[node.input(1)];
            if (matmulTp.dims_size() == 2) {
                fcParamSpec.num_outputs = matmulTp.dims(1);
            } else {
                UNI_ERROR_LOG("can not map operator name:%s type:%s to FullyConnected.\n",
                    this->node.name().c_str(), op.c_str());
            }
        } else if (op == "Linear") {
            const onnx::TensorProto &matmulTp = weights[node.input(1)];
            if (matmulTp.dims_size() == 2) {
                fcParamSpec.num_outputs = matmulTp.dims(0);
            } else {
                UNI_ERROR_LOG("can not map operator name:%s type:%s to FullyConnected.\n",
                    this->node.name().c_str(), op.c_str());
            }
        } else {
            float alpha = get_node_float_attribute_by_name(node, "alpha", 1.f);
            float beta = get_node_float_attribute_by_name(node, "beta", 1.f);
            int transA = get_node_single_int_attribute_by_name(node, "transA", 0);
            int transB = get_node_single_int_attribute_by_name(node, "transB", 0);
            auto weightTp = weights[node.input(1)];
            if (transB == 1.0) {
                fcParamSpec.num_outputs = weightTp.dims(0);
            } else {
                fcParamSpec.num_outputs = weightTp.dims(1);
            }
            if (!(alpha == 1.f && beta == 1.f && transA == 0)) {
                UNI_ERROR_LOG("can not map operator name:%s type:%s to FullyConnected.\n",
                    this->node.name().c_str(), op.c_str());
            }
        }
        fcParamSpec.num_slices = 1;
        fcParamSpec.slice_point[0] = fcParamSpec.num_outputs;
        curPs.fc_spec = fcParamSpec;
        return curPs;
    }

    ParameterSpec adapt_BatchNorm() override
    {
        if (weightOpIndexLists.size() == 0 ||
            weightOpIndexLists[weightOpIndexLists.size() - 1] != nodeIndex) {
            weightOpIndexLists.push_back(nodeIndex);
        }
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        BatchNormParamSpec bnPs;
        memset(&bnPs, 0, sizeof(bnPs));
        bnPs.eps = get_node_float_attribute_by_name(node, "epsilon", 1e-5f);
        if (op == "BatchNormalization") {
            bnPs.axis = 1;
        } else if (op == "BatchNorm") {
            bnPs.axis = -1;
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to BatchNorm.\n",
                this->node.name().c_str(), op.c_str());
        }
        bnPs.gama = 1;
        bnPs.momentum = get_node_float_attribute_by_name(node, "momentum", 0.9);
        curPs.bn_spec = bnPs;
        return curPs;
    }

    ParameterSpec adapt_Eltwise() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        EltwiseParamSpec eps;
        memset(&eps, 0, sizeof(eps));
        if (op == "Add" || op == "Sum") {
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
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Eltwise.\n",
                this->node.name().c_str(), op.c_str());
        }

        // insert shared weight
        std::string insertWeightName = "";
        std::string insertOpName = "";
        if (node.name().length() > 0) {
            insertOpName = node.name();
        } else {
            insertOpName = node.output(0);
        }
        if (weights.find(node.input(0)) != weights.end()) {
            insertWeightName = node.input(0);
        } else if (weights.find(node.input(1)) != weights.end()) {
            insertWeightName = node.input(1);
        }
        if (insertWeightName.length() > 0) {
            insertSharedWeight[insertWeightName] = insertOpName;
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
        memset(&curPs, 0, sizeof(curPs));
        PadParamSpec padPs;
        memset(&padPs, 0, sizeof(padPs));
        std::string padModeStr = get_node_str_attribute_by_name(node, "mode");
        std::vector<int> padVec = get_node_vector_ints_attribute_by_name(node, "pads");
        F32 padValue = get_node_float_attribute_by_name(node, "value", 0.f);
        if (padModeStr == "constant" || padModeStr.length() == 0) {
            padPs.pad_mode = Pad_Constant;
        } else if (padModeStr == "edge") {
            padPs.pad_mode = Pad_Edge;
        } else if (padModeStr == "reflect") {
            padPs.pad_mode = Pad_Reflect;
        }

        padPs.before = 0;
        padPs.after = 0;
        U32 padSize = padVec.size();
        if (padSize == 0) {
            const onnx::TensorProto &padsTp = weights[node.input(1)];
            padVec = get_int_vec_from_tensorProto(weights[node.input(1)]);
            padSize = padVec.size();
        }
        if (padSize == 8) {  // NCHW
            padPs.front = padVec[1];
            padPs.back = padVec[5];
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
            UNI_ERROR_LOG("can not process operator name:%s type:%s attributes.\n",
                this->node.name().c_str(), op.c_str());
        }
        padPs.constant_value = padValue;
        curPs.pad_spec = padPs;
        return curPs;
    }

    ParameterSpec adapt_Gather() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        GatherParamSpec gps;
        memset(&gps, 0, sizeof(gps));
        gps.gather_axis = get_node_single_int_attribute_by_name(node, "axis", 0);
        curPs.gather_spec = gps;
        return curPs;
    }

    ParameterSpec adapt_TfSlice() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        TfSliceParamSpec tfSlicePs;
        memset(&tfSlicePs, 0, sizeof(tfSlicePs));
        std::vector<int> startsInfo;
        std::vector<int> endsInfo;
        std::vector<int> axesInfo;
        std::vector<int> stepInfo;

        if (node.input_size() == 1) {
            startsInfo = get_node_vector_ints_attribute_by_name(node, "starts");
            endsInfo = get_node_vector_ints_attribute_by_name(node, "ends");
            axesInfo = get_node_vector_ints_attribute_by_name(node, "axes");
        } else {
            startsInfo = get_int_vec_from_tensorProto(weights[node.input(1)]);
            endsInfo = get_int_vec_from_tensorProto(weights[node.input(2)]);
            if (node.input_size() >= 4) {
                axesInfo = get_int_vec_from_tensorProto(weights[node.input(3)]);
                if (node.input_size() >= 5) {
                    stepInfo = get_int_vec_from_tensorProto(weights[node.input(4)]);
                }
            }
        }
        tfSlicePs.dim_size = 8;
        for (U32 i = 0; i < tfSlicePs.dim_size; i++) {
            tfSlicePs.begin[i] = 0;
            tfSlicePs.end[i] = -1;
            tfSlicePs.strides[i] = 1;
            tfSlicePs.begin_mask[i] = 1;
            tfSlicePs.end_mask[i] = 1;
        }
        for (U32 i = 0; i < startsInfo.size(); i++) {
            int axis;
            if (axesInfo.size() > 0) {
                axis = axesInfo[i];
            } else {
                axis = i;
            }
            tfSlicePs.begin[axis] = startsInfo[i];
            tfSlicePs.end[axis] = endsInfo[i];
            tfSlicePs.begin_mask[axis] = 0;
            tfSlicePs.end_mask[axis] = 0;
        }
        curPs.tfslice_spec = tfSlicePs;
        return curPs;
    }

    ParameterSpec adapt_Slice() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        SliceParamSpec slice_ps;
        memset(&slice_ps, 0, sizeof(slice_ps));
        if (op == "Gather") {
            ParameterSpec gather_ps = adapt_Gather();
            slice_ps.slice_points[0] = 1;
            slice_ps.slice_size = 1;
            slice_ps.axis = gather_ps.gather_spec.gather_axis;
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
        memset(&curPs, 0, sizeof(curPs));
        EmbedParamSpec embed_ps;
        memset(&embed_ps, 0, sizeof(embed_ps));
        std::string embed_weight_name = node.input(0);
        if (weights.find(node.input(0)) == weights.end()) {
            return curPs;
        }
        auto tensor_proto = weights[embed_weight_name];
        int size_of_dims = tensor_proto.dims_size();
        if (size_of_dims != 2) {
            UNI_ERROR_LOG("can not process operator name:%s type:%s attributes.\n",
                this->node.name().c_str(), op.c_str());
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
        memset(&curPs, 0, sizeof(curPs));
        SqueezeParamSpec squeezePs;
        memset(&squeezePs, 0, sizeof(squeezePs));
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
        memset(&curPs, 0, sizeof(curPs));
        UnsqueezeParamSpec unsqueezePs;
        memset(&unsqueezePs, 0, sizeof(unsqueezePs));
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
        memset(&curPs, 0, sizeof(curPs));
        CastParamSpec castPs;
        memset(&castPs, 0, sizeof(castPs));

        int cast_to;
        if (node.input_size() == 2 && weights.find(node.input(1)) != weights.end()) {
            cast_to = (get_int_vec_from_tensorProto(weights[node.input(1)]))[0];
        } else {
            cast_to = get_node_single_int_attribute_by_name(node, "to", 0);
        }

        if (cast_to == onnx::TensorProto::FLOAT) {
            castPs.targetDt = DT_F32;
        } else if (cast_to == onnx::TensorProto::FLOAT16) {
            castPs.targetDt = DT_F16;
        } else if (cast_to == onnx::TensorProto::INT16 || cast_to == onnx::TensorProto::INT32 ||
            cast_to == onnx::TensorProto::INT64) {
            castPs.targetDt = DT_I32;
        } else {
            castPs.targetDt = DT_F32;  // default
        }
        curPs.cast_spec = castPs;
        return curPs;
    }

    ParameterSpec adapt_Concat() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ConcatParamSpec concatPs;
        memset(&concatPs, 0, sizeof(concatPs));
        concatPs.axis = get_node_single_int_attribute_by_name(node, "axis", 1);
        curPs.concat_spec = concatPs;

        std::string insertWeightName = "";
        std::string insertOpName = "";
        if (node.name().length() > 0) {
            insertOpName = node.name();
        } else {
            insertOpName = node.output(0);
        }
        if (weights.find(node.input(0)) != weights.end()) {
            insertWeightName = node.input(0);
        } else if (node.input_size() > 1 && weights.find(node.input(1)) != weights.end()) {
            insertWeightName = node.input(1);
        }
        if (insertWeightName.length() > 0) {
            insertSharedWeight[insertWeightName] = insertOpName;
        }
        return curPs;
    }

    ParameterSpec adapt_Softmax() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        SoftmaxParamSpec softmaxPs;
        memset(&softmaxPs, 0, sizeof(softmaxPs));
        softmaxPs.axis = get_node_single_int_attribute_by_name(node, "axis", 1);
        curPs.softmax_spec = softmaxPs;
        return curPs;
    }

    ParameterSpec adapt_Relu() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ReLUParamSpec reluPs;
        memset(&reluPs, 0, sizeof(reluPs));
        reluPs.neg_slope = get_node_float_attribute_by_name(node, "alpha", 0.0);
        curPs.relu_spec = reluPs;
        return curPs;
    }

    ParameterSpec adapt_RNN() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        if (this->op == "Scan") {
            return adapt_Scan();
        }
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        RNNParamSpec rnnPs;
        memset(&rnnPs, 0, sizeof(rnnPs));
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
            UNI_ERROR_LOG("can not map operator name:%s type:%s to RNN.\n",
                this->node.name().c_str(), this->op.c_str());
        }
        rnnPs.numOutput = get_node_single_int_attribute_by_name(node, "hidden_size", 1);
        rnnPs.biDirection =
            get_node_str_attribute_by_name(node, "direction", "forward") == "bidirectional" ? true
                                                                                            : false;
        rnnPs.steps = 0;
        rnnPs.numProjection = 0;
        rnnPs.zoneoutCell = 0;
        rnnPs.zoneoutOutput = 0;
        rnnPs.forgetBias = 0;
        rnnPs.activationMode = ACTIVATION_TANH;
        curPs.rnn_spec = rnnPs;
        return curPs;
    }

    // (scale * x + shift) ^ power
    ParameterSpec adapt_Power() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        PowerParamSpec powerPs;
        memset(&powerPs, 0, sizeof(powerPs));
        powerPs.scale = 1;
        powerPs.shift = 0;
        powerPs.power = 1;
        int index = 0;
        float value = 0;
        if (this->op == "Pow" || this->op == "Mul" || this->op == "Div" || this->op == "Add" ||
            this->op == "Sub") {
            std::vector<int> indexes = getOperatorWeightInputIndex(this->nodeIndex);
            CHECK_REQUIREMENT(indexes.size() == 1);
            index = indexes[0];
            const onnx::TensorProto &tp = weights[this->node.input(index)];
            value = getSinFloat_from_tensorProto(tp);
        }
        if (this->op == "Pow") {
            powerPs.power = value;
        } else if (this->op == "Mul") {
            powerPs.scale = value;
        } else if (this->op == "Div") {
            powerPs.scale = 1 / value;
            if (index == 0) {
                powerPs.power = -1;
            }
        } else if (this->op == "Add") {
            powerPs.shift = value;
        } else if (this->op == "Sub") {
            if (index == 0) {
                powerPs.scale = -1;
                powerPs.shift = value;
            } else {
                powerPs.shift = -1 * value;
            }
        } else if (this->op == "Sqrt") {
            powerPs.power = 0.5;
        } else if (this->op == "Scale") {
            powerPs.scale = get_node_float_attribute_by_name(node, "scale", 1.0);
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Power.\n",
                this->node.name().c_str(), this->op.c_str());
        }
        curPs.power_spec = powerPs;
        return curPs;
    }

    ParameterSpec adapt_Scale() override
    {
        if (weightOpIndexLists.size() == 0 ||
            weightOpIndexLists[weightOpIndexLists.size() - 1] != nodeIndex) {
            weightOpIndexLists.push_back(nodeIndex);
        }
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ScaleParamSpec scale_ps;
        memset(&scale_ps, 0, sizeof(scale_ps));
        scale_ps.axis = 1;
        curPs.scale_spec = scale_ps;
        return curPs;
    }

    ParameterSpec adapt_Space2Depth() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        Space2DepthParamSpec s2dPs;
        memset(&s2dPs, 0, sizeof(s2dPs));
        s2dPs.blockSize = get_node_single_int_attribute_by_name(node, "blocksize", 1);
        curPs.space2depth_spec = s2dPs;
        return curPs;
    }

    ParameterSpec adapt_Depth2Space() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        Depth2SpaceParamSpec d2sPs;
        memset(&d2sPs, 0, sizeof(d2sPs));
        d2sPs.blockSize = get_node_single_int_attribute_by_name(node, "blocksize", 1);
        std::string d2s_mode = get_node_str_attribute_by_name(node, "mode", "DCR");
        str_copy(d2sPs.reMode, d2s_mode.c_str(), d2s_mode.length(), 8);
        curPs.depth2space_spec = d2sPs;
        return curPs;
    }

    ParameterSpec adapt_Reduction() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ReductionParamSpec rsPs;
        memset(&rsPs, 0, sizeof(rsPs));
        std::vector<int> axesInfo = get_node_vector_ints_attribute_by_name(node, "axes");
        int keepdimsInfo = get_node_single_int_attribute_by_name(node, "keepdims", 0);
        rsPs.axes_num = axesInfo.size();
        for (int i = 0; i < rsPs.axes_num; i++) {
            rsPs.axes[i] = axesInfo[i];
        }
        rsPs.keep_dim = keepdimsInfo == 0 ? false : true;
        rsPs.coeff = 1.0;
        if (op == "ReduceSum") {
            rsPs.reduction_mode = REDUCTION_SUM;
        } else if (op == "ReduceMean") {
            rsPs.reduction_mode = REDUCTION_MEAN;
        } else if (op == "ReduceMax") {
            rsPs.reduction_mode = REDUCTION_MAX;
        } else if (op == "ReduceMin") {
            rsPs.reduction_mode = REDUCTION_MIN;
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Reduction.\n",
                this->node.name().c_str(), this->op.c_str());
        }
        curPs.reduction_spec = rsPs;
        return curPs;
    }

    ParameterSpec adapt_ArgMax() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ArgMaxParamSpec amPs;
        memset(&amPs, 0, sizeof(amPs));
        amPs.axis = get_node_single_int_attribute_by_name(node, "axis", -1);
        curPs.argmax_spec = amPs;
        return curPs;
    }

    ParameterSpec adapt_PRelu() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        return curPs;
    }

    ParameterSpec adapt_Tile() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        TileParamSpec tilePs;
        memset(&tilePs, 0, sizeof(tilePs));
        std::vector<int> tileInfo = get_int_vec_from_tensorProto(weights[node.input(1)]);
        if (tileInfo.size() > 0 && tileInfo.size() <= 8) {
            tilePs.dimsSize = tileInfo.size();
        } else {
            UNI_ERROR_LOG("can not process operator name:%s type:%s attributes.\n",
                this->node.name().c_str(), this->op.c_str());
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
        memset(&curPs, 0, sizeof(curPs));
        SpliceParamSpec splicePs;
        memset(&splicePs, 0, sizeof(splicePs));
        std::vector<int> context = get_node_vector_ints_attribute_by_name(node, "context");
        std::vector<int> indexes = get_node_vector_ints_attribute_by_name(node, "forward_indexes");
        splicePs.num_context = context.size();
        if (splicePs.num_context == 0) {
            UNI_ERROR_LOG("can not process operator name:%s type:%s attributes.\n",
                this->node.name().c_str(), this->op.c_str());
        }
        for (int i = 0; i < splicePs.num_context; i++) {
            splicePs.context[i] = context[i];
        }
        splicePs.index_min = 0;
        splicePs.index_max = 0;
        for (U32 i = 0; i < indexes.size(); i++) {
            splicePs.index_min = UNI_MIN(splicePs.index_min, indexes[i]);
            splicePs.index_max = UNI_MAX(splicePs.index_max, indexes[i]);
        }

        curPs.splice_spec = splicePs;
        return curPs;
    }

    ParameterSpec adapt_Tdnn() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        TdnnParamSpec tdnnPs;
        memset(&tdnnPs, 0, sizeof(tdnnPs));
        const onnx::TensorProto &context = weights[node.input(1)];
        const onnx::TensorProto &params = weights[node.input(2)];
        tdnnPs.num_context = get_data_size_from_tensor_proto(context);
        U8 *ptr = (U8 *)get_ptr_from_weight_obj(context);
        for (int i = 0; i < tdnnPs.num_context; i++) {
            int64_t value;
            memcpy(&value, ptr + i * sizeof(int64_t), sizeof(int64_t));
            tdnnPs.context[i] = value;
        }
        tdnnPs.num_outputs = params.dims(0);
        tdnnPs.activation_type = ACTIVATION_NULL;
        curPs.tdnn_spec = tdnnPs;
        return curPs;
    }

    ParameterSpec adapt_TopK() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        TopKParamSpec p;
        memset(&p, 0, sizeof(p));
        p.axis = get_node_single_int_attribute_by_name(node, "axis", -1);
        p.largest = get_node_single_int_attribute_by_name(node, "largest", 1);
        p.sorted = get_node_single_int_attribute_by_name(node, "sorted", 1);
        p.topk = get_node_single_int_attribute_by_name(node, "k", 1);
        curPs.topk_spec = p;
        return curPs;
    }

    ParameterSpec adapt_Where() override
    {
        weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        WhereParamSpec wherePs;
        memset(&wherePs, 0, sizeof(wherePs));
        if (weights.find(node.input(0)) == weights.end()) {
            UNI_WARNING_LOG("not find condition initializer in operator name:%s type:%s "
                            "attributes.\n",
                this->node.name().c_str(), this->op.c_str());
        } else {
            const onnx::TensorProto &conditionTp = weights[node.input(0)];
            wherePs.conditionDesc = genDescFromTp(conditionTp);
        }

        if (weights.find(node.input(2)) == weights.end()) {
            UNI_WARNING_LOG("not find y initializer in operator name:%s type:%s attributes.\n",
                this->node.name().c_str(), this->op.c_str());
        } else {
            const onnx::TensorProto &yTp = weights[node.input(2)];
            wherePs.yDesc = genDescFromTp(yTp);
        }

        curPs.where_spec = wherePs;
        return curPs;
    }

    ParameterSpec adapt_Scan()
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        RNNParamSpec rnnPs;
        memset(&rnnPs, 0, sizeof(rnnPs));
        onnx::GraphProto curGraph;
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = node.attribute(i);
            if (attribute.name() == "body") {
                curGraph = attribute.g();
                break;
            }
        }

        std::vector<onnx::TensorProto> gemmTps;
        std::vector<onnx::TensorProto> matmulTps;

        for (int i = 0; i < curGraph.node_size(); i++) {
            auto curNode = curGraph.node(i);
            int input_size = (int)curNode.input_size();
            bool stopTag = false;
            if (curNode.op_type() == "MatMul") {
                for (int j = 0; j < input_size; j++) {
                    if (weights.find(curNode.input(j)) != weights.end()) {
                        auto hidWeightTp = weights[curNode.input(j)];
                        int hidWeightSize = get_data_size_from_tensor_proto(hidWeightTp);
                        if (hidWeightSize > 0) {
                            matmulTps.push_back(hidWeightTp);
                            stopTag = true;
                            break;
                        }
                    }
                }
                if (stopTag) {
                    break;
                }
            }
        }

        rnnPs.mode = RNN_LSTM;
        // numProjection
        int numProjection = matmulTps[0].dims(0);
        int numOutput = matmulTps[0].dims(1);
        rnnPs.numOutput = numOutput;
        rnnPs.steps = 0;
        rnnPs.numProjection = numProjection;
        rnnPs.zoneoutCell = 0;
        rnnPs.zoneoutOutput = 0;
        rnnPs.forgetBias = 1.0;
        rnnPs.activationMode = ACTIVATION_TANH;
        curPs.rnn_spec = rnnPs;
        return curPs;
    }

private:
    std::string op;
    std::string modelName;
    int removePreprocessOpNum;
    onnx::ModelProto onnxModel;
    onnx::GraphProto onnxGraph;
    onnx::NodeProto node;
    std::map<std::string, onnx::TensorProto> weights;
    int nodeIndex;
    std::vector<int> weightOpIndexLists;
    int opFinalInputNum;

    std::map<std::string, std::string> insertSharedWeight;  // weight->op
    std::vector<OperatorSpec> operatorSpecVec;
    std::vector<WeightSpec> weightSpecVec;
    std::vector<OperatorSpec> sharedWeightOps;
    std::vector<WeightSpec> sharedWeightWps;

    std::map<std::string, std::string> croppingNames;
};
#endif
