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
#include <algorithm>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "onnx.pb.h"

#include "model_adaptee.h"

class OnnxAdaptee : public ModelAdaptee {
public:
    OnnxAdaptee(int _removePreprocessOpNum, bool _useBNN)
    {
        this->modelName = crop_name("model");
        this->removePreprocessOpNum = _removePreprocessOpNum;
        this->useBNN = _useBNN;
        this->useShare = false;
    }

    ~OnnxAdaptee()
    {
        google::protobuf::ShutdownProtobufLibrary();
    }

protected:
    EE read_file(const char *modelPath, google::protobuf::Message *message)
    {
        std::ifstream fs(modelPath, std::ifstream::in | std::ifstream::binary);
        if (!fs.is_open()) {
            UNI_ERROR_LOG("can not open onnx model file %s.\n", modelPath);
        }

        google::protobuf::io::IstreamInputStream input(&fs);
        google::protobuf::io::CodedInputStream codedstr(&input);
        codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
        if (!message->ParseFromCodedStream(&codedstr)) {
            UNI_ERROR_LOG("can not parse onnx model file %s.\n", modelPath);
        }
        fs.close();
        return SUCCESS;
    }

    EE parse_file(std::string modelDirectory, std::string modelFileName) override
    {
        this->modelName = crop_name(modelFileName);
        std::string modelPath = modelDirectory + "/" + modelFileName + ".onnx";
        CHECK_STATUS(read_file(modelPath.c_str(), (google::protobuf::Message *)(&onnxModel)));

        onnxGraph = onnxModel.graph();
        for (int i = 0; i < onnxGraph.initializer_size(); i++) {
            const onnx::TensorProto &initializer = onnxGraph.initializer(i);
            onnxWeights[initializer.name()] = initializer;
        }
        for (int i = 0; i < onnxGraph.value_info_size(); i++) {
            const onnx::ValueInfoProto &value = onnxGraph.value_info(i);
            onnxValues[value.name()] = value;
        }
        return SUCCESS;
    }

    std::vector<int> get_weight_ids(const onnx::NodeProto &node)
    {
        std::vector<int> ids;
        for (int i = 0; i < node.input_size(); i++) {
            if (onnxWeights.end() != onnxWeights.find(node.input(i))) {
                ids.push_back(i);
            }
        }
        return ids;
    }

    OperatorType convert_onnx_type(const std::string &onnxNodeType)
    {
        std::map<std::string, OperatorType> operatorMap = {{"Conv", OT_Conv},
            {"BatchNormalization", OT_BatchNorm}, {"BatchNorm", OT_BatchNorm},
            {"InstanceNormalization", OT_InstanceNorm}, {"AveragePool", OT_Pooling},
            {"MaxPool", OT_Pooling}, {"GlobalAveragePool", OT_Pooling}, {"Relu", OT_Relu},
            {"LeakyRelu", OT_Relu}, {"Softmax", OT_Softmax}, {"Concat", OT_Concat}, {"Pad", OT_Pad},
            {"Max", OT_Clip}, {"Min", OT_Clip}, {"Clip", OT_Clip}, {"Reshape", OT_Reshape},
            {"Squeeze", OT_Squeeze}, {"Unsqueeze", OT_Unsqueeze}, {"Transpose", OT_Transpose},
            {"Gather", OT_Gather}, {"GatherElements", OT_Gather}, {"GatherND", OT_Gather},
            {"Resize", OT_Resize}, {"Upsample", OT_Resize}, {"Cast", OT_Cast},
            {"Constant", OT_Constant}, {"Flatten", OT_Flatten}, {"ConvTranspose", OT_Deconvolution},
            {"Tanh", OT_TanH}, {"LogSoftmax", OT_LogSoftmax}, {"Shape", OT_Shape}, {"Erf", OT_Erf},
            {"Pow", OT_Power}, {"Sqrt", OT_Power}, {"RNN", OT_RNN}, {"GRU", OT_RNN},
            {"Scan", OT_RNN}, {"LSTM", OT_RNN}, {"ConstantOfShape", OT_ConstantOfShape},
            {"SpaceToDepth", OT_Space2Depth}, {"DepthToSpace", OT_Depth2Space}, {"PRelu", OT_PRelu},
            {"ArgMax", OT_ArgMax}, {"Tile", OT_Tile}, {"Sigmoid", OT_Sigmoid}, {"Slice", OT_TfSlice},
            {"ReduceSum", OT_Reduction}, {"ReduceMin", OT_Reduction}, {"ReduceL2", OT_Reduction},
            {"Split", OT_Slice}, {"Splice", OT_Splice}, {"Where", OT_Where},
            {"Softplus", OT_Softplus}, {"Exp", OT_Exp}, {"NoOp", OT_Slice}, {"Tdnn", OT_Tdnn},
            {"Dropout", OT_Dropout}, {"Scale", OT_Power}, {"TopK", OT_TopK}, {"Equal", OT_Check},
            {"Sign", OT_Sign}, {"TFL_HARD_SWISH", OT_HSwish}, {"Expand", OT_Expand},
            {"Scatter", OT_Scatter}, {"ScatterND", OT_Scatter}, {"ScatterElements", OT_Scatter},
            {"Not", OT_Not}, {"Abs", OT_Abs}, {"Reciprocal", OT_Reciprocal}, {"And", OT_Eltwise},
            {"Or", OT_Eltwise}, {"Xor", OT_Eltwise}, {"Log", OT_Log}, {"Neg", OT_Neg},
            {"GenerateProposals", OT_GenerateProposals}, {"RoIAlign", OT_RoIAlign},
            {"Round", OT_Round}, {"Floor", OT_Floor}, {"Ceil", OT_Ceil}, {"CumSum", OT_Cum},
            {"CumProd", OT_Cum}, {"RandomUniformLike", OT_Random}, {"RandomUniform", OT_Random},
            {"GridSample", OT_GridSample}, {"HardSigmoid", OT_HSigmoid}, {"OneHot", OT_OneHot},
            {"Identity", OT_Slice}, {"TFL_L2_NORMALIZATION", OT_L2Norm},
            {"NonMaxSuppression", OT_NonMaxSuppression}, {"Less", OT_Check}, {"Greater", OT_Check},
            {"GreaterOrEqual", OT_Check}, {"LessOrEqual", OT_Check}, {"NonZero", OT_NonZero},
            {"RoiAlign", OT_RoIAlign}, {"Loop", OT_Range}, {"Range", OT_Range}, {"Sin", OT_Sin},
            {"Cos", OT_Cos}, {"Elu", OT_Elu}, {"Einsum", OT_Einsum}, {"MaxUnpool", OT_UnPooling},
            {"LpNormalization", OT_L2Norm}, {"RandomNormalLike", OT_Random},
            {"RandomNormal", OT_Random}, {"BilateralSliceApply", OT_BilateralSliceApply},
            {"ConvertColor", OT_ConvertColor}, {"ReduceProd", OT_Reduction},
            {"LutPreprocess", OT_LutPreprocess}, {"Lut", OT_Lut}};
        if (operatorMap.find(onnxNodeType) != operatorMap.end()) {
            return operatorMap[onnxNodeType];
        }
        std::vector<int> ids = get_weight_ids(this->onnxNode);
        if (onnxNodeType == "Sum" || onnxNodeType == "Add" || onnxNodeType == "Mul" ||
            onnxNodeType == "Div" || onnxNodeType == "Sub") {
            if (ids.size() == 0 || this->onnxNode.input_size() > 2) {
                return OT_Eltwise;
            }
            std::vector<U32> axis(ids.size());
            for (U32 i = 0; i < ids.size(); i++) {
                TensorDesc desc = get_desc(onnxWeights[this->onnxNode.input(ids[i])]);
                int count = 0;
                for (U32 j = 0; j < desc.nDims; j++) {
                    if (desc.dims[j] > 1) {
                        count++;
                        axis[i] = j;
                    }
                }
                if (count > 1) {
                    return OT_Eltwise;
                }
            }
            const onnx::TensorProto &weight = onnxWeights[this->onnxNode.input(ids[0])];
            int length = get_length(weight);
            bool same = true;
            std::vector<float> v = get_floats(weight);
            for (int i = 1; i < length; i++) {
                if (v[i] != v[0]) {
                    same = false;
                    break;
                }
            }
            if (length == 1 || (weight.dims_size() == 1 && same)) {
                return OT_Power;
            } else if ((onnxNodeType == "Div" || onnxNodeType == "Sub") && ids[0] == 0) {
                return OT_Eltwise;
            } else if (this->onnxWeightReferCount[this->onnxNode.input(ids[0])] > 1) {
                return OT_Eltwise;
            } else {
                return OT_Scale;
            }
        } else if (onnxNodeType == "ReduceMean" || onnxNodeType == "ReduceMax") {
            std::vector<int> axes = get_ints(this->onnxNode, "axes");
            int keepdims = get_int(this->onnxNode, "keepdims", 0);
            if (axes.size() == 2 && axes[0] == 2 && axes[1] == 3 && keepdims == 1) {
                return OT_Pooling;
            }
            return OT_Reduction;
        } else if (onnxNodeType == "MatMul" || onnxNodeType == "Gemm" || onnxNodeType == "Linear") {
            if (ids.size() == 0 || (ids.size() == 1 && ids[0] != 1)) {
                return OT_MatMul;
            } else {
                auto weightName = this->onnxNode.input(ids[0]);
                onnx::TensorProto &weightTp = onnxWeights[weightName];
                if (weightTp.dims_size() == 2 && this->onnxWeightReferCount[weightName] <= 1) {
                    return OT_FC;
                } else {
                    return OT_MatMul;
                }
            }
        } else {
            UNI_ERROR_LOG("operator name:%s type:%s not supported.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
            return OT_None;
        }
    }

    std::string to_string(onnx::AttributeProto::AttributeType type)
    {
        const google::protobuf::EnumDescriptor *descriptor =
            onnx::AttributeProto::AttributeType_descriptor();
        return descriptor->FindValueByNumber(type)->name();
    }

    std::string to_string(int onnxDataType)
    {
        const google::protobuf::EnumDescriptor *descriptor =
            onnx::TensorProto::DataType_descriptor();
        return descriptor->FindValueByNumber(onnxDataType)->name();
    }

    std::string get_name(const onnx::NodeProto &node)
    {
        std::string opName = node.name();
        if (opName.empty() && node.output_size() > 0) {
            opName = node.output(0);
        }
        return opName;
    }

    int get_attribute_id(const onnx::NodeProto &node, const char *attributeName)
    {
        int ret = -1;
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = node.attribute(i);
            if (attribute.name() == attributeName) {
                ret = i;
                break;
            }
        }
        return ret;
    }

    DataType get_type(
        const onnx::NodeProto &node, const char *attributeName, DataType defaultValue = DT_F32)
    {
        int id = get_attribute_id(node, attributeName);
        if (id < 0) {
            return defaultValue;
        }
        const onnx::AttributeProto &attr = node.attribute(id);
        const auto &type = attr.type();
        DataType ret;
        if (type == onnx::AttributeProto::INT || type == onnx::AttributeProto::INTS) {
            ret = DT_I32;
        } else if (type == onnx::AttributeProto::FLOAT || type == onnx::AttributeProto::FLOATS) {
            ret = DT_F32;
        } else if (type == onnx::AttributeProto::TENSOR) {
            ret = get_type(attr.t());
        } else {
            UNI_ERROR_LOG("can not get operator name:%s attribute:%s %s type.\n",
                this->onnxNode.name().c_str(), attr.name().c_str(), to_string(type).c_str());
        }
        return ret;
    }

    int get_int(const onnx::NodeProto &node, const char *attributeName, int defaultValue = 0)
    {
        int id = get_attribute_id(node, attributeName);
        if (id < 0) {
            return defaultValue;
        }
        const onnx::AttributeProto &attr = node.attribute(id);
        if (attr.type() != onnx::AttributeProto::INT) {
            UNI_ERROR_LOG("can not get operator name:%s attribute:%s %s type value.\n",
                this->onnxNode.name().c_str(), attr.name().c_str(), to_string(attr.type()).c_str());
        }
        return UNI_MIN(attr.i(), INT_MAX);
    }

    std::vector<int> get_ints(const onnx::NodeProto &node, const char *attributeName)
    {
        std::vector<int> result;
        int id = get_attribute_id(node, attributeName);
        if (id < 0) {
            return result;
        }
        const onnx::AttributeProto &attr = node.attribute(id);
        if (attr.type() == onnx::AttributeProto::TENSOR) {
            result = get_ints(attr.t());
        } else if (attr.type() == onnx::AttributeProto::INTS) {
            result.resize(attr.ints_size());
            for (int j = 0; j < attr.ints_size(); j++) {
                result[j] = UNI_MIN(attr.ints(j), INT_MAX);
            }
        } else {
            UNI_ERROR_LOG("can not get operator name:%s attribute:%s %s type value.\n",
                this->onnxNode.name().c_str(), attr.name().c_str(), to_string(attr.type()).c_str());
        }
        return result;
    }

    float get_float(const onnx::NodeProto &node, const char *attributeName, float defaultValue = 0.f)
    {
        int id = get_attribute_id(node, attributeName);
        if (id < 0) {
            return defaultValue;
        }
        const onnx::AttributeProto &attr = node.attribute(id);
        float ret;
        if (attr.type() == onnx::AttributeProto::FLOAT) {
            ret = attr.f();
        } else {
            ret = get_floats(node, attributeName)[0];
        }
        return ret;
    }

    std::vector<F32> get_floats(const onnx::NodeProto &node, const char *attributeName)
    {
        std::vector<F32> result;
        int id = get_attribute_id(node, attributeName);
        if (id < 0) {
            return result;
        }
        const onnx::AttributeProto &attr = node.attribute(id);
        if (attr.type() == onnx::AttributeProto::TENSOR) {
            result = get_floats(attr.t());
        } else if (attr.type() == onnx::AttributeProto::FLOATS) {
            result.resize(attr.floats_size());
            for (int j = 0; j < attr.floats_size(); j++) {
                result[j] = attr.floats(j);
            }
        } else {
            UNI_ERROR_LOG("can not get operator name:%s attribute:%s %s type value.\n",
                this->onnxNode.name().c_str(), attr.name().c_str(), to_string(attr.type()).c_str());
        }
        return result;
    }

    std::string get_string(const onnx::NodeProto &node,
        const char *attributeName,
        const std::string &defaultValue = std::string())
    {
        int id = get_attribute_id(node, attributeName);
        if (id < 0) {
            return defaultValue;
        }
        const onnx::AttributeProto &attr = node.attribute(id);
        if (attr.type() != onnx::AttributeProto::STRING) {
            UNI_ERROR_LOG("can not get operator name:%s attribute:%s %s type value.\n",
                this->onnxNode.name().c_str(), attr.name().c_str(), to_string(attr.type()).c_str());
        }
        return attr.s();
    }

    DataType get_type(const onnx::TensorProto::DataType &type)
    {
        std::map<onnx::TensorProto::DataType, DataType> types = {{onnx::TensorProto::INT64, DT_I64},
            {onnx::TensorProto::INT32, DT_I32}, {onnx::TensorProto::UINT64, DT_U64},
            {onnx::TensorProto::UINT32, DT_U32}, {onnx::TensorProto::UINT8, DT_U8},
            {onnx::TensorProto::INT8, DT_I8}, {onnx::TensorProto::BOOL, DT_U8},
            {onnx::TensorProto::FLOAT, DT_F32}, {onnx::TensorProto::FLOAT16, DT_F16},
            {onnx::TensorProto::UNDEFINED, DT_NUM}};
        DataType ret = DT_F32;
        if (types.find(type) == types.end()) {
            UNI_ERROR_LOG("can not process onnx data type %s.\n", to_string(type).c_str());
        } else {
            ret = types[type];
        }
        return ret;
    }

    DataType get_type(const onnx::TensorProto &tp)
    {
        return get_type((onnx::TensorProto::DataType)tp.data_type());
    }

    DataType cut_type(DataType type)
    {
        DataType ret;
        switch (type) {
            case DT_F64:
            case DT_F32:
            case DT_F16:
                ret = DT_F32;
                break;
            case DT_I32:
            case DT_I64:
            case DT_U32:
            case DT_U64:
                ret = DT_I32;
                break;
            case DT_I8:
                ret = DT_I8;
                break;
            case DT_U8:
                ret = DT_U8;
                break;
            default:
                UNI_ERROR_LOG("can not cut %s type to inner data type.\n", DataTypeName()[type]);
                break;
        }
        return ret;
    }

    U8 *get_ptr(const onnx::TensorProto &tp)
    {
        U8 *ptr = nullptr;
        if (tp.has_raw_data()) {
            const std::string &raw = tp.raw_data();
            ptr = (U8 *)raw.data();
        } else if (tp.data_type() == onnx::TensorProto::FLOAT) {
            ptr = (U8 *)tp.float_data().data();
        } else if (tp.data_type() == onnx::TensorProto::INT64) {
            ptr = (U8 *)tp.int64_data().data();
        } else if (tp.data_type() == onnx::TensorProto::INT32) {
            ptr = (U8 *)tp.int32_data().data();
        } else if (tp.data_type() == onnx::TensorProto::UNDEFINED) {
            ptr = nullptr;
        } else {
            UNI_ERROR_LOG("can not get operator name:%s tensor:%s data, type: %s.\n",
                this->onnxNode.name().c_str(), tp.name().c_str(), to_string(tp.data_type()).c_str());
        }
        return ptr;
    }

    int get_length(const onnx::TensorProto &tp)
    {
        int length = 0;
        if (tp.has_raw_data()) {
            length = tp.raw_data().size() / bytesOf(get_type(tp));
        } else if (tp.data_type() == onnx::TensorProto::FLOAT) {
            length = tp.float_data_size();
        } else if (tp.data_type() == onnx::TensorProto::INT32) {
            length = tp.int32_data_size();
        } else if (tp.data_type() == onnx::TensorProto::INT64) {
            length = tp.int64_data_size();
        } else if (tp.data_type() == onnx::TensorProto::UNDEFINED) {
            length = 0;
        } else {
            UNI_ERROR_LOG("can not get operator name:%s tensor:%s length, type:%s.\n",
                this->onnxNode.name().c_str(), tp.name().c_str(), to_string(tp.data_type()).c_str());
        }
        return length;
    }

    std::vector<int> get_ints(const onnx::TensorProto &tp, std::vector<int> defaultValue = {})
    {
        int length = get_length(tp);
        std::vector<int> data(length);
        transformToInt(get_type(tp), get_ptr(tp), data.data(), length);
        if (data.size() == 0 && defaultValue.size() > 0) {
            data = defaultValue;
        }
        return data;
    }

    std::vector<F32> get_floats(const onnx::TensorProto &tp, std::vector<F32> defaultValue = {})
    {
        int length = get_length(tp);
        std::vector<F32> data(length);
        transformToFloat(get_type(tp), get_ptr(tp), data.data(), length);
        if (data.size() == 0 && defaultValue.size() > 0) {
            data = defaultValue;
        }
        return data;
    }

    TensorDesc get_desc(const onnx::TensorProto &tp)
    {
        TensorDesc desc = tensor0d();
        desc.dt = cut_type(get_type(tp));
        desc.nDims = tp.dims_size();
        desc.df = getTensorDefaultDataFormat(desc.nDims);
        for (U32 j = 0; j < desc.nDims; j++) {
            desc.dims[desc.nDims - 1 - j] = tp.dims(j);
        }
        if (desc.nDims == 0) {
            int length = get_length(tp);
            if (length > 0) {
                desc.nDims = 1;
                desc.df = DF_SCALAR;
                desc.dims[0] = length;
            }
        }
        return desc;
    }

    TensorDesc get_desc(const onnx::ValueInfoProto &vip, std::string name = "")
    {
        const auto &t = vip.type().tensor_type();
        TensorDesc desc;
        desc.dt = cut_type(get_type((onnx::TensorProto::DataType)t.elem_type()));
        desc.nDims = t.shape().dim().size();
        desc.df = getTensorDefaultDataFormat(desc.nDims);
        for (U32 j = 0; j < desc.nDims; j++) {
            int value = t.shape().dim(j).dim_value();
            if (value == 0) {
                std::string param = t.shape().dim(j).dim_param();
                if (param != "") {
                    if (param == "batch") {
                        value = 1;
                    } else if (j == 0) {
                        value = 1;
                    } else {
                        value = UNI_DYNAMIC_SHAPE;
                    }
                    UNI_WARNING_LOG("input:%s has dynamic dimension(%s) on axis(%d), will be set "
                                    "to static value(%d), you can change it at this.\n",
                        name.c_str(), param.c_str(), j, value);
                }
            }
            desc.dims[desc.nDims - 1 - j] = value;
        }
        return desc;
    }

    onnx::TensorProto get_weight(const std::string &name)
    {
        onnx::TensorProto t;
        if (onnxWeights.find(name) != onnxWeights.end()) {
            t = onnxWeights[name];
        }
        return t;
    }

    // dst's dimension is [N, K];
    // src's dimension is [K, N];
    void transformToFloatWithTranspose(DataType dt, const void *src, float *dst, int N, int K)
    {
        for (int r = 0, index = 0; r < N; r++) {
            for (int c = 0; c < K; c++, index++) {
                const void *p = ((const U8 *)src) + (c * N + r) * bytesOf(dt);
                transformToFloat(dt, p, dst + index, 1);
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
        bool ret = false;
        if (multidim > 1) {
            ret = true;
        } else if (multidim == 1) {
            if (wps.dims_size() == 4 && wps.dims(1) == 1) {
                ret = true;
            }
        }
        return ret;
    }

    void copy_tensors(std::vector<onnx::TensorProto> tp, U8 *dst, DataType type)
    {
        for (U32 i = 0; i < tp.size(); i++) {
            U8 *ptr = get_ptr(tp[i]);
            DataType dt = get_type(tp[i]);
            int length = get_length(tp[i]);
            if (type == dt) {
                UNI_MEMCPY(dst, ptr, length * bytesOf(dt));
            } else if (type == DT_I32) {
                transformToInt(dt, ptr, (int *)dst, length);
            } else if (type == DT_F32) {
                transformToFloat(dt, ptr, (float *)dst, length);
            } else {
                UNI_ERROR_LOG(
                    "can not convert %s data to %s.\n", DataTypeName()[dt], DataTypeName()[type]);
            }
            dst += length * bytesOf(dt);
        }
    }

    WeightSpec convert_weight(std::string operatorName,
        std::vector<onnx::TensorProto> weight,
        std::vector<onnx::TensorProto> bias)
    {
        U32 bytes0 = 0, bytes1 = 0;
        DataType wdt = DT_F32, mdt = DT_F32, vdt = DT_F32;
        for (U32 i = 0; i < bias.size(); i++) {
            DataType dt = cut_type(get_type(bias[i]));
            bytes1 += get_length(bias[i]) * bytesOf(dt);
            wdt = vdt = dt;
            UNI_DEBUG_LOG("copy tensor:%s type:%s->%s to bias section.\n", bias[i].name().c_str(),
                to_string(bias[i].data_type()).c_str(), DataTypeName()[dt]);
        }
        for (U32 i = 0; i < weight.size(); i++) {
            DataType dt = cut_type(get_type(weight[i]));
            bytes0 += get_length(weight[i]) * bytesOf(dt);
            wdt = mdt = dt;
            UNI_DEBUG_LOG("copy tensor:%s type:%s->%s to weight section.\n",
                weight[i].name().c_str(), to_string(weight[i].data_type()).c_str(),
                DataTypeName()[dt]);
        }
        WeightSpec w = mt_create_weight(operatorName.c_str(), wdt, bytes0, bytes1, 0);
        copy_tensors(weight, w.weight, mdt);
        copy_tensors(bias, w.vec, vdt);
        return w;
    }

    ResizeMode get_interp_mode(std::string _mode)
    {
        std::string mode = lower(_mode);
        ResizeMode ret;
        if (mode == std::string("linear")) {
            ret = RESIZE_LINEAR;
        } else if (mode == std::string("nearest")) {
            ret = RESIZE_NEAREST;
        } else if (mode == std::string("cubic")) {
            ret = RESIZE_CUBIC;
        } else {
            const std::string &onnxNodeType = this->onnxNode.op_type();
            UNI_ERROR_LOG("can not support interp mode:%s in operator name:%s type:%s.\n",
                _mode.c_str(), this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        return ret;
    }

    EE adapt_operators(ModelSpec *ms) override
    {
        str_copy(ms->model_name, this->modelName.c_str(), this->modelName.length());
        ms->dt = DT_F32;

        ms->num_inputs = 0;
        for (int i = 0; i < onnxGraph.input().size(); i++) {
            const std::string &input_name = onnxGraph.input(i).name();
            if (onnxWeights.find(input_name) != onnxWeights.end()) {
                continue;
            }
            ms->num_inputs++;
        }
        ms->input_names = (I8 **)mt_malloc(ms->num_inputs * sizeof(I8 *));
        ms->input_dims = (TensorDesc *)mt_malloc(sizeof(TensorDesc) * ms->num_inputs);
        for (int i = 0, index = 0; i < onnxGraph.input().size(); i++) {
            auto input_node = onnxGraph.input(i);
            auto input_name = input_node.name();
            if (onnxWeights.find(input_name) != onnxWeights.end()) {
                continue;
            }
            ms->input_names[index] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            input_name = this->crop_name(input_name);
            str_copy(ms->input_names[index], input_name.c_str(), input_name.length());

            ms->input_dims[index] = get_desc(input_node, input_name);
            if (tensorIsShape(ms->input_dims[index])) {
                for (U32 j = 0; j < tensorNumElements(ms->input_dims[index]); j++) {
                    ms->input_dims[index].dims[ms->input_dims[index].nDims + j] = UNI_DYNAMIC_SHAPE;
                }
            }

            if (ms->input_dims[index].nDims == 0) {
                ms->input_dims[index].df = DF_SCALAR;
                ms->input_dims[index].nDims = 1;
                ms->input_dims[index].dims[0] = 1;
                ms->input_dims[index].dims[1] = 1;
            }
            index++;
        }

        ms->num_outputs = onnxGraph.output().size();
        ms->output_names = (I8 **)mt_malloc(ms->num_outputs * sizeof(I8 *));
        for (int i = 0; i < onnxGraph.output().size(); i++) {
            std::string output_name = onnxGraph.output(i).name();
            output_name = this->crop_name(output_name);
            ms->output_names[i] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            str_copy(ms->output_names[i], output_name.c_str(), output_name.length());
        }

        for (int nodeIndex = 0; nodeIndex < onnxGraph.node_size(); nodeIndex++) {
            this->onnxNode = onnxGraph.node(nodeIndex);
            const std::string &onnxNodeType = this->onnxNode.op_type();
            for (int j = 0; j < this->onnxNode.input_size(); j++) {
                const std::string &input_name = this->onnxNode.input(j);
                if (this->onnxWeights.find(input_name) != this->onnxWeights.end()) {
                    if (this->useShare) {
                        if (this->onnxWeightReferCount.find(input_name) ==
                                this->onnxWeightReferCount.end() ||
                            get_length(onnxWeights[input_name]) == 1) {
                            this->onnxWeightReferCount[input_name] = 1;
                        } else {
                            this->onnxWeightReferCount[input_name]++;
                        }
                        if (onnxNodeType == "Gemm" && j == 2) {
                            auto BName = this->onnxNode.input(1);
                            auto CName = this->onnxNode.input(2);
                            this->onnxWeightReferCount[CName] =
                                UNI_MAX(this->onnxWeightReferCount[BName],
                                    this->onnxWeightReferCount[CName]);
                        }
                    } else {
                        this->onnxWeightReferCount[input_name] = 1;
                    }
                }
            }
        }
        for (auto iter = this->onnxWeightReferCount.begin();
             iter != this->onnxWeightReferCount.end(); iter++) {
            if (iter->second > 1) {
                this->sharedWeights.insert(iter->first);
            }
        }

        std::vector<OperatorSpec> ops;
        for (int nodeIndex = 0; nodeIndex < onnxGraph.node_size(); nodeIndex++) {
            this->onnxNode = onnxGraph.node(nodeIndex);
            std::string opName = get_name(this->onnxNode);
            const std::string &onnxNodeType = this->onnxNode.op_type();
            UNI_DEBUG_LOG("process operator name:%s type:%s parameter.\n", opName.c_str(),
                onnxNodeType.c_str());
            if (onnxNodeType == "Constant") {
                handle_Constant();
                continue;
            }
            if (nodeIndex < removePreprocessOpNum) {
                UNI_INFO_LOG("operator name:%s is removed.\n", opName.c_str());
                continue;
            }

            opName = crop_name(opName);
            OperatorType opType = convert_onnx_type(onnxNodeType);

            std::vector<std::string> inputNames, outputNames;
            for (int j = 0; j < this->onnxNode.input_size(); j++) {
                const std::string &input_name = this->onnxNode.input(j);
                if (OT_FC == opType && 2 == j) {
                    if (this->onnxWeights.find(input_name) == this->onnxWeights.end()) {
                        break;
                    }
                }
                if (opType == OT_Gather || opType == OT_Scatter || opType == OT_Einsum ||
                    opType == OT_Eltwise || opType == OT_Concat || opType == OT_MatMul ||
                    opType == OT_Check || opType == OT_Where || opType == OT_ConstantOfShape ||
                    ((opType == OT_Reshape || opType == OT_TfSlice || opType == OT_Pad ||
                         opType == OT_Expand || opType == OT_Tile) &&
                        j == 0)) {
                    inputNames.push_back(input_name);
                } else if (input_name == "" ||
                    this->onnxWeights.find(input_name) != this->onnxWeights.end()) {
                    if (opType == OT_FC) {
                        if (this->onnxWeightReferCount[input_name] > 1) {
                            inputNames.push_back(input_name);
                        }
                    }
                } else {
                    inputNames.push_back(input_name);
                    if (onnxNodeType == "Max" || onnxNodeType == "Min") {
                        break;
                    }
                }
            }
            for (int j = 0; j < this->onnxNode.output_size(); j++) {
                const std::string &output_name = this->onnxNode.output(j);
                outputNames.push_back(output_name);
            }

            // input names order correction
            if (onnxNodeType == "Scan") {
                for (int k = 0; k < (int)(inputNames.size() / 2); k++) {
                    std::string frontStr = inputNames[k];
                    inputNames[k] = inputNames[inputNames.size() - 1 - k];
                    inputNames[inputNames.size() - 1 - k] = frontStr;
                }
                if (outputNames.size() >= 2) {
                    std::string firOutput = outputNames[0];
                    std::string lastOutput = outputNames[outputNames.size() - 1];
                    outputNames.clear();
                    outputNames.push_back(lastOutput);
                    outputNames.push_back(firOutput);
                }
            }

            OperatorSpec os =
                mt_create_operator(opName.c_str(), opType, inputNames.size(), outputNames.size());
            for (U32 j = 0; j < os.num_inputs; j++) {
                inputNames[j] = crop_name(inputNames[j]);
                str_copy(os.input_tensors_name[j], inputNames[j].c_str(), inputNames[j].length());
            }
            for (U32 j = 0; j < os.num_outputs; j++) {
                outputNames[j] = crop_name(outputNames[j]);
                str_copy(os.output_tensors_name[j], outputNames[j].c_str(), outputNames[j].length());
            }
            CHECK_STATUS(adapt_operator(opType, &(os.ps)));
            ops.push_back(os);

            if (onnxNodeType == "BatchNormalization") {
                std::string scaleName = crop_name(opName + "_scale");
                OperatorSpec os = mt_create_operator(scaleName.c_str(), OT_Scale, 1, 1);
                str_copy(os.input_tensors_name[0], outputNames[0].c_str(), outputNames[0].length());
                str_copy(os.output_tensors_name[0], outputNames[0].c_str(), outputNames[0].length());
                CHECK_STATUS(adapt_operator(os.type, &(os.ps)));
                ops.push_back(os);
            }
            if (3 == this->onnxNode.input_size() && "Gemm" == onnxNodeType) {
                if (this->onnxWeightReferCount[this->onnxNode.input(2)] == 0) {
                    std::string matrixCName = this->onnxNode.input(2);
                    std::string addName = crop_name(opName + "_add");
                    OperatorSpec os = mt_create_operator(addName.c_str(), OT_Eltwise, 2, 1);
                    str_copy(
                        os.input_tensors_name[0], outputNames[0].c_str(), outputNames[0].length());
                    str_copy(os.input_tensors_name[1], matrixCName.c_str(), matrixCName.length());
                    str_copy(
                        os.output_tensors_name[0], outputNames[0].c_str(), outputNames[0].length());
                    EltwiseParamSpec p;
                    UNI_MEMSET(&p, 0, sizeof(p));
                    p.mode = ELTWISE_SUM;
                    p.sum_spec.num_coeff = 2;
                    p.sum_spec.coeff[0] = 1.0;
                    p.sum_spec.coeff[1] = 1.0;
                    p.activation_type = ACTIVATION_NULL;
                    os.ps.eltwise_spec = p;
                    ops.push_back(os);
                }
            }
        }

        std::vector<OperatorSpec> sops;
        for (auto iter = this->sharedWeights.begin(); iter != this->sharedWeights.end(); iter++) {
            std::string opName = crop_name("weight_" + *iter);
            OperatorSpec os = mt_create_operator(opName.c_str(), OT_SharedWeight, 0, 1);
            str_copy(os.output_tensors_name[0], iter->c_str(), iter->length());
            const auto &weightTp = onnxWeights[*iter];
            TensorDesc desc = get_desc(weightTp);
            //int num = get_length(weightTp);
            //if (desc.nDims == 0 && num > 0) {
            //    desc.nDims = 1;
            //    desc.dims[0] = num;
            //}
            if (tensorIsShape(desc)) {
                std::vector<int> ptr = get_ints(weightTp);
                for (U32 i = 0; i < tensorNumElements(desc); i++) {
                    desc.dims[desc.nDims + i] = ptr[i];
                }
            }
            os.ps.shared_weight_spec.desc = desc;
            sops.push_back(os);
        }
        ms->num_operator_specs = sops.size() + ops.size();
        ms->ops = (OperatorSpec *)mt_malloc(sizeof(OperatorSpec) * ms->num_operator_specs);
        UNI_MEMCPY(ms->ops, sops.data(), sizeof(OperatorSpec) * sops.size());
        UNI_MEMCPY(ms->ops + sops.size(), ops.data(), sizeof(OperatorSpec) * ops.size());
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            ms->ops[i].tensor_positions = nullptr;
            ms->ops[i].num_quant_feature = 0;
            ms->ops[i].feature_scale = nullptr;
        }
        return SUCCESS;
    }

    DataType use_bnn(const onnx::NodeProto &node)
    {
        if (!useBNN) {
            return DT_F32;
        }
        int weight_id = 1;
        if (onnxWeights.find(node.input(weight_id)) == onnxWeights.end()) {
            return DT_F32;
        }
        auto weight = onnxWeights[node.input(weight_id)];
        int length = get_length(weight);
        if (1 >= length) {
            return DT_F32;
        }
        int oc = weight.dims(0);
        int ic = weight.dims(1);
        int fhfw = 1;
        std::vector<int> kernel = get_ints(node, "kernel_shape");
        for (U32 i = 0; i < kernel.size(); i++) {
            fhfw *= kernel[i];
        }
        if (ic % 32 != 0 || oc % 16 != 0 || fhfw % 8 != 1) {
            UNI_WARNING_LOG("operator name:%s can not use 1-bit calculation, because 1-bit only "
                            "support input_channel(%d) mod 32 = 0, output_channel(%d) mod 16 = 0, "
                            "and fhfw(%d) mod 8 = 1.\n",
                node.name().c_str(), ic, oc, fhfw);
            return DT_F32;
        }
        int count0 = 0, count1 = 0, count_1 = 0;
        float value;
        U8 *ptr = get_ptr(weight);
        DataType dt = get_type(weight);
        for (int i = 0; i < length; i++) {
            transformToFloat(dt, ptr, &value, 1);
            ptr += bytesOf(dt);
            if (value == 0) {
                count0++;
            } else if (value == 1) {
                count1++;
            } else if (value == -1) {
                count_1++;
            }
        }
        if (count0 + count1 == length) {
            return DT_BIN01;
        }
        if (count_1 + count1 == length) {
            return DT_BIN11;
        }
        UNI_WARNING_LOG("operator name:%s can not use 1-bit calculation, because weight is not 0/1 "
                        "or -1/1.\n",
            node.name().c_str());
        return DT_F32;
    }

    EE adapt_weights(ModelSpec *ms) override
    {
        std::vector<WeightSpec> ws;
        for (auto iter = this->sharedWeights.begin(); iter != this->sharedWeights.end(); iter++) {
            ws.push_back(convert_weight(crop_name("weight_" + *iter), {onnxWeights[*iter]}, {}));
        }
        for (int nodeIndex = 0; nodeIndex < onnxGraph.node_size(); nodeIndex++) {
            this->onnxNode = onnxGraph.node(nodeIndex);
            std::string opName = get_name(this->onnxNode);
            const std::string &onnxNodeType = this->onnxNode.op_type();
            UNI_DEBUG_LOG(
                "process operator name:%s type:%s weight.\n", opName.c_str(), onnxNodeType.c_str());
            opName = crop_name(opName);
            auto indices = get_weight_ids(this->onnxNode);

            WeightSpec weightSpec;
            if (onnxNodeType == "Conv" || onnxNodeType == "ConvTranspose") {
                weightSpec = convert_weight(opName, {onnxWeights[this->onnxNode.input(1)]}, {});
                weightSpec.mdt = use_bnn(this->onnxNode);
                if (this->onnxNode.input_size() == 3) {
                    auto &bias = onnxWeights[this->onnxNode.input(2)];
                    DataType dt = get_type(bias);
                    int length = get_length(bias);
                    U8 *ptr = get_ptr(bias);
                    float *p;
                    weightSpec.bytes_of_vec = length * sizeof(float);
                    // BNN conv must have a scale vector and a bias vector, so that it can fuse with BN
                    if (DT_BIN11 == weightSpec.mdt || DT_BIN01 == weightSpec.mdt) {
                        weightSpec.bytes_of_vec *= 2;
                        p = (float *)mt_malloc(weightSpec.bytes_of_vec);
                        UNI_INIT(length, DT_F32, 1, p);
                        // Copy bias (if any) to the second half for BNN
                        transformToFloat(dt, ptr, p + length, length);
                    } else {
                        p = (float *)mt_malloc(weightSpec.bytes_of_vec);
                        transformToFloat(dt, ptr, p, length);
                    }
                    weightSpec.vec = (U8 *)p;
                }
                ws.push_back(weightSpec);
            } else if (onnxNodeType == "Gemm" || onnxNodeType == "Linear") {
                if (onnxNodeType == "Linear" || (this->onnxNode.input_size() < 3) ||
                    this->onnxWeightReferCount[this->onnxNode.input(2)] > 1 ||
                    this->onnxWeightReferCount[this->onnxNode.input(2)] == 0) {
                    weightSpec = convert_weight(opName, {}, {});
                } else {
                    weightSpec = convert_weight(opName, {}, {onnxWeights[this->onnxNode.input(2)]});
                }
                int transB = 1;
                if (onnxNodeType == "Gemm") {
                    transB = get_int(this->onnxNode, "transB", 0);
                }
                if (this->onnxWeightReferCount[this->onnxNode.input(1)] <= 1 &&
                    onnxWeights.count(this->onnxNode.input(1))) {
                    const onnx::TensorProto &weight = onnxWeights[this->onnxNode.input(1)];
                    int length = get_length(weight);
                    DataType dt = get_type(weight);
                    U8 *ptr = get_ptr(weight);
                    weightSpec.bytes_of_weight = length * sizeof(float);
                    weightSpec.weight = (U8 *)mt_malloc(weightSpec.bytes_of_weight);
                    if (transB) {
                        transformToFloat(dt, ptr, (float *)weightSpec.weight, length);
                    } else {
                        transformToFloatWithTranspose(
                            dt, ptr, (float *)weightSpec.weight, weight.dims(1), weight.dims(0));
                    }
                }
                ws.push_back(weightSpec);
            } else if (onnxNodeType == "BatchNormalization") {
                ws.push_back(convert_weight(opName, {onnxWeights[this->onnxNode.input(3)]},
                    {onnxWeights[this->onnxNode.input(4)]}));
                std::string scaleName = crop_name(opName + "_scale");
                ws.push_back(convert_weight(scaleName, {onnxWeights[this->onnxNode.input(1)]},
                    {onnxWeights[this->onnxNode.input(2)]}));
            } else if (onnxNodeType == "BatchNorm" || onnxNodeType == "InstanceNormalization") {
                ws.push_back(convert_weight(opName, {onnxWeights[this->onnxNode.input(1)]},
                    {onnxWeights[this->onnxNode.input(2)]}));
            } else if (onnxNodeType == "Tdnn") {
                ws.push_back(convert_weight(opName, {onnxWeights[this->onnxNode.input(2)]},
                    {onnxWeights[this->onnxNode.input(3)]}));
            } else if ((onnxNodeType == "MatMul" || onnxNodeType == "PRelu") && indices.size() > 0) {
                weightSpec = convert_weight(opName, {}, {});
                std::string weightName = this->onnxNode.input(1);
                if (onnxNodeType != "MatMul" ||
                    (this->onnxWeightReferCount[weightName] <= 1 &&
                        this->sharedWeights.find(weightName) == this->sharedWeights.end())) {
                    const onnx::TensorProto &weight = onnxWeights[weightName];
                    int length = get_length(weight);
                    weightSpec.bytes_of_weight = length * sizeof(float);
                    weightSpec.weight = (U8 *)mt_malloc(weightSpec.bytes_of_weight);
                    int row = weight.dims(0);
                    transformToFloatWithTranspose(get_type(weight), get_ptr(weight),
                        (float *)weightSpec.weight, length / row, row);
                }
                ws.push_back(weightSpec);
            } else if (onnxNodeType == "Mul" || onnxNodeType == "Div") {
                OperatorType type = convert_onnx_type(onnxNodeType);
                if (indices.size() == 0 || type == OT_Power || type == OT_Eltwise) {
                    continue;
                }
                weightSpec =
                    convert_weight(opName, {onnxWeights[this->onnxNode.input(indices[0])]}, {});
                if (onnxNodeType == "Div") {
                    F32 *scale = (F32 *)weightSpec.weight;
                    for (U32 j = 0; j < weightSpec.bytes_of_weight / sizeof(float); j++) {
                        scale[j] = 1 / scale[j];
                    }
                }
                ws.push_back(weightSpec);
            } else if (onnxNodeType == "Add" || onnxNodeType == "Sub") {
                OperatorType type = convert_onnx_type(onnxNodeType);
                if (indices.size() == 0 || type == OT_Power || type == OT_Eltwise) {
                    continue;
                }
                weightSpec =
                    convert_weight(opName, {}, {onnxWeights[this->onnxNode.input(indices[0])]});
                if (onnxNodeType == "Sub") {
                    F32 *scale = (F32 *)weightSpec.vec;
                    for (U32 j = 0; j < weightSpec.bytes_of_vec / sizeof(float); j++) {
                        scale[j] = -1 * scale[j];
                    }
                }
                ws.push_back(weightSpec);
            } else if (onnxNodeType == "Transpose" && indices.size() > 0) {
                ws.push_back(convert_weight(opName, {onnxWeights[this->onnxNode.input(0)]}, {}));
            } else if (onnxNodeType == "RNN" || onnxNodeType == "GRU" || onnxNodeType == "LSTM") {
                const onnx::TensorProto &W = onnxWeights[this->onnxNode.input(1)];
                const onnx::TensorProto &R = onnxWeights[this->onnxNode.input(2)];
                const onnx::TensorProto &B = onnxWeights[this->onnxNode.input(3)];
                if (W.dims_size() != 3 || R.dims_size() != 3) {
                    UNI_ERROR_LOG("can not process operator name:%s type:%s W.dims_size() != "
                                  "3 || R.dims_size() != 3.\n",
                        this->onnxNode.name().c_str(), onnxNodeType.c_str());
                }
                if (W.dims(0) != R.dims(0) || W.dims(1) != R.dims(1)) {
                    UNI_ERROR_LOG("can not process operator name:%s type:%s W.dims(0) != "
                                  "R.dims(0) || W.dims(1) != R.dims(1).\n",
                        this->onnxNode.name().c_str(), onnxNodeType.c_str());
                }
                int biasNum = (B.dims_size() == 0) ? 0 : 1;
                for (int j = 0; j < B.dims_size(); j++) {
                    biasNum *= B.dims(j);
                }
                // reorganize bias
                if (biasNum % 2 != 0) {
                    UNI_ERROR_LOG("can not process operator name:%s type:%s bias.\n",
                        this->onnxNode.name().c_str(), onnxNodeType.c_str());
                }
                biasNum /= 2;
                int gates = 0;
                std::vector<int> order;
                bool gru_lbr = false;
                if (onnxNodeType == "RNN") {
                    gates = 1;
                    order = {0};
                } else if (onnxNodeType == "GRU") {
                    gates = 3;
                    order = {0, 1, 2};
                    if (0 != get_int(this->onnxNode, "linear_before_reset", 0)) {
                        gru_lbr = true;
                        biasNum += biasNum / gates;
                    }
                } else if (onnxNodeType == "LSTM") {
                    gates = 4;
                    order = {0, 3, 2, 1};
                } else {
                    UNI_ERROR_LOG("can not process operator name:%s type:%s.\n",
                        this->onnxNode.name().c_str(), onnxNodeType.c_str());
                }
                weightSpec = mt_create_weight(opName.c_str(), DT_F32,
                    (W.dims(0) * W.dims(1) * (W.dims(2) + R.dims(2))) * sizeof(float),
                    biasNum * sizeof(float), 0);
                int hidden = W.dims(1) / gates;
                F32 *weightPtr = (F32 *)weightSpec.weight;
                F32 *biasPtr = (F32 *)weightSpec.vec;
                U8 *W_ptr = get_ptr(W);
                U8 *R_ptr = get_ptr(R);
                U8 *B_ptr = get_ptr(B);
                auto W_dt = get_type(W);
                auto R_dt = get_type(R);
                auto B_dt = get_type(B);
                // loop direction
                for (int j = 0; j < W.dims(0); j++) {
                    // loop LSTM(iofc), GRU(zrh), RNN(g)
                    for (int m = 0; m < gates; m++) {
                        int k = order[m];
                        for (int n = 0; n < hidden; n++) {
                            transformToFloat(W_dt,
                                W_ptr + ((j * gates + k) * hidden + n) * W.dims(2) * bytesOf(W_dt),
                                weightPtr, W.dims(2));
                            weightPtr += W.dims(2);
                            transformToFloat(R_dt,
                                R_ptr + ((j * gates + k) * hidden + n) * R.dims(2) * bytesOf(R_dt),
                                weightPtr, R.dims(2));
                            weightPtr += R.dims(2);

                            if (biasNum > 0) {
                                float W_B, R_B;
                                transformToFloat(B_dt,
                                    B_ptr + (((j * 2) * gates + k) * hidden + n) * bytesOf(B_dt),
                                    &W_B, 1);
                                transformToFloat(B_dt,
                                    B_ptr + (((j * 2 + 1) * gates + k) * hidden + n) * bytesOf(B_dt),
                                    &R_B, 1);
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
                ws.push_back(weightSpec);
            } else if (onnxNodeType == "Splice") {
                std::vector<int> indices = get_ints(this->onnxNode, "forward_indexes");
                weightSpec =
                    mt_create_weight(opName.c_str(), DT_U32, indices.size() * sizeof(int), 0, 0);
                UNI_MEMCPY(weightSpec.weight, indices.data(), weightSpec.bytes_of_weight);
                ws.push_back(weightSpec);
                //} else if (onnxNodeType == "Where") {
                //    if (onnxWeights.find(this->onnxNode.input(2)) != onnxWeights.end()) {
                //        weightSpec = convert_weight(opName, {}, {onnxWeights[this->onnxNode.input(2)]});
                //    } else {
                //        weightSpec = convert_weight(opName, {}, {});
                //    }
                //    if (onnxWeights.find(this->onnxNode.input(0)) != onnxWeights.end()) {
                //        auto &condition = onnxWeights[this->onnxNode.input(0)];
                //        int length = get_length(condition);
                //        weightSpec.bytes_of_weight = length * sizeof(float);
                //        weightSpec.weight = (U8 *)mt_new_storage(weightSpec.bytes_of_weight);
                //        transformToFloat(DT_I8, get_ptr(condition), (float *)weightSpec.weight, length);
                //    }
                //    ws.push_back(weightSpec);
                //} else if (onnxNodeType == "Less" || onnxNodeType == "LessOrEqual" ||
                //    onnxNodeType == "Equal" || onnxNodeType == "Greater" ||
                //    onnxNodeType == "GreaterOrEqual") {
                //    if (indices.size() > 0) {
                //        ws.push_back(convert_weight(opName, {onnxWeights[this->onnxNode.input(indices[0])]}, {}));
                //    }
            } else if (onnxNodeType == "Scan") {
                onnx::GraphProto gp;
                for (int k = 0; k < this->onnxNode.attribute_size(); k++) {
                    const onnx::AttributeProto &attribute = this->onnxNode.attribute(k);
                    if (attribute.name() == "body") {
                        gp = attribute.g();
                        break;
                    }
                }
                // extract the weight from scan tp
                std::map<std::string, std::vector<onnx::TensorProto>> weightMap = {
                    {"Gemm", {}},
                    {"MatMul", {}},
                };
                std::map<std::string, std::vector<int>> transMap = {{"Gemm", {}}, {"MatMul", {}}};
                for (int j = 0; j < gp.node_size(); j++) {
                    auto node = gp.node(j);
                    auto type = node.op_type();
                    if (type == "Gemm" || type == "MatMul") {
                        for (int k = 0; k < node.input_size(); k++) {
                            auto name = node.input(k);
                            if (onnxWeights.find(name) != onnxWeights.end()) {
                                auto weight = onnxWeights[name];
                                if (get_length(weight) == 0) {
                                    continue;
                                } else {
                                    weightMap[type].push_back(weight);
                                }
                            }
                        }
                        transMap[type].push_back(get_int(node, "transB", 0));
                    }
                }
                std::vector<onnx::TensorProto> bias;
                if (weightMap["Gemm"].size() > 1) {
                    bias.push_back(weightMap["Gemm"][1]);
                }
                if (weightMap["MatMul"].size() > 1) {
                    bias.push_back(weightMap["MatMul"][1]);
                }
                weightSpec = convert_weight(opName, {}, bias);

                U32 length1 = 0, length2 = 0;
                if (weightMap["Gemm"].size() > 0) {
                    length1 = get_length(weightMap["Gemm"][0]);
                }
                if (weightMap["MatMul"].size() > 0) {
                    length2 = get_length(weightMap["MatMul"][0]);
                }
                weightSpec.bytes_of_weight = (length1 + length2) * sizeof(float);
                if (weightSpec.bytes_of_weight > 0) {
                    weightSpec.weight = (U8 *)mt_malloc(weightSpec.bytes_of_weight);
                    if (length1 > 0) {
                        U8 *ptr = get_ptr(weightMap["Gemm"][0]);
                        DataType dt = get_type(weightMap["Gemm"][0]);
                        if (transMap["Gemm"][0]) {
                            transformToFloat(dt, ptr, (float *)weightSpec.weight, length1);
                        } else {
                            transformToFloatWithTranspose(dt, ptr, (float *)weightSpec.weight,
                                weightMap["Gemm"][0].dims(1), weightMap["Gemm"][0].dims(0));
                        }
                    }
                    if (length2 > 0) {
                        U8 *ptr = get_ptr(weightMap["MatMul"][0]);
                        DataType dt = get_type(weightMap["Gemm"][0]);
                        float *p = (float *)weightSpec.weight + length1;
                        if (transMap["MatMul"][0]) {
                            transformToFloat(dt, ptr, p, length2);
                        } else {
                            transformToFloatWithTranspose(dt, ptr, p,
                                weightMap["MatMul"][0].dims(1), weightMap["MatMul"][0].dims(0));
                        }
                    }
                }
                ws.push_back(weightSpec);
            } else if (onnxNodeType == "GenerateProposals") {
                ws.push_back(convert_weight(opName, {onnxWeights[this->onnxNode.input(3)]}, {}));
            }
        }
        ms->num_weight_specs = ws.size();
        ms->ws = (WeightSpec *)mt_malloc(sizeof(WeightSpec) * ms->num_weight_specs);
        UNI_MEMCPY(ms->ws, ws.data(), sizeof(WeightSpec) * ws.size());
        return SUCCESS;
    }

    void add_shared_weight(const onnx::NodeProto &node, std::set<int> input_ids = std::set<int>())
    {
        for (int i = 0; i < this->onnxNode.input_size(); i++) {
            if (input_ids.size() > 0 && input_ids.find(i) == input_ids.end()) {
                continue;
            }
            const std::string &name = this->onnxNode.input(i);
            if (onnxWeights.find(name) != onnxWeights.end()) {
                this->sharedWeights.insert(name);
            }
        }
    }

    ParameterSpec adapt_Reshape() override
    {
        ParameterSpec ps;
        ReshapeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        std::vector<int> shape;
        if (this->onnxNode.input_size() == 1) {
            shape = get_ints(this->onnxNode, "shape");
        } else {
            shape = get_ints(get_weight(onnxNode.input(1)));
        }
        p.num_shape = shape.size();
        UNI_MEMCPY(p.shape, shape.data(), p.num_shape * sizeof(I32));
        p.axis = 0;
        p.num_axes = -1;
        ps.reshape_spec = p;
        add_shared_weight(this->onnxNode, {0});
        return ps;
    }

    ParameterSpec adapt_Flatten() override
    {
        ParameterSpec ps;
        FlattenParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = get_int(this->onnxNode, "axis", 1);
        ps.flatten_spec = p;
        return ps;
    }

    ParameterSpec adapt_Resize() override
    {
        ParameterSpec ps;
        ResizeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<float> scales;
        std::vector<int> sizes;
        std::string scalesIndex = "";
        std::string sizesIndex = "";
        std::string mode = get_string(this->onnxNode, "mode", "nearest");
        std::string trans_mode =
            get_string(this->onnxNode, "coordinate_transformation_mode", "half_pixel");
        std::string nearest_mode = get_string(this->onnxNode, "nearest_mode", "round_prefer_floor");
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Resize") {
            sizes = get_ints(this->onnxNode, "sizes");
            for (int i = 0; i < this->onnxNode.input_size(); i++) {
                if (onnxWeights.find(this->onnxNode.input(i)) != onnxWeights.end()) {
                    auto tp = onnxWeights[this->onnxNode.input(i)];
                    if (tp.data_type() == onnx::TensorProto::FLOAT) {
                        scales = get_floats(tp);
                    } else if (tp.data_type() == onnx::TensorProto::INT64) {
                        sizes = get_ints(tp);
                    } else {
                        UNI_ERROR_LOG("can not process operator name:%s %s type attributes.\n",
                            this->onnxNode.name().c_str(), to_string(tp.data_type()).c_str());
                    }
                }
            }
        } else if (onnxNodeType == "Upsample") {
            if (this->onnxNode.input_size() > 1) {
                scales = get_floats(onnxWeights[this->onnxNode.input(1)]);
            } else {
                scales = get_floats(this->onnxNode, "scales");
            }
            trans_mode = "asymmetric";
            nearest_mode = "floor";
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Resize.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        if (scales.size() > 0) {
            p.num_scales = scales.size();
            UNI_MEMCPY(p.scales, scales.data(), p.num_scales * bytesOf(DT_F32));
        }
        if (sizes.size() > 0) {
            p.num_sizes = 0;
            if (sizes.size() > 2) {
                p.num_sizes = sizes.size() - 2;
                UNI_MEMCPY(p.sizes, sizes.data() + 2, p.num_sizes * bytesOf(DT_I32));
            }
        }

        p.mode = get_interp_mode(mode);

        if (trans_mode == std::string("align_corners")) {
            p.trans_mode = COORDINATE_TRANS_ALIGN_CORNERS;
        } else if (trans_mode == std::string("half_pixel")) {
            p.trans_mode = COORDINATE_TRANS_HALF_PIXEL;
        } else if (trans_mode == std::string("pytorch_half_pixel")) {
            p.trans_mode = COORDINATE_TRANS_PYTORCH_HALF_PIXEL;
        } else if (trans_mode == std::string("asymmetric")) {
            p.trans_mode = COORDINATE_TRANS_ASYMMETRIC;
        } else {
            UNI_ERROR_LOG("can not support coordinate transformation mode:%s in operator name:%s "
                          "type:%s.\n",
                trans_mode.c_str(), this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }

        if (nearest_mode == std::string("round_prefer_floor")) {
            p.round_mode = ROUND_PREFER_FLOOR;
        } else if (nearest_mode == std::string("round_prefer_ceil")) {
            p.round_mode = ROUND_PREFER_CEIL;
        } else if (nearest_mode == std::string("floor")) {
            p.round_mode = ROUND_FLOOR;
        } else {
            p.round_mode = ROUND_CEIL;
        }
        ps.resize_spec = p;
        return ps;
    }

    ParameterSpec adapt_Transpose() override
    {
        ParameterSpec ps;
        TransposeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> axes = get_ints(this->onnxNode, "perm");
        p.num_axes = axes.size();
        UNI_MEMCPY(p.axes, axes.data(), p.num_axes * sizeof(U32));
        ps.transpose_spec = p;
        return ps;
    }

    ParameterSpec adapt_Clip() override
    {
        ParameterSpec ps;
        ClipParamSpec p;
        float max_inf = 1.0 / 0.0;
        float min_inf = -1.0 / 0.0;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Max") {
            if (onnxWeights.find(this->onnxNode.input(1)) != onnxWeights.end()) {
                p.min = get_floats(onnxWeights[this->onnxNode.input(1)])[0];
            } else {
                p.min = 0.0;
            }
            p.max = max_inf;
        } else if (onnxNodeType == "Min") {
            p.min = min_inf;
            if (onnxWeights.find(this->onnxNode.input(1)) != onnxWeights.end()) {
                p.max = get_floats(onnxWeights[this->onnxNode.input(1)])[0];
            } else {
                p.max = 1.0;
            }
        } else {
            if (this->onnxNode.input_size() == 1) {
                p.min = get_float(this->onnxNode, "min", -UNI_F16_MAX);
                p.max = get_float(this->onnxNode, "max", UNI_F16_MAX);
            } else {
                p.min = (this->onnxNode.input(1) == "")
                    ? min_inf
                    : get_floats(onnxWeights[this->onnxNode.input(1)])[0];
                p.max = (this->onnxNode.input(2) == "")
                    ? max_inf
                    : get_floats(onnxWeights[this->onnxNode.input(2)])[0];
            }
        }
        ps.clip_spec = p;
        return ps;
    }

    ParameterSpec adapt_Conv() override
    {
        ParameterSpec ps;
        ConvolutionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::string autoPad = get_string(this->onnxNode, "auto_pad");
        std::vector<int> kernels = get_ints(this->onnxNode, "kernel_shape");
        std::vector<int> dilations = get_ints(this->onnxNode, "dilations");
        std::vector<int> strides = get_ints(this->onnxNode, "strides");
        std::vector<int> pads = get_ints(this->onnxNode, "pads");
        int group = get_int(this->onnxNode, "group", 1);

        const onnx::TensorProto &weight = onnxWeights[this->onnxNode.input(1)];
        int num_inputs = weight.dims(1);
        p.num_outputs = weight.dims(0);
        p.num_outputs_origin = p.num_outputs;
        p.kernel_t = 1;
        p.kernel_h = 1;
        p.kernel_w = 1;
        if (kernels.size() == 3) {
            p.kernel_t = kernels[0];
            p.kernel_h = kernels[1];
            p.kernel_w = kernels[2];
        } else if (kernels.size() == 2) {
            p.kernel_h = kernels[0];
            p.kernel_w = kernels[1];
        } else if (kernels.size() == 1) {
            p.kernel_h = kernels[0];
        }

        p.dilatedRate_t = 1;
        p.dilatedRate_h = 1;
        p.dilatedRate_w = 1;
        if (dilations.size() == 3) {
            p.dilatedRate_t = dilations[0];
            p.dilatedRate_h = dilations[1];
            p.dilatedRate_w = dilations[2];
        } else if (dilations.size() == 2) {
            p.dilatedRate_h = dilations[0];
            p.dilatedRate_w = dilations[1];
        } else if (dilations.size() == 1) {
            p.dilatedRate_h = dilations[0];
        }

        p.stride_t = 1;
        p.stride_h = 1;
        p.stride_w = 1;
        if (strides.size() == 3) {
            p.stride_t = strides[0];
            p.stride_h = strides[1];
            p.stride_w = strides[2];
        } else if (strides.size() == 2) {
            p.stride_h = strides[0];
            p.stride_w = strides[1];
        } else if (strides.size() == 1) {
            p.stride_h = strides[0];
        }

        p.pad_before = 0;
        p.pad_top = 0;
        p.pad_left = 0;
        p.pad_after = 0;
        p.pad_bottom = 0;
        p.pad_right = 0;
        p.round_mode = ROUND_CEIL;
        if (pads.size() == 6) {
            p.pad_before = pads[0];
            p.pad_top = pads[1];
            p.pad_left = pads[2];
            p.pad_after = pads[3];
            p.pad_bottom = pads[4];
            p.pad_right = pads[5];
        } else if (pads.size() == 4) {
            p.pad_top = pads[0];
            p.pad_left = pads[1];
            p.pad_bottom = pads[2];
            p.pad_right = pads[3];
        } else if (pads.size() == 2) {
            p.pad_top = pads[0];
            p.pad_bottom = pads[1];
        } else if (autoPad == "SAME_UPPER") {
            p.pad_top = (p.kernel_h - 1) / 2;
            p.pad_bottom = (p.kernel_h - 1) - p.pad_top;
            p.pad_left = (p.kernel_w - 1) / 2;
            p.pad_right = (p.kernel_w - 1) - p.pad_left;
            p.round_mode = ROUND_SAME_UPPER;
        }

        p.group = group;
        if (p.group != 1 && p.group == p.num_outputs && num_inputs == 1) {
            p.convolution_type = CONVOLUTION_DEPTHWISE;
        } else {
            p.convolution_type = CONVOLUTION_POINTWISE;
        }

        p.dw_activation_type = ACTIVATION_NULL;
        p.pw_activation_type = ACTIVATION_NULL;
        ps.conv_spec = p;
        return ps;
    }

    ParameterSpec adapt_Deconvolution() override
    {
        const std::string &onnxNodeType = this->onnxNode.op_type();
        ParameterSpec ps;
        ConvolutionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::string autoPad = get_string(this->onnxNode, "auto_pad", "NOTSET");
        std::vector<int> kernels = get_ints(this->onnxNode, "kernel_shape");
        std::vector<int> dilations = get_ints(this->onnxNode, "dilations");
        std::vector<int> strides = get_ints(this->onnxNode, "strides");
        std::vector<int> pads = get_ints(this->onnxNode, "pads");
        int group = get_int(this->onnxNode, "group", 1);
        std::vector<int> output_padding = get_ints(this->onnxNode, "output_padding");
        std::vector<int> output_shapes = get_ints(this->onnxNode, "output_shape");

        const onnx::TensorProto &weight = onnxWeights[this->onnxNode.input(1)];
        p.num_outputs = weight.dims(1);
        p.kernel_t = 1;
        p.kernel_h = 1;
        p.kernel_w = 1;
        if (kernels.size() == 3) {
            p.kernel_t = kernels[0];
            p.kernel_h = kernels[1];
            p.kernel_w = kernels[2];
        } else if (kernels.size() == 2) {
            p.kernel_h = kernels[0];
            p.kernel_w = kernels[1];
        } else if (kernels.size() == 1) {
            p.kernel_h = kernels[0];
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Deconvolution.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }

        p.dilatedRate_t = 1;
        p.dilatedRate_h = 1;
        p.dilatedRate_w = 1;
        if (dilations.size() == 3) {
            p.dilatedRate_t = dilations[0];
            p.dilatedRate_h = dilations[1];
            p.dilatedRate_w = dilations[2];
        } else if (dilations.size() == 2) {
            p.dilatedRate_h = dilations[0];
            p.dilatedRate_w = dilations[1];
        } else if (dilations.size() == 1) {
            p.dilatedRate_h = dilations[0];
        }

        p.stride_t = 1;
        p.stride_h = 1;
        p.stride_w = 1;
        if (strides.size() == 3) {
            p.stride_t = strides[0];
            p.stride_h = strides[1];
            p.stride_w = strides[2];
        } else if (strides.size() == 2) {
            p.stride_h = strides[0];
            p.stride_w = strides[1];
        } else if (strides.size() == 1) {
            p.stride_h = strides[0];
        }

        p.round_mode = ROUND_CEIL;
        if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
            p.round_mode = ROUND_TF_SAME;
        }
        //if (output_shapes.size() > 0) {
        //    U32 count = 0;
        //    for (U32 i = 0; i < output_shapes.size(); i++) {
        //        if (output_shapes[i] % strides[i] == 0) {
        //            count++;
        //        }
        //    }
        //    if (count == output_shapes.size()) {
        //        p.round_mode = ROUND_TF_SAME;
        //    }
        //}
        TensorDesc inputDesc = tensor0d();
        if (onnxValues.find(this->onnxNode.input(0)) != onnxValues.end()) {
            inputDesc = get_desc(onnxValues[this->onnxNode.input(0)]);
        }
        if (inputDesc.nDims > 0 && output_shapes.size() > 0) {
            if (inputDesc.nDims > 2) {
                int ih = inputDesc.dims[inputDesc.nDims - 3];
                if (output_shapes[0] == (int)p.stride_h * ih) {
                    if (output_shapes.size() > 1) {
                        int iw = inputDesc.dims[inputDesc.nDims - 4];
                        if (output_shapes[1] == (int)p.stride_w * iw) {
                            p.round_mode = ROUND_TF_SAME;
                        }
                    } else {
                        p.round_mode = ROUND_TF_SAME;
                    }
                }
            }
        }
        if (pads.size() == 0 && inputDesc.nDims > 0 && p.round_mode == ROUND_CEIL) {
            unsigned int dim = kernels.size();
            pads = std::vector<int>(dim * 2);
            std::vector<int> input_size;
            for (int i = inputDesc.nDims - 3; i >= 0; i--) {
                input_size.push_back(inputDesc.dims[i]);
            }
            CHECK_REQUIREMENT(dim == input_size.size());
            CHECK_REQUIREMENT(dim == output_shapes.size());
            if (strides.size() == 0) {
                strides = std::vector<int>(dim, 1);
            }
            if (dilations.size() == 0) {
                dilations = std::vector<int>(dim, 1);
            }
            if (output_padding.size() == 0) {
                output_padding = std::vector<int>(dim, 0);
            }
            for (unsigned int i = 0; i < dim; i++) {
                int total_padding = strides[i] * (input_size[i] - 1) + output_padding[i] +
                    ((kernels[i] - 1) * dilations[i] + 1) - output_shapes[i];
                if (autoPad == "SAME_UPPER") {
                    pads[i] = total_padding / 2;
                    pads[i + dim] = total_padding - (total_padding / 2);
                } else {
                    pads[i] = total_padding - (total_padding / 2);
                    pads[i + dim] = (total_padding / 2);
                }
            }
        }
        p.pad_before = 0;
        p.pad_after = 0;
        p.pad_top = 0;
        p.pad_bottom = 0;
        p.pad_left = 0;
        p.pad_right = 0;
        if (pads.size() == 6) {
            p.pad_before = pads[0];
            p.pad_top = pads[1];
            p.pad_left = pads[2];
            p.pad_after = pads[3];
            p.pad_bottom = pads[4];
            p.pad_right = pads[5];
        } else if (pads.size() == 4) {
            p.pad_top = pads[0];
            p.pad_left = pads[1];
            p.pad_bottom = pads[2];
            p.pad_right = pads[3];
        } else if (pads.size() == 2) {
            p.pad_top = pads[0];
            p.pad_bottom = pads[1];
            p.pad_left = 0;
            p.pad_right = 0;
        }
        p.output_pad_t = 0;
        p.output_pad_h = 0;
        p.output_pad_w = 0;
        if (output_padding.size() == 3) {
            p.output_pad_t = output_padding[0];
            p.output_pad_h = output_padding[1];
            p.output_pad_w = output_padding[2];
        } else if (output_padding.size() == 2) {
            p.output_pad_h = output_padding[0];
            p.output_pad_w = output_padding[1];
        } else if (output_padding.size() == 1) {
            p.output_pad_h = output_padding[0];
        }

        p.group = group;
        if (1 == group) {
            p.convolution_type = CONVOLUTION_DECONVOLUTION;
        } else {
            p.convolution_type = CONVOLUTION_DEPTHWISE_DECONVOLUTION;
            p.num_outputs = weight.dims(0);
        }
        p.num_outputs_origin = p.num_outputs;
        p.dw_activation_type = ACTIVATION_NULL;
        p.pw_activation_type = ACTIVATION_NULL;
        ps.conv_spec = p;
        return ps;
    }

    ParameterSpec adapt_Pooling() override
    {
        ParameterSpec ps;
        PoolingParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::string autoPad = get_string(this->onnxNode, "auto_pad");
        std::vector<int> kernels = get_ints(this->onnxNode, "kernel_shape");
        std::vector<int> strides = get_ints(this->onnxNode, "strides");
        std::vector<int> pads = get_ints(this->onnxNode, "pads");
        int ceil_mode = get_int(this->onnxNode, "ceil_mode", 0);
        p.count_include_pad = get_int(this->onnxNode, "count_include_pad", 0);

        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "AveragePool" || onnxNodeType == "ReduceMean" ||
            onnxNodeType == "GlobalAveragePool") {
            p.mode = POOLING_MEAN;
        } else {
            p.mode = POOLING_MAX;
        }

        if (ceil_mode) {
            p.round_mode = ROUND_CEIL;
        } else {
            p.round_mode = ROUND_FLOOR;
        }

        p.kernel_t = 0;
        p.kernel_h = 0;
        p.kernel_w = 0;
        if (kernels.size() == 3) {
            p.kernel_t = kernels[0];
            p.kernel_h = kernels[1];
            p.kernel_w = kernels[2];
        } else if (kernels.size() == 2) {
            p.kernel_t = 1;
            p.kernel_h = kernels[0];
            p.kernel_w = kernels[1];
        } else if (kernels.size() == 1) {
            p.kernel_t = 1;
            p.kernel_h = kernels[0];
            p.kernel_w = 1;
        }

        p.stride_t = 1;
        p.stride_h = 1;
        p.stride_w = 1;
        if (strides.size() == 3) {
            p.stride_t = strides[0];
            p.stride_h = strides[1];
            p.stride_w = strides[2];
        } else if (strides.size() == 2) {
            p.stride_h = strides[0];
            p.stride_w = strides[1];
        } else if (strides.size() == 1) {
            p.stride_h = strides[0];
        }

        p.pad_before = 0;
        p.pad_top = 0;
        p.pad_left = 0;
        p.pad_after = 0;
        p.pad_bottom = 0;
        p.pad_right = 0;
        if (pads.size() == 6) {
            p.pad_before = pads[0];
            p.pad_top = pads[1];
            p.pad_left = pads[2];
            p.pad_after = pads[3];
            p.pad_bottom = pads[4];
            p.pad_right = pads[5];
        } else if (pads.size() == 4) {
            p.pad_top = pads[0];
            p.pad_left = pads[1];
            p.pad_bottom = pads[2];
            p.pad_right = pads[3];
        } else if (pads.size() == 2) {
            p.pad_top = pads[0];
            p.pad_bottom = pads[1];
        } else if (autoPad == "SAME_UPPER") {
            p.pad_top = (p.kernel_h - 1) / 2;
            p.pad_bottom = (p.kernel_h - 1) - p.pad_top;
            p.pad_left = (p.kernel_w - 1) / 2;
            p.pad_right = (p.kernel_w - 1) - p.pad_left;
        }
        ps.pooling_spec = p;
        return ps;
    }

    ParameterSpec adapt_UnPooling() override
    {
        ParameterSpec ps;
        PoolingParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> kernels = get_ints(this->onnxNode, "kernel_shape");
        std::vector<int> strides = get_ints(this->onnxNode, "strides");
        std::vector<int> pads = get_ints(this->onnxNode, "pads");

        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "MaxUnpool") {
            p.mode = POOLING_MAX;
        } else {
            p.mode = POOLING_MEAN;
        }

        p.kernel_t = 0;
        p.kernel_h = 0;
        p.kernel_w = 0;
        if (kernels.size() == 3) {
            p.kernel_t = kernels[0];
            p.kernel_h = kernels[1];
            p.kernel_w = kernels[2];
        } else if (kernels.size() == 2) {
            p.kernel_t = 1;
            p.kernel_h = kernels[0];
            p.kernel_w = kernels[1];
        } else if (kernels.size() == 1) {
            p.kernel_t = 1;
            p.kernel_h = kernels[0];
            p.kernel_w = 1;
        }

        p.stride_t = 1;
        p.stride_h = 1;
        p.stride_w = 1;
        if (strides.size() == 3) {
            p.stride_t = strides[0];
            p.stride_h = strides[1];
            p.stride_w = strides[2];
        } else if (strides.size() == 2) {
            p.stride_h = strides[0];
            p.stride_w = strides[1];
        } else if (strides.size() == 1) {
            p.stride_h = strides[0];
        }

        p.pad_before = 0;
        p.pad_top = 0;
        p.pad_left = 0;
        p.pad_after = 0;
        p.pad_bottom = 0;
        p.pad_right = 0;
        if (pads.size() == 6) {
            p.pad_before = pads[0];
            p.pad_top = pads[1];
            p.pad_left = pads[2];
            p.pad_after = pads[3];
            p.pad_bottom = pads[4];
            p.pad_right = pads[5];
        } else if (pads.size() == 4) {
            p.pad_top = pads[0];
            p.pad_left = pads[1];
            p.pad_bottom = pads[2];
            p.pad_right = pads[3];
        } else if (pads.size() == 2) {
            p.pad_top = pads[0];
            p.pad_bottom = pads[1];
        }
        ps.pooling_spec = p;
        return ps;
    }

    ParameterSpec adapt_MatMul() override
    {
        ParameterSpec ps;
        MatMulParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.transpose_a = false;
        p.transpose_b = false;
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Gemm") {
            p.transpose_a = get_int(this->onnxNode, "transA", 0);
            p.transpose_b = get_int(this->onnxNode, "transB", 0);
        }
        ps.matmul_spec = p;
        add_shared_weight(this->onnxNode);
        return ps;
    }

    ParameterSpec adapt_Fc() override
    {
        ParameterSpec ps;
        FullyConnectedParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.num_outputs = -1;

        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "MatMul") {
            const onnx::TensorProto &matmulTp = onnxWeights[this->onnxNode.input(1)];
            if (matmulTp.dims_size() == 2) {
                p.num_outputs = matmulTp.dims(1);
            } else {
                UNI_ERROR_LOG("can not map operator name:%s type:%s to FullyConnected.\n",
                    this->onnxNode.name().c_str(), onnxNodeType.c_str());
            }
        } else if (onnxNodeType == "Linear") {
            const onnx::TensorProto &matmulTp = onnxWeights[this->onnxNode.input(1)];
            if (matmulTp.dims_size() == 2) {
                p.num_outputs = matmulTp.dims(0);
            } else {
                UNI_ERROR_LOG("can not map operator name:%s type:%s to FullyConnected.\n",
                    this->onnxNode.name().c_str(), onnxNodeType.c_str());
            }
        } else {
            float alpha = get_float(this->onnxNode, "alpha", 1.f);
            float beta = get_float(this->onnxNode, "beta", 1.f);
            int transA = get_int(this->onnxNode, "transA", 0);
            int transB = get_int(this->onnxNode, "transB", 0);
            auto weightTp = onnxWeights[this->onnxNode.input(1)];
            if (transB == 1.0) {
                p.num_outputs = weightTp.dims(0);
            } else {
                p.num_outputs = weightTp.dims(1);
            }
            if (!(alpha == 1.f && beta == 1.f && transA == 0)) {
                UNI_ERROR_LOG("can not map operator name:%s type:%s to FullyConnected.\n",
                    this->onnxNode.name().c_str(), onnxNodeType.c_str());
            }
        }
        p.num_slices = 1;
        p.slice_point[0] = p.num_outputs;
        ps.fc_spec = p;
        return ps;
    }

    ParameterSpec adapt_BatchNorm() override
    {
        ParameterSpec ps;
        BatchNormParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.eps = get_float(this->onnxNode, "epsilon", 1e-5f);
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "BatchNormalization") {
            p.axis = 1;
        } else if (onnxNodeType == "BatchNorm") {
            p.axis = -1;
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to BatchNorm.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        p.gama = 1;
        p.momentum = get_float(this->onnxNode, "momentum", 0.9);
        ps.bn_spec = p;
        return ps;
    }

    ParameterSpec adapt_InstanceNorm() override
    {
        ParameterSpec ps;
        InstanceNormParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.eps = get_float(this->onnxNode, "epsilon", 1e-5f);
        p.axis = 1;
        p.axis_dim = -1;
        ps.in_spec = p;
        return ps;
    }

    ParameterSpec adapt_Eltwise() override
    {
        ParameterSpec ps;
        EltwiseParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Add" || onnxNodeType == "Sum") {
            p.mode = ELTWISE_SUM;
            p.sum_spec.num_coeff = 2;
            for (I32 j = 0; j < p.sum_spec.num_coeff; j++) {
                p.sum_spec.coeff[j] = 1.0;
            }
        } else if (onnxNodeType == "Mul") {
            p.mode = ELTWISE_PROD;
        } else if (onnxNodeType == "Sub") {
            p.mode = ELTWISE_SUB;
        } else if (onnxNodeType == "Div") {
            p.mode = ELTWISE_DIV;
        } else if (onnxNodeType == "And") {
            p.mode = ELTWISE_AND;
        } else if (onnxNodeType == "Or") {
            p.mode = ELTWISE_OR;
        } else if (onnxNodeType == "Xor") {
            p.mode = ELTWISE_XOR;
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Eltwise.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        p.activation_type = ACTIVATION_NULL;
        ps.eltwise_spec = p;
        add_shared_weight(this->onnxNode);
        return ps;
    }

    ParameterSpec adapt_Einsum() override
    {
        ParameterSpec ps;
        EinsumParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::string equ = get_string(this->onnxNode, "equation");
        std::size_t s0 = equ.find(",");
        std::size_t s1 = equ.find("->");
        unsigned l_idx = s0;
        unsigned r_idx = s1;
        if (s0 == std::string::npos) {
            p.num_equation_l = 0;
            l_idx = s1;
        }

        if (s1 == std::string::npos) {
            p.num_equation_o = 0;
            if (s0 == std::string::npos) {
                l_idx = equ.size();
            }
        } else {
            p.num_equation_o = equ.size() - s1 - 2;
            std::string equ_o = equ.substr(s1 + 2, p.num_equation_o);
            std::reverse(equ_o.begin(), equ_o.end());
            UNI_STRCPY(p.equation_o, equ_o.c_str());
        }

        std::string equ_l = equ.substr(0, l_idx);
        std::reverse(equ_l.begin(), equ_l.end());
        UNI_STRCPY(p.equation_l, equ_l.c_str());
        p.num_equation_l = l_idx;

        if (l_idx < r_idx) {
            std::string equ_r = equ.substr(l_idx + 1, r_idx - l_idx - 1);
            std::reverse(equ_r.begin(), equ_r.end());
            UNI_STRCPY(p.equation_r, equ_r.c_str());
            p.num_equation_r = r_idx - l_idx - 1;
        }
        ps.einsum_spec = p;
        add_shared_weight(this->onnxNode);
        return ps;
    }

    void handle_Constant()
    {
        const auto &output_name = this->onnxNode.output(0);
        for (int i = 0; i < this->onnxNode.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = this->onnxNode.attribute(i);
            if (attribute.name() == "value") {
                CHECK_REQUIREMENT(4 == attribute.type());
                const onnx::TensorProto &tp = attribute.t();
                this->onnxWeights[output_name] = tp;
                this->onnxWeightReferCount[output_name] = -INT_MAX;
                break;
            }
        }
        for (int j = 0; j < onnxGraph.output().size(); j++) {
            if (output_name == onnxGraph.output(j).name()) {
                this->sharedWeights.insert(output_name);
                break;
            }
        }
    }

    ParameterSpec adapt_Pad() override
    {
        ParameterSpec ps;
        PadParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::string padModeStr = get_string(this->onnxNode, "mode");
        std::vector<int> padVec = get_ints(this->onnxNode, "pads");
        F32 padValue = get_float(this->onnxNode, "value", 0.f);
        if (padModeStr == "constant" || padModeStr.length() == 0) {
            p.pad_mode = PAD_CONSTANT;
        } else if (padModeStr == "edge") {
            p.pad_mode = PAD_EDGE;
        } else if (padModeStr == "reflect") {
            p.pad_mode = PAD_REFLECT;
        }

        U32 padSize = padVec.size();
        if (padSize == 0) {
            padVec = get_ints(get_weight(this->onnxNode.input(1)));
            padSize = padVec.size();
        }
        if (padSize == 0) {
            padVec = std::vector<int>(8, UNI_RESERVE);
            padSize = padVec.size();
        }
        if (padSize == 8) {  // NCHW
            p.front = padVec[1];
            p.top = padVec[2];
            p.left = padVec[3];
            p.back = padVec[5];
            p.bottom = padVec[6];
            p.right = padVec[7];
        } else if (padSize == 6) {  // NCH
            p.front = padVec[1];
            p.top = padVec[2];
            p.back = padVec[4];
            p.bottom = padVec[5];
        } else if (padSize == 4) {  // HW
            p.front = padVec[1];
            p.back = padVec[3];
        } else {
            UNI_ERROR_LOG("can not process operator name:%s type:%s attributes.\n",
                this->onnxNode.name().c_str(), this->onnxNode.op_type().c_str());
        }
        p.constant_value = padValue;
        ps.pad_spec = p;
        add_shared_weight(this->onnxNode, {0});
        return ps;
    }

    ParameterSpec adapt_Gather() override
    {
        ParameterSpec ps;
        GatherParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Gather" || onnxNodeType == "GatherElements") {
            p.axis = get_int(this->onnxNode, "axis", 0);
        } else {
            p.axis = INT_MAX;
        }
        if (onnxNodeType == "GatherElements") {
            p.element_level = true;
        } else {
            p.element_level = false;
        }
        if (onnxNodeType == "GatherND") {
            p.batch_dims = get_int(this->onnxNode, "batch_dims", 0);
        } else {
            p.batch_dims = 0;
        }
        ps.gather_spec = p;
        add_shared_weight(this->onnxNode);
        return ps;
    }

    ParameterSpec adapt_TfSlice() override
    {
        ParameterSpec ps;
        TfSliceParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.num_dims = 8;
        std::vector<int> starts(p.num_dims, UNI_RESERVE), ends(p.num_dims, UNI_RESERVE), axes, steps;
        if (this->onnxNode.input_size() == 1) {
            starts = get_ints(this->onnxNode, "starts");
            ends = get_ints(this->onnxNode, "ends");
            axes = get_ints(this->onnxNode, "axes");
            steps = get_ints(this->onnxNode, "steps");
        } else {
            starts = get_ints(get_weight(this->onnxNode.input(1)), starts);
            ends = get_ints(get_weight(this->onnxNode.input(2)), ends);
            if (this->onnxNode.input_size() >= 4) {
                axes = get_ints(get_weight(this->onnxNode.input(3)));
            }
            if (this->onnxNode.input_size() >= 5) {
                steps = get_ints(get_weight(this->onnxNode.input(4)));
            }
        }
        for (U32 i = 0; i < p.num_dims; i++) {
            p.begin[i] = 0;
            p.end[i] = -1;
            p.strides[i] = 1;
            p.begin_mask[i] = 1;
            p.end_mask[i] = 1;
        }
        U32 num = UNI_MIN(starts.size(), ends.size());
        if (axes.size() > 0) {
            num = UNI_MIN(axes.size(), num);
        }
        TensorDesc desc = tensor0d();
        if (onnxValues.find(this->onnxNode.input(0)) != onnxValues.end()) {
            desc = get_desc(onnxValues[this->onnxNode.input(0)]);
        }
        for (U32 i = 0; i < num; i++) {
            int axis;
            if (axes.size() > 0) {
                axis = axes[i];
            } else {
                axis = i;
            }
            if (axis < 0) {
                if (desc.nDims > 0) {
                    axis += desc.nDims;
                } else {
                    UNI_ERROR_LOG("can not process %s(%s)'s negative axis(%d), you can change it "
                                  "to positive and reconvert model.\n",
                        this->onnxNode.op_type().c_str(), this->onnxNode.name().c_str(), axis);
                }
            }
            p.begin[axis] = starts[i];
            p.end[axis] = ends[i];
            if (steps.size() > 0) {
                p.strides[axis] = steps[i];
            }
            p.begin_mask[axis] = 0;
            p.end_mask[axis] = 0;
        }
        ps.tfslice_spec = p;
        add_shared_weight(this->onnxNode, {0});
        return ps;
    }

    ParameterSpec adapt_Slice() override
    {
        const std::string &onnxNodeType = this->onnxNode.op_type();
        ParameterSpec ps;
        SliceParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> split = get_ints(this->onnxNode, "split");
        if (split.size() == 0 && onnxNodeType == "Split" && this->onnxNode.input_size() > 1) {
            split = get_ints(get_weight(this->onnxNode.input(1)));
        }
        if (split.size() > sizeof(p.slice_points) / sizeof(p.slice_points[0])) {
            UNI_ERROR_LOG("Slice parameter size(%d) is smaller than need(%d). Please modify "
                          "parameter defination and rebuild bolt.\n",
                (int)(sizeof(p.slice_points) / sizeof(p.slice_points[0])), (int)split.size());
        }
        p.axis = get_int(this->onnxNode, "axis", 0);
        // Split equally by default. Set all slice_points to 0
        if (0 == split.size()) {
            p.num_slice = this->onnxNode.output_size() - 1;
            UNI_MEMSET(p.slice_points, 0, p.num_slice * sizeof(I32));
        } else {
            p.num_slice = split.size() - 1;
            p.slice_points[0] = split[0];
            for (U32 i = 1; i < p.num_slice; i++) {
                p.slice_points[i] = p.slice_points[i - 1] + split[i];
            }
        }
        ps.slice_spec = p;
        return ps;
    }

    ParameterSpec adapt_Squeeze() override
    {
        ParameterSpec ps;
        SqueezeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> squeezeAxes;
        if (this->onnxNode.input_size() > 1) {
            squeezeAxes = get_ints(get_weight(this->onnxNode.input(1)));
        } else {
            squeezeAxes = get_ints(this->onnxNode, "axes");
        }
        p.num_axes = squeezeAxes.size();
        for (int squeeze_i = 0; squeeze_i < (int)squeezeAxes.size(); squeeze_i++) {
            p.axes[squeeze_i] = squeezeAxes[squeeze_i];
        }
        ps.squeeze_spec = p;
        return ps;
    }

    ParameterSpec adapt_Unsqueeze() override
    {
        ParameterSpec ps;
        UnsqueezeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> unsqueezeAxes;
        if (this->onnxNode.input_size() > 1) {
            unsqueezeAxes = get_ints(get_weight(this->onnxNode.input(1)));
        } else {
            unsqueezeAxes = get_ints(this->onnxNode, "axes");
        }
        p.num_axes = unsqueezeAxes.size();
        for (int unsqueeze_i = 0; unsqueeze_i < (int)unsqueezeAxes.size(); unsqueeze_i++) {
            p.axes[unsqueeze_i] = unsqueezeAxes[unsqueeze_i];
        }
        ps.unsqueeze_spec = p;
        return ps;
    }

    ParameterSpec adapt_Cast() override
    {
        ParameterSpec ps;
        CastParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        int dst;
        if (this->onnxNode.input_size() == 2 &&
            onnxWeights.find(this->onnxNode.input(1)) != onnxWeights.end()) {
            dst = (get_ints(onnxWeights[this->onnxNode.input(1)]))[0];
        } else {
            dst = get_int(this->onnxNode, "to", 0);
        }
        p.dt = cut_type(get_type((onnx::TensorProto::DataType)dst));
        if (p.dt == DT_F16) {
            p.dt = DT_F32;
        }
        ps.cast_spec = p;
        return ps;
    }

    ParameterSpec adapt_Concat() override
    {
        ParameterSpec ps;
        ConcatParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = get_int(this->onnxNode, "axis", 1);
        ps.concat_spec = p;
        add_shared_weight(this->onnxNode);
        return ps;
    }

    ParameterSpec adapt_Softmax() override
    {
        ParameterSpec ps;
        SoftmaxParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = get_int(this->onnxNode, "axis", -1);
        ps.softmax_spec = p;
        return ps;
    }

    ParameterSpec adapt_Relu() override
    {
        ParameterSpec ps;
        ReLUParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.neg_slope = get_float(this->onnxNode, "alpha", 0.0);
        ps.relu_spec = p;
        return ps;
    }

    ParameterSpec adapt_RNN() override
    {
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Scan") {
            return adapt_Scan();
        }
        ParameterSpec ps;
        RNNParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        if (onnxNodeType == "RNN") {
            p.mode = RNN_RNN;
        } else if (onnxNodeType == "LSTM") {
            p.mode = RNN_LSTM;
        } else if (onnxNodeType == "GRU") {
            int linear_before_reset = get_int(this->onnxNode, "linear_before_reset", 0);
            if (linear_before_reset == 0) {
                p.mode = RNN_GRU;
            } else {
                p.mode = RNN_GRU_LBR;
            }
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to RNN.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        p.num_outputs = get_int(this->onnxNode, "hidden_size", 1);
        p.bi_direction =
            get_string(this->onnxNode, "direction", "forward") == "bidirectional" ? true : false;
        p.steps = 0;
        p.num_projection = 0;
        p.zoneout_cell = 0;
        p.zoneout_output = 0;
        p.forget_bias = 0;
        p.activation_type = ACTIVATION_TANH;
        ps.rnn_spec = p;
        return ps;
    }

    // (scale * x + shift) ^ power
    ParameterSpec adapt_Power() override
    {
        ParameterSpec ps;
        PowerParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.scale = 1;
        p.shift = 0;
        p.power = 1;
        int index = 0;
        float value = 0;
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Pow" || onnxNodeType == "Mul" || onnxNodeType == "Div" ||
            onnxNodeType == "Add" || onnxNodeType == "Sub") {
            std::vector<int> ids = get_weight_ids(this->onnxNode);
            CHECK_REQUIREMENT(ids.size() == 1);
            index = ids[0];
            value = get_floats(onnxWeights[this->onnxNode.input(index)])[0];
        }
        if (onnxNodeType == "Pow") {
            p.power = value;
        } else if (onnxNodeType == "Mul") {
            p.scale = value;
        } else if (onnxNodeType == "Div") {
            p.scale = 1 / value;
            if (index == 0) {
                p.power = -1;
            }
        } else if (onnxNodeType == "Add") {
            p.shift = value;
        } else if (onnxNodeType == "Sub") {
            if (index == 0) {
                p.scale = -1;
                p.shift = value;
            } else {
                p.shift = -1 * value;
            }
        } else if (onnxNodeType == "Sqrt") {
            p.power = 0.5;
        } else if (onnxNodeType == "Scale") {
            p.scale = get_float(this->onnxNode, "scale", 1.0);
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Power.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        ps.power_spec = p;
        return ps;
    }

    ParameterSpec adapt_Scale() override
    {
        ParameterSpec ps;
        ScaleParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Add" || onnxNodeType == "Sub" || onnxNodeType == "Mul" ||
            onnxNodeType == "Div") {
            std::vector<int> ids = get_weight_ids(this->onnxNode);
            const auto &tensor = onnxWeights[this->onnxNode.input(ids[0])];
            if (tensor.dims_size() > 1) {
                for (int idx = 0; idx < tensor.dims_size(); ++idx) {
                    if (tensor.dims(idx) > 1) {
                        p.axis = idx - tensor.dims_size();
                        break;
                    }
                }
            } else {
                p.axis = -1;
            }
        } else {
            p.axis = 1;
        }
        ps.scale_spec = p;
        return ps;
    }

    ParameterSpec adapt_Space2Depth() override
    {
        ParameterSpec ps;
        Space2DepthParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.block_size = get_int(this->onnxNode, "blocksize", 1);
        ps.space2depth_spec = p;
        return ps;
    }

    ParameterSpec adapt_Depth2Space() override
    {
        ParameterSpec ps;
        Depth2SpaceParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.block_size = get_int(this->onnxNode, "blocksize", 1);
        std::string mode = get_string(this->onnxNode, "mode", "DCR");
        str_copy(p.mode, mode.c_str(), mode.length(), 8);
        ps.depth2space_spec = p;
        return ps;
    }

    ParameterSpec adapt_Reduction() override
    {
        ParameterSpec ps;
        ReductionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> axes = get_ints(this->onnxNode, "axes");
        if (axes.size() == 0 && this->onnxNode.input_size() > 1) {
            axes = get_ints(get_weight(this->onnxNode.input(1)));
        }
        int keepdims = get_int(this->onnxNode, "keepdims", 1);
        p.num_axes = axes.size();
        for (int i = 0; i < p.num_axes; i++) {
            p.axes[i] = axes[i];
        }
        p.keep_dim = keepdims == 0 ? false : true;
        p.coeff = 1.0;
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "ReduceSum") {
            p.mode = REDUCTION_SUM;
        } else if (onnxNodeType == "ReduceMean") {
            p.mode = REDUCTION_MEAN;
        } else if (onnxNodeType == "ReduceMax") {
            p.mode = REDUCTION_MAX;
        } else if (onnxNodeType == "ReduceMin") {
            p.mode = REDUCTION_MIN;
        } else if (onnxNodeType == "ReduceL2") {
            p.mode = REDUCTION_L2;
        } else if (onnxNodeType == "ReduceProd") {
            p.mode = REDUCTION_SCALAR_PRODUCT;
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Reduction.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        ps.reduction_spec = p;
        return ps;
    }

    ParameterSpec adapt_ArgMax() override
    {
        ParameterSpec ps;
        ArgMaxParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = get_int(this->onnxNode, "axis", -1);
        ps.argmax_spec = p;
        return ps;
    }

    ParameterSpec adapt_PRelu() override
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        return ps;
    }

    ParameterSpec adapt_Tile() override
    {
        ParameterSpec ps;
        TileParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> repeats = get_ints(get_weight(this->onnxNode.input(1)));
        p.num_repeats = repeats.size();
        UNI_MEMCPY(p.repeats, repeats.data(), p.num_repeats * sizeof(int));
        ps.tile_spec = p;
        add_shared_weight(this->onnxNode, {0});
        return ps;
    }

    ParameterSpec adapt_Splice() override
    {
        ParameterSpec ps;
        SpliceParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> context = get_ints(this->onnxNode, "context");
        std::vector<int> ids = get_ints(this->onnxNode, "forward_indexes");
        p.num_context = context.size();
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (p.num_context == 0) {
            UNI_ERROR_LOG("can not process operator name:%s type:%s attributes.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        UNI_MEMCPY(p.context, context.data(), p.num_context * sizeof(int));
        p.index_min = 0;
        p.index_max = 0;
        for (U32 i = 0; i < ids.size(); i++) {
            p.index_min = UNI_MIN(p.index_min, ids[i]);
            p.index_max = UNI_MAX(p.index_max, ids[i]);
        }
        ps.splice_spec = p;
        return ps;
    }

    ParameterSpec adapt_Tdnn() override
    {
        ParameterSpec ps;
        TdnnParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const onnx::TensorProto &context = get_weight(this->onnxNode.input(1));
        const onnx::TensorProto &params = get_weight(this->onnxNode.input(2));
        p.num_context = get_length(context);
        UNI_MEMCPY(p.context, get_ints(context).data(), p.num_context * sizeof(int));
        p.num_outputs = params.dims(0);
        p.activation_type = ACTIVATION_NULL;
        ps.tdnn_spec = p;
        return ps;
    }

    ParameterSpec adapt_TopK() override
    {
        ParameterSpec ps;
        TopKParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = get_int(this->onnxNode, "axis", -1);
        p.largest = get_int(this->onnxNode, "largest", 1);
        p.sorted = get_int(this->onnxNode, "sorted", 1);
        if (this->onnxNode.input_size() == 1) {
            p.k = get_int(this->onnxNode, "k", 1);
        } else {
            if (onnxWeights.find(this->onnxNode.input(1)) != onnxWeights.end()) {
                p.k = get_ints(onnxWeights[this->onnxNode.input(1)])[0];
            } else {
                p.k = 0;
            }
        }
        ps.topk_spec = p;
        return ps;
    }

    ParameterSpec adapt_Where() override
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        add_shared_weight(this->onnxNode);
        return ps;
    }

    ParameterSpec adapt_Scan()
    {
        ParameterSpec ps;
        RNNParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        onnx::GraphProto subGraph;
        for (int i = 0; i < this->onnxNode.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = this->onnxNode.attribute(i);
            if (attribute.name() == "body") {
                subGraph = attribute.g();
                break;
            }
        }

        std::vector<onnx::TensorProto> gemmTps;
        std::vector<onnx::TensorProto> matmulTps;
        for (int i = 0; i < subGraph.node_size(); i++) {
            auto node = subGraph.node(i);
            int input_size = (int)node.input_size();
            bool stopTag = false;
            if (node.op_type() == "MatMul") {
                for (int j = 0; j < input_size; j++) {
                    if (onnxWeights.find(node.input(j)) != onnxWeights.end()) {
                        auto hidWeightTp = onnxWeights[node.input(j)];
                        int hidWeightSize = get_length(hidWeightTp);
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

        p.mode = RNN_LSTM;
        p.num_outputs = matmulTps[0].dims(1);
        p.steps = 0;
        p.num_projection = matmulTps[0].dims(0);
        p.zoneout_cell = 0;
        p.zoneout_output = 0;
        p.forget_bias = 1.0;
        p.activation_type = ACTIVATION_TANH;
        ps.rnn_spec = p;
        return ps;
    }

    ParameterSpec adapt_Expand() override
    {
        ParameterSpec ps;
        ExpandParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        if (onnxWeights.find(this->onnxNode.input(0)) != onnxWeights.end()) {
            add_shared_weight(this->onnxNode, {0});
        }
        std::vector<int> shape;
        if (this->onnxNode.input_size() == 1) {
            shape = get_ints(this->onnxNode, "shape");
        } else {
            shape = get_ints(get_weight(this->onnxNode.input(1)));
        }
        p.num_shape = shape.size();
        UNI_MEMCPY(p.shape, shape.data(), p.num_shape * sizeof(I32));
        ps.expand_spec = p;
        return ps;
    }

    ParameterSpec adapt_Scatter() override
    {
        ParameterSpec ps;
        ScatterParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Scatter" || onnxNodeType == "ScatterElements") {
            p.axis = get_int(this->onnxNode, "axis", 0);
        } else {
            p.axis = INT_MAX;
        }
        ps.scatter_spec = p;
        add_shared_weight(this->onnxNode);
        return ps;
    }

    ParameterSpec adapt_RoIAlign() override
    {
        ParameterSpec ps;
        RoIAlignParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        std::string trans_mode;
        if (onnxNodeType == "RoiAlign") {
            trans_mode = get_string(this->onnxNode, "coordinate_transformation_mode", "half_pixel");
        } else {
            trans_mode = get_string(this->onnxNode, "coordinate_transformation_mode", "NO_SET");
        }
        if (trans_mode == "NO_SET") {
            int aligned = get_int(this->onnxNode, "aligned", 1);
            if (aligned <= 0) {
                trans_mode = "output_half_pixel";
            } else {
                trans_mode = "half_pixel";
            }
        }
        if (trans_mode == "half_pixel") {
            p.trans_mode = COORDINATE_TRANS_HALF_PIXEL;
        } else if (trans_mode == "output_half_pixel") {
            p.trans_mode = COORDINATE_TRANS_OUTPUT_HALF_PIXEL;
        } else {
            UNI_ERROR_LOG("can not support trans_mode:%s in operator name:%s "
                          "type:%s.\n",
                trans_mode.c_str(), this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }

        std::string poolingMode = get_string(this->onnxNode, "mode", "avg");
        if (poolingMode == "avg") {
            p.mode = POOLING_MEAN;
        } else if (poolingMode == "max") {
            p.mode = POOLING_MAX;
        } else {
            UNI_ERROR_LOG("can not support mode:%s in operator name:%s type:%s.\n",
                poolingMode.c_str(), this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        p.output_w = get_int(this->onnxNode, "pooled_w", 1);
        if (p.output_w == 1) {
            p.output_w = get_int(this->onnxNode, "output_width", 1);
        }
        p.output_h = get_int(this->onnxNode, "pooled_h", 1);
        if (p.output_h == 1) {
            p.output_h = get_int(this->onnxNode, "output_height", 1);
        }
        p.sampling_ratio = get_int(this->onnxNode, "sampling_ratio", 0);
        p.spatial_scale = get_float(this->onnxNode, "spatial_scale", 1.0);
        ps.roialign_spec = p;
        return ps;
    }

    ParameterSpec adapt_GenerateProposals() override
    {
        ParameterSpec ps;
        GenerateProposalsParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.angle_bound_hi = get_int(this->onnxNode, "angle_bound_hi", 0);
        p.angle_bound_lo = get_int(this->onnxNode, "angle_bound_lo", 0);
        p.angle_bound_on = get_int(this->onnxNode, "angle_bound_on", 0);
        p.clip_angle_thresh = get_float(this->onnxNode, "clip_angle_thresh", 0.0);
        p.legacy_plus_one = get_int(this->onnxNode, "legacy_plus_one", 0);
        p.min_size = get_float(this->onnxNode, "min_size", 0.0);
        p.nms_thresh = get_float(this->onnxNode, "nms_thresh", 0.0);
        p.post_nms_topN = get_int(this->onnxNode, "post_nms_topN", 0);
        p.pre_nms_topN = get_int(this->onnxNode, "pre_nms_topN", 0);
        p.spatial_scale = get_float(this->onnxNode, "spatial_scale", 0.0);
        ps.generate_proposals_spec = p;
        return ps;
    }

    ParameterSpec adapt_GridSample() override
    {
        ParameterSpec ps;
        GridSampleParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        std::string mode = get_string(this->onnxNode, "mode", "bilinear");
        if (mode.compare("bilinear") == 0) {
            p.mode = RESIZE_LINEAR;
        } else if (mode.compare("nearest") == 0) {
            p.mode = RESIZE_NEAREST;
        } else {
            UNI_ERROR_LOG("can not support mode:%s in operator name:%s type:%s.\n", mode.c_str(),
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        p.constant_value = 0;
        std::string pad_mode = get_string(this->onnxNode, "padding_mode", "zeros");
        if (pad_mode == "zeros") {
            p.pad_mode = PAD_CONSTANT;
        } else if (pad_mode == "border") {
            p.pad_mode = PAD_EDGE;
        } else if (pad_mode == "reflection") {
            p.pad_mode = PAD_REFLECT;
        } else {
            UNI_ERROR_LOG("can not support pad_mode:%s in operator name:%s type:%s.\n",
                pad_mode.c_str(), this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        p.align_corners = get_int(this->onnxNode, "align_corners", 1);
        ps.grid_sample_spec = p;
        return ps;
    }

    ParameterSpec adapt_OneHot() override
    {
        ParameterSpec ps;
        OneHotParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = get_int(this->onnxNode, "axis", -1);
        p.depth = get_ints(get_weight(this->onnxNode.input(1)))[0];
        std::vector<float> values = get_floats(get_weight(this->onnxNode.input(2)));
        UNI_MEMCPY(p.values, values.data(), sizeof(float) * values.size());
        ps.onehot_spec = p;
        return ps;
    }

    ParameterSpec adapt_Cum() override
    {
        ParameterSpec ps;
        CumParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "CumSum") {
            p.mode = ELTWISE_SUM;
        } else {
            p.mode = ELTWISE_PROD;
        }
        p.exclusive = get_int(this->onnxNode, "exclusive", 0);
        p.reverse = get_int(this->onnxNode, "reverse", 0);
        p.axis = get_ints(get_weight(this->onnxNode.input(1)))[0];
        ps.cum_spec = p;
        return ps;
    }

    ParameterSpec adapt_NonMaxSuppression() override
    {
        ParameterSpec ps;
        NonMaxSuppressionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.center_point_box = get_int(this->onnxNode, "center_point_box", 0);
        if (this->onnxNode.input_size() > 2) {
            p.max_output_boxes_per_class = get_ints(get_weight(this->onnxNode.input(2)))[0];
        }
        if (this->onnxNode.input_size() > 3) {
            p.iou_threshold = get_floats(get_weight(this->onnxNode.input(3)))[0];
        }
        if (this->onnxNode.input_size() > 4) {
            p.score_threshold = get_floats(get_weight(this->onnxNode.input(4)))[0];
        }
        ps.non_max_suppression_spec = p;
        return ps;
    }

    ParameterSpec adapt_Check() override
    {
        ParameterSpec ps;
        CheckParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Less") {
            p.mode = CHECK_LESS;
        } else if (onnxNodeType == "LessOrEqual") {
            p.mode = CHECK_LESS_EQUAL;
        } else if (onnxNodeType == "Equal") {
            p.mode = CHECK_EQUAL;
        } else if (onnxNodeType == "Greater") {
            p.mode = CHECK_GREATER;
        } else if (onnxNodeType == "GreaterOrEqual") {
            p.mode = CHECK_GREATER_EQUAL;
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Check.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        ps.check_spec = p;
        add_shared_weight(this->onnxNode);
        return ps;
    }

    ParameterSpec adapt_ConstantOfShape() override
    {
        ParameterSpec ps;
        ConstantOfShapeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.dt = cut_type(get_type(this->onnxNode, "value"));
        p.value = get_float(this->onnxNode, "value", 0);
        ps.constant_of_shape_spec = p;
        add_shared_weight(this->onnxNode);
        return ps;
    }

    ParameterSpec adapt_Range() override
    {
        ParameterSpec ps;
        RangeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Loop") {
            p.dt = DT_I32;
            p.start = get_ints(get_weight(this->onnxNode.input(2)))[0];
            p.limit = get_ints(get_weight(this->onnxNode.input(0)))[0];
            p.delta = 1;
        } else if (onnxNodeType == "Range") {
            p.dt = DT_I32;
            std::vector<float> v(1, (float)UNI_RESERVE);
            const auto &start = get_weight(this->onnxNode.input(0));
            const auto &limit = get_weight(this->onnxNode.input(1));
            const auto &delta = get_weight(this->onnxNode.input(2));
            if (get_length(start) > 0) {
                p.dt = cut_type(get_type(start));
            }
            if (get_length(limit) > 0) {
                p.dt = cut_type(get_type(limit));
            }
            if (get_length(delta) > 0) {
                p.dt = cut_type(get_type(delta));
            }
            p.start = get_floats(start, v)[0];
            p.limit = get_floats(limit, v)[0];
            p.delta = get_floats(delta, v)[0];
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Range.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        ps.range_spec = p;
        return ps;
    }

    ParameterSpec adapt_Random() override
    {
        ParameterSpec ps;
        RandomParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        p.dt = cut_type(get_type(this->onnxNode, "dtype"));
        if (onnxNodeType == "RandomNormal" || onnxNodeType == "RandomNormalLike") {
            p.mode = RANDOM_NORMAL;
            p.value[0] = get_float(this->onnxNode, "mean", 0);
            p.value[1] = get_float(this->onnxNode, "scale", 1);
        } else {
            p.mode = RANDOM_UNIFORM;
            p.value[0] = get_float(this->onnxNode, "high", 1);
            p.value[1] = get_float(this->onnxNode, "low", 0);
        }
        if (onnxNodeType == "RandomNormal" || onnxNodeType == "RandomUniform") {
            std::vector<int> shape = get_ints(this->onnxNode, "shape");
            p.num_shape = shape.size();
            UNI_MEMCPY(p.shape, shape.data(), p.num_shape * sizeof(I32));
        }
        p.seed = get_float(this->onnxNode, "seed", UNI_RESERVE);
        ps.random_spec = p;
        return ps;
    }

    ParameterSpec adapt_BilateralSliceApply() override
    {
        ParameterSpec ps;
        BilateralSliceApplyParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::string mode = get_string(this->onnxNode, "mode");
        if (mode == "null") {
            p.mode = BILATERAL_SLICE_APPLY_NULL;
            if (this->onnxNode.input_size() != 3) {
                UNI_ERROR_LOG("BilateralSliceApply need 3 inputs(input, grid, guide) under mode == "
                              "'null'. If you want to integrate guide calculation into big "
                              "operator, you can use 'conv' mode and that need 2 inputs.\n");
            } else {
                UNI_WARNING_LOG("We provide BilateralSliceApply big operator by using 'conv' mode, "
                                "It's relatively faster than 'null' mode.\n");
            }
        } else {
            if (this->onnxNode.input_size() != 2) {
                UNI_ERROR_LOG("BilateralSliceApply only need 2 inputs(input, grid) under mode == "
                              "'conv'. If you don't want to integrate guide calculation into big "
                              "operator, you can use 'null' mode and that need 3 inputs.\n");
            } else {
                UNI_WARNING_LOG("BilateralSliceApply will use inner guide calculation function. If "
                                "you want to change implementation, you can modify "
                                "compute/image/src/gpu/mali/cl/bilateral_slice_apply_c12.cl for "
                                "GPU, or compute/image/src/cpu/bilateral_slice_apply.cpp for "
                                "CPU.\n");
            }
            p.mode = BILATERAL_SLICE_APPLY_CONV;
        }
        ps.bilateral_slice_apply_spec = p;
        return ps;
    }

    ParameterSpec adapt_ConvertColor() override
    {
        ParameterSpec ps;
        ConvertColorParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.src = get_color(get_string(this->onnxNode, "src"));
        p.dst = get_color(get_string(this->onnxNode, "dst"));
        ps.convert_color_spec = p;
        return ps;
    }

    ParameterSpec adapt_Lut() override
    {
        ParameterSpec ps;
        LutParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.type = get_color(get_string(this->onnxNode, "type", "YUV_NV21"));
        p.mode = get_interp_mode(get_string(this->onnxNode, "mode", "CUBIC"));
        ps.lut_spec = p;
        return ps;
    }

private:
    int removePreprocessOpNum;
    // whether to use 1bit bnn
    bool useBNN;
    // whether to use onnx shared weight
    bool useShare;

    std::string modelName;
    onnx::ModelProto onnxModel;
    onnx::GraphProto onnxGraph;
    onnx::NodeProto onnxNode;
    std::map<std::string, onnx::TensorProto> onnxWeights;
    std::map<std::string, int> onnxWeightReferCount;
    std::map<std::string, onnx::ValueInfoProto> onnxValues;

    std::set<std::string> sharedWeights;

    std::map<std::string, std::string> nameMap;
};
#endif
