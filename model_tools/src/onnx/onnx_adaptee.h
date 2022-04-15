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

#include "model_adaptee.h"

class OnnxAdaptee : public ModelAdaptee {
public:
    OnnxAdaptee(int _removePreprocessOpNum, bool _useBNN)
    {
        this->removePreprocessOpNum = _removePreprocessOpNum;
        this->useBNN = _useBNN;
    }
    ~OnnxAdaptee()
    {}

protected:
    DataType get_weight_data_type(U32 weightLen, F32 *weight)
    {
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
        //CHECK_REQUIREMENT(0 == val0 + val1);
        return DT_BIN11;
    }

    std::vector<int> getOperatorWeightInputIndex(const onnx::NodeProto &weightNode)
    {
        std::vector<int> index;
        for (int i = 0; i < weightNode.input_size(); i++) {
            if (onnxWeights.end() != onnxWeights.find(weightNode.input(i))) {
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

    OperatorType convert_onnx_type(const std::string &onnxNodeType)
    {
        std::vector<int> indexes = getOperatorWeightInputIndex(this->onnxNode);
        if (onnxNodeType == "Conv") {
            return OT_Conv;
        } else if (onnxNodeType == "BatchNormalization" || onnxNodeType == "BatchNorm") {
            return OT_BatchNorm;
        } else if (onnxNodeType == "InstanceNormalization") {
            return OT_InstanceNorm;
        } else if (onnxNodeType == "Sum" || onnxNodeType == "Add" || onnxNodeType == "Mul" ||
            onnxNodeType == "Div" || onnxNodeType == "Sub") {
            if (indexes.size() == 0 || this->onnxNode.input_size() > 2) {
                return OT_Eltwise;
            }
            for (U32 i = 0; i < indexes.size(); i++) {
                onnx::TensorProto &weightTp = onnxWeights[this->onnxNode.input(indexes[i])];
                if (is_multi_dim(weightTp)) {
                    return OT_Eltwise;
                }
            }
            const onnx::TensorProto &weightTp = onnxWeights[this->onnxNode.input(indexes[0])];
            int weightNum = get_data_size_from_tensor_proto(weightTp);
            if (weightNum == 1) {
                return OT_Power;
            } else if ((onnxNodeType == "Div" || onnxNodeType == "Sub") && indexes[0] == 0) {
                return OT_Eltwise;
            } else {
                if (this->onnxWeightReferCount[this->onnxNode.input(indexes[0])] > 1) {
                    return OT_Eltwise;
                } else {
                    return OT_Scale;
                }
            }
        } else if (onnxNodeType == "AveragePool" || onnxNodeType == "MaxPool" ||
            onnxNodeType == "GlobalAveragePool") {
            return OT_Pooling;
        } else if (onnxNodeType == "ReduceMean" || onnxNodeType == "ReduceMax") {
            std::vector<int> axesInfo =
                get_node_vector_ints_attribute_by_name(this->onnxNode, "axes");
            int keepdimsInfo = get_node_single_int_attribute_by_name(this->onnxNode, "keepdims", 0);
            if (axesInfo.size() == 2 && axesInfo[0] == 2 && axesInfo[1] == 3 && keepdimsInfo == 1) {
                return OT_Pooling;
            }
            return OT_Reduction;
        } else if (onnxNodeType == "Relu" || onnxNodeType == "LeakyRelu") {
            return OT_Relu;
        } else if (onnxNodeType == "Softmax") {
            return OT_Softmax;
        } else if (onnxNodeType == "Concat") {
            return OT_Concat;
        } else if (onnxNodeType == "Pad") {
            return OT_Pad;
        } else if (onnxNodeType == "Max" || onnxNodeType == "Min" || onnxNodeType == "Clip") {
            return OT_Clip;
        } else if (onnxNodeType == "Reshape") {
            return OT_Reshape;
        } else if (onnxNodeType == "Squeeze") {
            return OT_Squeeze;
        } else if (onnxNodeType == "Transpose") {
            return OT_Transpose;
        } else if (onnxNodeType == "Gather" || onnxNodeType == "GatherElements" ||
            onnxNodeType == "GatherND") {
            return OT_Gather;
        } else if (onnxNodeType == "Unsqueeze") {
            return OT_Unsqueeze;
        } else if (onnxNodeType == "Resize" || onnxNodeType == "Upsample") {
            return OT_Resize;
        } else if (onnxNodeType == "Cast") {
            return OT_Cast;
        } else if (onnxNodeType == "Constant") {
            return OT_Constant;
        } else if (onnxNodeType == "MatMul" || onnxNodeType == "Gemm" || onnxNodeType == "Linear") {
            if (indexes.size() == 0 || (indexes.size() == 1 && indexes[0] == 2)) {
                return OT_MatMul;
            } else {
                auto weightName = this->onnxNode.input(indexes[0]);
                onnx::TensorProto &weightTp = onnxWeights[weightName];
                if (weightTp.dims_size() == 2 && this->onnxWeightReferCount[weightName] == 1) {
                    return OT_FC;
                } else {
                    return OT_MatMul;
                }
            }
        } else if (onnxNodeType == "Flatten") {
            return OT_Reshape;
        } else if (onnxNodeType == "ConvTranspose") {
            return OT_Deconvolution;
        } else if (onnxNodeType == "Tanh") {
            return OT_TanH;
        } else if (onnxNodeType == "LogSoftmax") {
            return OT_LogSoftmax;
        } else if (onnxNodeType == "Shape") {
            return OT_Shape;
        } else if (onnxNodeType == "Erf") {
            return OT_Erf;
        } else if (onnxNodeType == "Pow" || onnxNodeType == "Sqrt") {
            return OT_Power;
        } else if (onnxNodeType == "RNN" || onnxNodeType == "GRU" || onnxNodeType == "LSTM" ||
            onnxNodeType == "Scan") {
            return OT_RNN;
        } else if (onnxNodeType == "ConstantOfShape") {
            return OT_ConstantOfShape;
        } else if (onnxNodeType == "SpaceToDepth") {
            return OT_Space2Depth;
        } else if (onnxNodeType == "DepthToSpace") {
            return OT_Depth2Space;
        } else if (onnxNodeType == "PRelu") {
            return OT_PRelu;
        } else if (onnxNodeType == "ArgMax") {
            return OT_ArgMax;
        } else if (onnxNodeType == "Tile") {
            return OT_Tile;
        } else if (onnxNodeType == "Sigmoid") {
            return OT_Sigmoid;
        } else if (onnxNodeType == "Slice") {
            return OT_TfSlice;
        } else if (onnxNodeType == "ReduceSum" || onnxNodeType == "ReduceMin" ||
            onnxNodeType == "ReduceL2") {
            return OT_Reduction;
        } else if (onnxNodeType == "Split") {
            return OT_Slice;
        } else if (onnxNodeType == "Splice") {
            return OT_Splice;
        } else if (onnxNodeType == "Greater") {
            return OT_Greater;
        } else if (onnxNodeType == "Where") {
            return OT_Where;
        } else if (onnxNodeType == "SoftPlus") {
            return OT_SoftPlus;
        } else if (onnxNodeType == "Exp") {
            return OT_Exp;
        } else if (onnxNodeType == "NoOp") {
            return OT_Split;
        } else if (onnxNodeType == "Tdnn") {
            return OT_Tdnn;
        } else if (onnxNodeType == "Dropout") {
            return OT_Dropout;
        } else if (onnxNodeType == "Scale") {
            return OT_Power;
        } else if (onnxNodeType == "TopK") {
            return OT_TopK;
        } else if (onnxNodeType == "Equal") {
            return OT_Equal;
        } else if (onnxNodeType == "Sign") {
            return OT_Sign;
        } else if (onnxNodeType == "TFL_HARD_SWISH") {
            return OT_HSwish;
        } else if (onnxNodeType == "Expand") {
            return OT_Expand;
        } else if (onnxNodeType == "ScatterND" || onnxNodeType == "ScatterElements") {
            return OT_Scatter;
        } else if (onnxNodeType == "Not") {
            return OT_Not;
        } else if (onnxNodeType == "Abs") {
            return OT_Abs;
        } else if (onnxNodeType == "Reciprocal") {
            return OT_Reciprocal;
        } else if (onnxNodeType == "And" || onnxNodeType == "Or" || onnxNodeType == "Xor") {
            return OT_Eltwise;
        } else if (onnxNodeType == "Log") {
            return OT_Log;
        } else if (onnxNodeType == "Neg") {
            return OT_Neg;
        } else if (onnxNodeType == "GenerateProposals") {
            return OT_GenerateProposals;
        } else if (onnxNodeType == "RoIAlign") {
            return OT_RoIAlign;
        } else {
            UNI_ERROR_LOG("operator name:%s type:%s not supported.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        return OT_None;
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

    std::vector<int> get_node_vector_ints_attribute_by_name(
        const onnx::NodeProto &node, const char *key)
    {
        std::vector<int> result;
        int id = get_attribute_id(node, key);
        if (id < 0) {
            return result;
        }
        const onnx::AttributeProto &attribute = node.attribute(id);
        result.resize(attribute.ints_size());
        for (int j = 0; j < attribute.ints_size(); j++) {
            result[j] = UNI_MIN(attribute.ints(j), INT_MAX);
        }
        return result;
    }

    std::vector<F32> get_node_vector_float_tensor_attribute_by_name(
        const onnx::NodeProto &node, const char *key)
    {
        std::vector<F32> result;
        int id = get_attribute_id(node, key);
        if (id < 0) {
            return result;
        }
        const onnx::AttributeProto &attribute = node.attribute(id);
        CHECK_REQUIREMENT(4 == attribute.type());
        const onnx::TensorProto &tp = attribute.t();
        U8 *value;
        if (tp.has_raw_data()) {
            const std::string &rawData = tp.raw_data();
            value = (U8 *)(rawData.data());
        } else if (tp.data_type() == onnx::TensorProto::FLOAT) {
            value = (U8 *)(tp.float_data().data());
        } else {
            UNI_ERROR_LOG("can not process operator name:%s tensor:%s %s type attribute.\n",
                this->onnxNode.name().c_str(), tp.name().c_str(),
                onnx_data_type_string(tp.data_type()).c_str());
        }

        result.resize(tp.dims(0));
        memcpy(result.data(), value, tp.dims(0) * sizeof(float));
        return result;
    }

    int get_node_single_int_attribute_by_name(
        const onnx::NodeProto &node, const char *key, int defaultValue = 0)
    {
        int id = get_attribute_id(node, key);
        if (id < 0) {
            return defaultValue;
        }
        const onnx::AttributeProto &attribute = node.attribute(id);
        return UNI_MIN(attribute.i(), INT_MAX);
    }

    std::string get_node_str_attribute_by_name(const onnx::NodeProto &node,
        const char *key,
        const std::string &defaultValue = std::string())
    {
        int id = get_attribute_id(node, key);
        if (id < 0) {
            return defaultValue;
        }
        return node.attribute(id).s();
    }

    float get_node_float_attribute_by_name(
        const onnx::NodeProto &node, const char *key, float defaultValue = 0.f)
    {
        int id = get_attribute_id(node, key);
        if (id < 0) {
            return defaultValue;
        }
        return node.attribute(id).f();
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
                UNI_ERROR_LOG("can not process onnx converter name:%s tensor:%s %s type raw "
                              "tensor.\n",
                    this->onnxNode.name().c_str(), tensorProto.name().c_str(),
                    onnx_data_type_string(tensorProto.data_type()).c_str());
            }
        } else if (tensorProto.data_type() == onnx::TensorProto::FLOAT) {
            size = tensorProto.float_data_size();
        } else {
            UNI_ERROR_LOG("can not process operator name:%s tensor:%s %s type tensor.\n",
                this->onnxNode.name().c_str(), tensorProto.name().c_str(),
                onnx_data_type_string(tensorProto.data_type()).c_str());
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
            UNI_ERROR_LOG("can not process operator name:%s tensor:%s %s type tensor desc.\n",
                this->onnxNode.name().c_str(), tp.name().c_str(),
                onnx_data_type_string(tp.data_type()).c_str());
        }
        TensorDesc desc = tensor0d();
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
            UNI_ERROR_LOG("can not process operator name:%s tensor:%s %s type non-raw bool "
                          "tensor.\n",
                this->onnxNode.name().c_str(), tensorProto.name().c_str(),
                onnx_data_type_string(tensorProto.data_type()).c_str());
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
            UNI_ERROR_LOG("can not process operator name:%s tensor:%s %s type weight.\n",
                this->onnxNode.name().c_str(), tensorProto.name().c_str(),
                onnx_data_type_string(tensorProto.data_type()).c_str());
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
            UNI_ERROR_LOG("can not process operator name:%s tensor:%s %s type tensor.\n",
                this->onnxNode.name().c_str(), tp.name().c_str(),
                onnx_data_type_string(tp.data_type()).c_str());
        }
        return shape;
    }

    float getSinFloat_from_tensorProto(const onnx::TensorProto &tp)
    {
        float value = 0;
        int size = get_data_size_from_tensor_proto(tp);
        auto type = tp.data_type();
        if (size == 1) {
            if (type == onnx::TensorProto::FLOAT) {
                memcpy(&value, get_ptr_from_weight_obj(tp), sizeof(float));
            } else if (type == onnx::TensorProto::INT64 || type == onnx::TensorProto::INT32) {
                value = get_int_vec_from_tensorProto(tp)[0];
            } else {
                UNI_ERROR_LOG("can not process operator name:%s tensor:%s %d-%s type tensor.\n",
                    this->onnxNode.name().c_str(), tp.name().c_str(), size,
                    onnx_data_type_string(type).c_str());
            }
        } else {
            UNI_ERROR_LOG("can not process operator name:%s tensor:%s %d-%s type tensor.\n",
                this->onnxNode.name().c_str(), tp.name().c_str(), size,
                onnx_data_type_string(type).c_str());
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

    void assign_weight(WeightSpec &ws,
        std::string opName,
        std::vector<onnx::TensorProto> weightTp,
        std::vector<onnx::TensorProto> biasTp)
    {
        // basic
        str_copy(ws.op_name, opName.c_str(), opName.length());
        ws.mdt = DT_F32;
        // aasign onnxWeights
        if (weightTp.size() == 0) {
            ws.bytes_of_weight = 0;
            ws.weight = nullptr;
        } else {
            U8 *weight_ptr = get_ptr_from_weight_obj(weightTp[0]);
            ws.bytes_of_weight = get_data_size_from_tensor_proto(weightTp[0]) * sizeof(float);
            ws.weight = (U8 *)mt_new_storage(ws.bytes_of_weight);
            memcpy(ws.weight, weight_ptr, ws.bytes_of_weight);
        }
        // assign bias
        if (biasTp.size() == 0) {
            ws.bytes_of_vec = 0;
            ws.vec = nullptr;
        } else {
            U8 *vec_ptr = get_ptr_from_weight_obj(biasTp[0]);
            ws.bytes_of_vec = get_data_size_from_tensor_proto(biasTp[0]) * sizeof(float);
            ws.vec = (U8 *)mt_new_storage(ws.bytes_of_vec);
            memcpy(ws.vec, vec_ptr, ws.bytes_of_vec);
        }
    }

    EE parse_file(std::string dir, std::string mfn) override
    {
        std::string onnxSuffix = ".onnx";
        std::string onnxPath = dir + "/" + mfn + onnxSuffix;

        EE ret = read_from_onnx_file(onnxPath.c_str(), (google::protobuf::Message *)(&onnxModel));
        if (ret != SUCCESS) {
            UNI_ERROR_LOG("can not read onnx model file %s.\n", onnxPath.c_str());
        }

        onnxGraph = onnxModel.graph();

        for (int i = 0; i < onnxGraph.initializer_size(); i++) {
            const onnx::TensorProto &initializer = onnxGraph.initializer(i);
            onnxWeights[initializer.name()] = initializer;
        }

        for (int i = 0; i < onnxGraph.value_info_size(); i++) {
            const onnx::ValueInfoProto &value = onnxGraph.value_info(i);
            onnxValues[value.name()] = value;
        }
        return ret;
    }

    std::string crop_name(const std::string &name)
    {
        std::string ret;
        if (name.length() < NAME_LEN) {
            ret = name;
        } else if (this->nameMap.find(name) != this->nameMap.end()) {
            ret = this->nameMap[name];
        } else {
            ret = "brief_" + std::to_string(this->nameMap.size());
            this->nameMap[name] = ret;
        }
        return ret;
    }

    std::string get_name(const onnx::NodeProto &node)
    {
        std::string opName = node.name();
        if (opName.empty() && node.output_size() > 0) {
            opName = node.output(0);
        }
        return opName;
    }

    EE adapt_operators(ModelSpec *ms) override
    {
        str_copy(ms->model_name, onnxGraph.name().c_str(), onnxGraph.name().length());
        ms->dt = DT_F32;

        ms->num_inputs = 0;
        for (int i = 0; i < onnxGraph.input().size(); i++) {
            const std::string &input_name = onnxGraph.input(i).name();
            if (onnxWeights.find(input_name) != onnxWeights.end()) {
                continue;
            }
            ms->num_inputs++;
        }
        ms->input_names = (I8 **)mt_new_storage(ms->num_inputs * sizeof(I8 *));
        ms->input_dims = (TensorDesc *)mt_new_storage(sizeof(TensorDesc) * ms->num_inputs);
        for (int i = 0, index = 0; i < onnxGraph.input().size(); i++) {
            auto input_node = onnxGraph.input(i);
            auto input_name = input_node.name();
            if (onnxWeights.find(input_name) != onnxWeights.end()) {
                continue;
            }
            ms->input_names[index] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            input_name = this->crop_name(input_name);
            str_copy(ms->input_names[index], input_name.c_str(), input_name.length());

            ms->input_dims[index] = tensor0d();
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
        for (int i = 0; i < onnxGraph.output().size(); i++) {
            std::string output_name = onnxGraph.output(i).name();
            output_name = this->crop_name(output_name);
            ms->output_names[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            str_copy(ms->output_names[i], output_name.c_str(), output_name.length());
        }

        for (int nodeIndex = 0; nodeIndex < onnxGraph.node_size(); nodeIndex++) {
            this->onnxNode = onnxGraph.node(nodeIndex);
            const std::string &onnxNodeType = this->onnxNode.op_type();
            for (int j = 0; j < this->onnxNode.input_size(); j++) {
                const std::string &input_name = this->onnxNode.input(j);
                if (this->onnxWeights.find(input_name) != this->onnxWeights.end()) {
#if 0
                    this->onnxWeightReferCount[input_name] = 1;
#else
                    if (this->onnxWeightReferCount.find(input_name) ==
                        this->onnxWeightReferCount.end()) {
                        this->onnxWeightReferCount[input_name] = 1;
                    } else {
                        this->onnxWeightReferCount[input_name]++;
                    }
                    if (onnxNodeType == "Gemm" && j == 2) {
                        auto BName = this->onnxNode.input(1);
                        auto CName = this->onnxNode.input(2);
                        this->onnxWeightReferCount[CName] = UNI_MAX(
                            this->onnxWeightReferCount[BName], this->onnxWeightReferCount[CName]);
                    }
#endif
                }
            }
        }
        for (auto iter = this->onnxWeightReferCount.begin();
             iter != this->onnxWeightReferCount.end(); iter++) {
            if (iter->second > 1) {
                this->sharedWeights.insert(iter->first);
            }
        }

        std::vector<OperatorSpec> operatorSpecVec;
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
                if (opType == OT_Eltwise || opType == OT_Concat || opType == OT_MatMul) {
                    inputNames.push_back(input_name);
                } else if (input_name == "" ||
                    this->onnxWeights.find(input_name) != this->onnxWeights.end()) {
                    if (opType == OT_Gather || opType == OT_FC) {
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

            OperatorSpec operatorSpec;
            str_copy(operatorSpec.name, opName.c_str(), opName.length());
            operatorSpec.type = opType;
            operatorSpec.num_inputs = inputNames.size();
            operatorSpec.input_tensors_name =
                (I8 **)mt_new_storage(operatorSpec.num_inputs * sizeof(I8 *));
            for (U32 j = 0; j < operatorSpec.num_inputs; j++) {
                inputNames[j] = crop_name(inputNames[j]);
                operatorSpec.input_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(operatorSpec.input_tensors_name[j], inputNames[j].c_str(),
                    inputNames[j].length());
            }
            operatorSpec.num_outputs = outputNames.size();
            operatorSpec.output_tensors_name =
                (I8 **)mt_new_storage(operatorSpec.num_outputs * sizeof(I8 *));
            for (U32 j = 0; j < operatorSpec.num_outputs; j++) {
                outputNames[j] = crop_name(outputNames[j]);
                operatorSpec.output_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(operatorSpec.output_tensors_name[j], outputNames[j].c_str(),
                    outputNames[j].length());
            }

            CHECK_STATUS(adapt_operator(opType, &(operatorSpec.ps)));
            operatorSpecVec.push_back(operatorSpec);

            if (onnxNodeType == "BatchNormalization") {
                OperatorSpec operatorSpec;
                std::string scaleInputName = outputNames[0];
                std::string scaleOpName = opName + "_scale";
                str_copy(operatorSpec.name, scaleOpName.c_str(), scaleOpName.length());
                operatorSpec.type = OT_Scale;
                operatorSpec.num_inputs = 1;
                operatorSpec.input_tensors_name = (I8 **)mt_new_storage(sizeof(I8 *));
                operatorSpec.input_tensors_name[0] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(operatorSpec.input_tensors_name[0], scaleInputName.c_str(),
                    scaleInputName.length());
                operatorSpec.num_outputs = 1;
                operatorSpec.output_tensors_name = (I8 **)mt_new_storage(sizeof(I8 *));
                operatorSpec.output_tensors_name[0] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(operatorSpec.output_tensors_name[0], scaleInputName.c_str(),
                    scaleInputName.length());

                CHECK_STATUS(adapt_operator(operatorSpec.type, &(operatorSpec.ps)));
                operatorSpecVec.push_back(operatorSpec);
            }
        }

        std::vector<OperatorSpec> sharedWeightOperatorSpecVec;
        for (auto iter = this->sharedWeights.begin(); iter != this->sharedWeights.end(); iter++) {
            OperatorSpec tmpOps = mt_create_operator(iter->c_str(), OT_SharedWeight, 0, 1);
            str_copy(tmpOps.output_tensors_name[0], iter->c_str(), iter->length());
            const auto &weightTp = onnxWeights[*iter];
            tmpOps.ps.shared_weight_spec.desc = genDescFromTp(weightTp);
            int num = get_data_size_from_tensor_proto(weightTp);
            if (tmpOps.ps.shared_weight_spec.desc.nDims == 0 && num > 0) {
                tmpOps.ps.shared_weight_spec.desc.nDims = 1;
                tmpOps.ps.shared_weight_spec.desc.dims[0] = num;
            }
            sharedWeightOperatorSpecVec.push_back(tmpOps);
        }
        ms->num_operator_specs = sharedWeightOperatorSpecVec.size() + operatorSpecVec.size();
        ms->ops = (OperatorSpec *)mt_new_storage(sizeof(OperatorSpec) * ms->num_operator_specs);
        memcpy(ms->ops, sharedWeightOperatorSpecVec.data(),
            sizeof(OperatorSpec) * sharedWeightOperatorSpecVec.size());
        memcpy(ms->ops + sharedWeightOperatorSpecVec.size(), operatorSpecVec.data(),
            sizeof(OperatorSpec) * operatorSpecVec.size());
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            ms->ops[i].tensor_positions = nullptr;
            ms->ops[i].num_quant_feature = 0;
            ms->ops[i].feature_scale = nullptr;
        }
        return SUCCESS;
    }

    EE adapt_weights(ModelSpec *ms) override
    {
        std::vector<WeightSpec> weightSpecVec;
        for (auto iter = this->sharedWeights.begin(); iter != this->sharedWeights.end(); iter++) {
            const auto &weightTp = onnxWeights[*iter];
            DataType dt;
            std::vector<int> intVec;
            U8 *ptr;
            switch (weightTp.data_type()) {
                case onnx::TensorProto::INT64:
                case onnx::TensorProto::INT32:
                case onnx::TensorProto::UINT64:
                case onnx::TensorProto::UINT32:
                    intVec = get_int_vec_from_tensorProto(weightTp);
                    ptr = (U8 *)intVec.data();
                    dt = DT_I32;
                    break;
                case onnx::TensorProto::DOUBLE:
                case onnx::TensorProto::FLOAT:
                case onnx::TensorProto::FLOAT16:
                case onnx::TensorProto::BFLOAT16:
                    ptr = get_ptr_from_weight_obj(weightTp);
                    dt = DT_F32;
                    break;
                default:
                    UNI_ERROR_LOG("can not process tensor:%s %s type weight.\n",
                        weightTp.name().c_str(),
                        onnx_data_type_string(weightTp.data_type()).c_str());
                    break;
            }
            U32 bytes = get_data_size_from_tensor_proto(weightTp) * bytesOf(dt);
            WeightSpec weightSpec = mt_create_weight(iter->c_str(), dt, bytes, 0, 0);
            memcpy(weightSpec.weight, ptr, bytes);
            weightSpecVec.push_back(weightSpec);
        }
        for (int nodeIndex = 0; nodeIndex < onnxGraph.node_size(); nodeIndex++) {
            this->onnxNode = onnxGraph.node(nodeIndex);
            std::string opName = get_name(this->onnxNode);
            const std::string &onnxNodeType = this->onnxNode.op_type();
            UNI_DEBUG_LOG(
                "process operator name:%s type:%s weight.\n", opName.c_str(), onnxNodeType.c_str());
            if (this->nameMap.find(opName) != this->nameMap.end()) {
                opName = this->nameMap[opName];
            }
            auto indices = getOperatorWeightInputIndex(this->onnxNode);

            WeightSpec weightSpec;
            if (onnxNodeType == "Conv" || onnxNodeType == "ConvTranspose") {
                // if convInputNum == 3, means has bias , otherwise do not have bias
                int convInputNum = this->onnxNode.input_size();

                const onnx::TensorProto &convWeightTp = onnxWeights[this->onnxNode.input(1)];

                int convWeightNum = get_data_size_from_tensor_proto(convWeightTp);
                U8 *convWeightParamPtr = get_ptr_from_weight_obj(convWeightTp);
                str_copy(weightSpec.op_name, opName.c_str(), opName.length());

                // Please do not change to bytesOf(mdt)
                weightSpec.bytes_of_weight = convWeightNum * sizeof(float);
                weightSpec.weight = (U8 *)mt_new_storage(weightSpec.bytes_of_weight);
                memcpy(weightSpec.weight, convWeightParamPtr, weightSpec.bytes_of_weight);
                // traverse weight elements to see whether it is bnn convolution
                weightSpec.mdt = get_weight_data_type(convWeightNum, (F32 *)weightSpec.weight);

                int convBiasNum = 0;
                U8 *convBiasParamPtr = nullptr;
                if (convInputNum == 3) {
                    const onnx::TensorProto &convBiasTp = onnxWeights[this->onnxNode.input(2)];
                    convBiasNum = get_data_size_from_tensor_proto(convBiasTp);
                    convBiasParamPtr = get_ptr_from_weight_obj(convBiasTp);
                    weightSpec.bytes_of_vec = convBiasNum * sizeof(float);
                    if (DT_BIN11 == weightSpec.mdt || DT_BIN01 == weightSpec.mdt) {
                        // BNN conv must have a scale vector and a bias vector, so that it can fuse with BN
                        weightSpec.bytes_of_vec *= 2;
                    }
                    weightSpec.vec = (U8 *)mt_new_storage(weightSpec.bytes_of_vec);
                    if (DT_BIN11 == weightSpec.mdt || DT_BIN01 == weightSpec.mdt) {
                        U32 vecBytes = convBiasNum * sizeof(float);
                        F32 *scale = (F32 *)weightSpec.vec;
                        for (I32 j = 0; j < convBiasNum; j++) {
                            scale[j] = 1.0;
                        }
                        // Copy bias (if any) to the second half for BNN
                        memcpy(weightSpec.vec + vecBytes, convBiasParamPtr, vecBytes);
                    } else {
                        memcpy(weightSpec.vec, convBiasParamPtr, weightSpec.bytes_of_vec);
                    }
                } else {
                    weightSpec.bytes_of_vec = 0;
                    weightSpec.vec = nullptr;
                }
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "Gemm" || onnxNodeType == "Linear") {
                str_copy(weightSpec.op_name, opName.c_str(), opName.length());
                weightSpec.mdt = DT_F32;
                int transB = 1;
                if (onnxNodeType == "Linear" ||
                    this->onnxWeightReferCount[this->onnxNode.input(2)] > 1) {
                    weightSpec.bytes_of_vec = 0;
                    weightSpec.vec = nullptr;
                } else {
                    const onnx::TensorProto &fcBiasTp = onnxWeights[this->onnxNode.input(2)];
                    int fcBiasNum = get_data_size_from_tensor_proto(fcBiasTp);
                    U8 *fcBiasParamPtr = get_ptr_from_weight_obj(fcBiasTp);
                    weightSpec.bytes_of_vec = fcBiasNum * sizeof(float);
                    weightSpec.vec = (U8 *)mt_new_storage(weightSpec.bytes_of_vec);
                    memcpy(weightSpec.vec, fcBiasParamPtr, weightSpec.bytes_of_vec);
                }
                if (onnxNodeType == "Gemm") {
                    transB = get_node_single_int_attribute_by_name(this->onnxNode, "transB", 0);
                }
                if (this->onnxWeightReferCount[this->onnxNode.input(1)] > 1) {
                    weightSpec.bytes_of_weight = 0;
                    weightSpec.weight = nullptr;
                } else {
                    const onnx::TensorProto &fcWeightTp = onnxWeights[this->onnxNode.input(1)];
                    int fcWeightNum = get_data_size_from_tensor_proto(fcWeightTp);
                    U8 *fcWeightParamPtr = get_ptr_from_weight_obj(fcWeightTp);
                    weightSpec.bytes_of_weight = fcWeightNum * sizeof(float);
                    weightSpec.weight = (U8 *)mt_new_storage(weightSpec.bytes_of_weight);
                    if (transB) {
                        memcpy(weightSpec.weight, fcWeightParamPtr, fcWeightNum * sizeof(float));
                    } else {
                        memcpy_trans2d(weightSpec.weight, fcWeightParamPtr, (int)fcWeightTp.dims(1),
                            (int)fcWeightTp.dims(0));
                    }
                }
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "BatchNormalization") {
                const onnx::TensorProto &scale = onnxWeights[this->onnxNode.input(1)];
                const onnx::TensorProto &bias = onnxWeights[this->onnxNode.input(2)];
                const onnx::TensorProto &mean = onnxWeights[this->onnxNode.input(3)];
                const onnx::TensorProto &var = onnxWeights[this->onnxNode.input(4)];

                U8 *meanPtr = get_ptr_from_weight_obj(mean);
                int bnMeanNum = get_data_size_from_tensor_proto(mean);
                U8 *varPtr = get_ptr_from_weight_obj(var);
                int bnVarNum = get_data_size_from_tensor_proto(var);

                str_copy(weightSpec.op_name, opName.c_str(), opName.length());
                weightSpec.mdt = DT_F32;
                weightSpec.bytes_of_weight = bnMeanNum * sizeof(float);
                weightSpec.bytes_of_vec = bnVarNum * sizeof(float);

                weightSpec.weight = (U8 *)mt_new_storage(weightSpec.bytes_of_weight);
                memcpy(weightSpec.weight, meanPtr, weightSpec.bytes_of_weight);
                weightSpec.vec = (U8 *)mt_new_storage(weightSpec.bytes_of_vec);
                memcpy(weightSpec.vec, varPtr, weightSpec.bytes_of_vec);
                weightSpecVec.push_back(weightSpec);

                // for scale
                std::string scaleWeightOpName = opName + "_scale";
                U8 *scalePtr = get_ptr_from_weight_obj(scale);
                int scaleWeightNum = get_data_size_from_tensor_proto(scale);
                U8 *biasPtr = get_ptr_from_weight_obj(bias);
                int scaleBiasNum = get_data_size_from_tensor_proto(bias);

                str_copy(weightSpec.op_name, scaleWeightOpName.c_str(), scaleWeightOpName.length());
                weightSpec.mdt = DT_F32;
                weightSpec.bytes_of_weight = scaleWeightNum * sizeof(float);
                weightSpec.bytes_of_vec = scaleBiasNum * sizeof(float);

                weightSpec.weight = (U8 *)mt_new_storage(weightSpec.bytes_of_weight);
                memcpy(weightSpec.weight, scalePtr, weightSpec.bytes_of_weight);
                weightSpec.vec = (U8 *)mt_new_storage(weightSpec.bytes_of_vec);
                memcpy(weightSpec.vec, biasPtr, weightSpec.bytes_of_vec);
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "BatchNorm" || onnxNodeType == "InstanceNormalization") {
                std::vector<onnx::TensorProto> weightTp = {onnxWeights[this->onnxNode.input(1)]};
                std::vector<onnx::TensorProto> biasTp = {onnxWeights[this->onnxNode.input(2)]};
                assign_weight(weightSpec, opName, weightTp, biasTp);
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "Tdnn") {
                std::vector<onnx::TensorProto> weightTp = {onnxWeights[this->onnxNode.input(2)]};
                std::vector<onnx::TensorProto> biasTp = {onnxWeights[this->onnxNode.input(3)]};
                assign_weight(weightSpec, opName, weightTp, biasTp);
                weightSpecVec.push_back(weightSpec);
            } else if ((onnxNodeType == "MatMul" || onnxNodeType == "PRelu") && indices.size() > 0) {
                str_copy(weightSpec.op_name, opName.c_str(), opName.length());
                weightSpec.mdt = DT_F32;
                std::string weightName = this->onnxNode.input(indices[0]);
                if (onnxNodeType == "MatMul" &&
                    (this->onnxWeightReferCount[weightName] > 1 ||
                        this->sharedWeights.find(weightName) != this->sharedWeights.end())) {
                    weightSpec.bytes_of_weight = 0;
                    weightSpec.weight = nullptr;
                } else {
                    const onnx::TensorProto &weight = onnxWeights[weightName];
                    U8 *weight_ptr = get_ptr_from_weight_obj(weight);
                    int weight_num = get_data_size_from_tensor_proto(weight);
                    weightSpec.bytes_of_weight = weight_num * sizeof(float);
                    weightSpec.weight = (U8 *)mt_new_storage(weightSpec.bytes_of_weight);
                    int row = weight.dims(0);
                    int column = weight_num / row;
                    for (int m = 0, index = 0; m < column; m++) {
                        for (int n = 0; n < row; n++, index += sizeof(float)) {
                            memcpy(weightSpec.weight + index,
                                weight_ptr + (n * column + m) * sizeof(float), sizeof(float));
                        }
                    }
                }
                weightSpec.bytes_of_vec = 0;
                weightSpec.vec = nullptr;
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "Mul" || onnxNodeType == "Div") {
                if (indices.size() == 0 ||
                    get_data_size_from_tensor_proto(
                        onnxWeights[this->onnxNode.input(indices[0])]) == 1 ||
                    this->sharedWeights.find(this->onnxNode.input(indices[0])) !=
                        this->sharedWeights.end()) {
                    continue;
                }
                std::vector<onnx::TensorProto> weightTp = {
                    onnxWeights[this->onnxNode.input(indices[0])]};
                std::vector<onnx::TensorProto> biasTp;
                assign_weight(weightSpec, opName, weightTp, biasTp);
                if (onnxNodeType == "Div") {
                    F32 *scale = (F32 *)weightSpec.weight;
                    for (U32 j = 0; j < weightSpec.bytes_of_weight / sizeof(float); j++) {
                        scale[j] = 1 / scale[j];
                    }
                }
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "Add" || onnxNodeType == "Sub") {
                if (indices.size() == 0 ||
                    get_data_size_from_tensor_proto(
                        onnxWeights[this->onnxNode.input(indices[0])]) == 1 ||
                    this->sharedWeights.find(this->onnxNode.input(indices[0])) !=
                        this->sharedWeights.end()) {
                    continue;
                }
                std::vector<onnx::TensorProto> weightTp;
                std::vector<onnx::TensorProto> biasTp = {
                    onnxWeights[this->onnxNode.input(indices[0])]};
                assign_weight(weightSpec, opName, weightTp, biasTp);
                if (onnxNodeType == "Sub") {
                    F32 *scale = (F32 *)weightSpec.vec;
                    for (U32 j = 0; j < weightSpec.bytes_of_vec / sizeof(float); j++) {
                        scale[j] = (-1) * scale[j];
                    }
                }
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "Transpose" && indices.size() > 0) {
                std::vector<onnx::TensorProto> weightTp = {onnxWeights[this->onnxNode.input(0)]};
                std::vector<onnx::TensorProto> biasTp;
                assign_weight(weightSpec, opName, weightTp, biasTp);
                weightSpecVec.push_back(weightSpec);
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
                    if (0 !=
                        get_node_single_int_attribute_by_name(
                            this->onnxNode, "linear_before_reset", 0)) {
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
                U8 *W_ptr = get_ptr_from_weight_obj(W);
                U8 *R_ptr = get_ptr_from_weight_obj(R);
                U8 *B_ptr = get_ptr_from_weight_obj(B);
                weightSpec.mdt = DT_F32;
                str_copy(weightSpec.op_name, opName.c_str(), opName.length());
                weightSpec.bytes_of_weight =
                    (W.dims(0) * W.dims(1) * (W.dims(2) + R.dims(2))) * sizeof(float);
                weightSpec.weight = (U8 *)mt_new_storage(weightSpec.bytes_of_weight);
                weightSpec.bytes_of_vec = biasNum * sizeof(float);
                weightSpec.vec = (U8 *)mt_new_storage(weightSpec.bytes_of_vec);
                int hidden = W.dims(1) / gates;
                U8 *weightPtr = weightSpec.weight;
                F32 *biasPtr = (F32 *)weightSpec.vec;
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
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "Splice") {
                std::vector<int> indices =
                    get_node_vector_ints_attribute_by_name(this->onnxNode, "forward_indexes");
                str_copy(weightSpec.op_name, opName.c_str(), opName.length());
                weightSpec.mdt = DT_U32;
                weightSpec.bytes_of_weight = indices.size() * sizeof(U32);
                weightSpec.weight = (U8 *)mt_new_storage(weightSpec.bytes_of_weight);
                memcpy(weightSpec.weight, indices.data(), weightSpec.bytes_of_weight);
                weightSpec.bytes_of_vec = 0;
                weightSpec.vec = nullptr;
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "Where") {
                bool *conditionTpPtr = nullptr;
                int conditionTpSize = 0;
                std::vector<float> conditionVec;
                if (onnxWeights.find(this->onnxNode.input(0)) != onnxWeights.end()) {
                    auto conditionTp = onnxWeights[this->onnxNode.input(0)];
                    conditionTpPtr = (bool *)(get_ptr_from_weight_obj(conditionTp));
                    conditionTpSize = get_data_size_from_tensor_proto(conditionTp);
                    for (int i = 0; i < conditionTpSize; i++) {
                        float curCon = (conditionTpPtr[i] == true) ? 1.0 : 0.0;
                        conditionVec.push_back(curCon);
                    }
                }
                U8 *yPtr = nullptr;
                int yTpSize = 0;
                if (onnxWeights.find(this->onnxNode.input(2)) != onnxWeights.end()) {
                    auto yTp = onnxWeights[this->onnxNode.input(2)];
                    yPtr = get_ptr_from_weight_obj(yTp);
                    yTpSize = get_data_size_from_tensor_proto(yTp);
                }
                weightSpec.mdt = DT_F32;
                str_copy(weightSpec.op_name, opName.c_str(), opName.length());
                weightSpec.bytes_of_weight = conditionTpSize * sizeof(float);
                if (weightSpec.bytes_of_weight == 0) {
                    weightSpec.weight = nullptr;
                } else {
                    weightSpec.weight = (U8 *)mt_new_storage(weightSpec.bytes_of_weight);
                    memcpy(weightSpec.weight, conditionVec.data(), weightSpec.bytes_of_weight);
                }
                weightSpec.bytes_of_vec = yTpSize * sizeof(float);
                if (weightSpec.bytes_of_vec == 0) {
                    weightSpec.vec = nullptr;
                } else {
                    weightSpec.vec = (U8 *)mt_new_storage(weightSpec.bytes_of_vec);
                    memcpy(weightSpec.vec, yPtr, weightSpec.bytes_of_vec);
                }
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "Equal") {
                auto cmpTp = onnxWeights[this->onnxNode.input(1)];
                int cmpTpSize = get_data_size_from_tensor_proto(cmpTp);
                if (cmpTp.data_type() == onnx::TensorProto::FLOAT) {
                    weightSpec.mdt = DT_F32;
                } else if (cmpTp.data_type() == onnx::TensorProto::INT32) {
                    weightSpec.mdt = DT_I32;
                } else {
                    UNI_ERROR_LOG("can not process operator name:%s %s type Equal.\n",
                        this->onnxNode.name().c_str(),
                        onnx_data_type_string(cmpTp.data_type()).c_str());
                }
                U8 *cmpPtr = (U8 *)get_ptr_from_weight_obj(cmpTp);
                str_copy(weightSpec.op_name, opName.c_str(), opName.length());
                weightSpec.bytes_of_weight = cmpTpSize * sizeof(float);
                weightSpec.weight = (U8 *)mt_new_storage(weightSpec.bytes_of_weight);
                memcpy(weightSpec.weight, cmpPtr, weightSpec.bytes_of_weight);
                weightSpec.bytes_of_vec = 0;
                weightSpec.vec = nullptr;
                weightSpecVec.push_back(weightSpec);
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
                            if (onnxWeights.find(curNode.input(k)) != onnxWeights.end()) {
                                auto hidWeightTp = onnxWeights[curNode.input(k)];
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
                TensorDesc wDesc1 = genDescFromTp(weightMap["Gemm"][0]);
                TensorDesc bDesc1;
                if (weightMap["Gemm"].size() > 1) {
                    bDesc1 = genDescFromTp(weightMap["Gemm"][1]);
                } else {
                    bDesc1 = tensor0d();
                }
                TensorDesc wDesc2;
                if (weightMap["MatMul"].size() > 0) {
                    wDesc2 = genDescFromTp(weightMap["MatMul"][0]);
                } else {
                    wDesc2 = tensor0d();
                }
                TensorDesc bDesc2;
                if (weightMap["MatMul"].size() > 1) {
                    bDesc2 = genDescFromTp(weightMap["MatMul"][1]);
                } else {
                    bDesc2 = tensor0d();
                }

                weightSpec.mdt = DT_F32;
                str_copy(weightSpec.op_name, opName.c_str(), opName.length());
                int wBytes = tensorNumElements(wDesc1) + tensorNumElements(wDesc2);
                weightSpec.bytes_of_weight = wBytes * sizeof(float);
                if (weightSpec.bytes_of_weight == 0) {
                    weightSpec.weight = nullptr;
                } else {
                    weightSpec.weight = (U8 *)mt_new_storage(weightSpec.bytes_of_weight);
                    int wOffSet = 0;
                    if (tensorNumElements(wDesc1) > 0) {
                        U8 *tmpWPtr1 = get_ptr_from_weight_obj(weightMap["Gemm"][0]);
                        if (transMap["Gemm"][0]) {
                            memcpy(
                                &((weightSpec.weight)[wOffSet]), tmpWPtr1, tensorNumBytes(wDesc1));
                        } else {
                            memcpy_trans2d(&((weightSpec.weight)[wOffSet]), tmpWPtr1,
                                (int)weightMap["Gemm"][0].dims(1),
                                (int)weightMap["Gemm"][0].dims(0));
                        }
                        wOffSet += tensorNumBytes(wDesc1);
                    }
                    if (tensorNumElements(wDesc2) > 0) {
                        U8 *tmpWPtr2 = get_ptr_from_weight_obj(weightMap["MatMul"][0]);
                        if (transMap["MatMul"][0]) {
                            memcpy(
                                &((weightSpec.weight)[wOffSet]), tmpWPtr2, tensorNumBytes(wDesc2));
                        } else {
                            memcpy_trans2d(&((weightSpec.weight)[wOffSet]), tmpWPtr2,
                                (int)weightMap["MatMul"][0].dims(1),
                                (int)weightMap["MatMul"][0].dims(0));
                        }
                    }
                }

                int bBytes = tensorNumElements(bDesc1) + tensorNumElements(bDesc2);
                weightSpec.bytes_of_vec = bBytes * sizeof(float);
                if (weightSpec.bytes_of_vec == 0) {
                    weightSpec.vec = nullptr;
                } else {
                    weightSpec.vec = (U8 *)mt_new_storage(weightSpec.bytes_of_vec);
                    int bOffSet = 0;
                    if (tensorNumElements(bDesc1) > 0) {
                        U8 *tmpBPtr1 = get_ptr_from_weight_obj(weightMap["Gemm"][1]);
                        memcpy(&((weightSpec.vec)[bOffSet]), tmpBPtr1, tensorNumBytes(bDesc1));
                        bOffSet += tensorNumBytes(bDesc1);
                    }
                    if (tensorNumElements(bDesc2) > 0) {
                        U8 *tmpBPtr2 = get_ptr_from_weight_obj(weightMap["MatMul"][1]);
                        memcpy(&((weightSpec.vec)[bOffSet]), tmpBPtr2, tensorNumBytes(bDesc2));
                    }
                }
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "ScatterND" || onnxNodeType == "ScatterElements" ||
                onnxNodeType == "Gather" || onnxNodeType == "GatherND" ||
                onnxNodeType == "GatherElements") {
                std::vector<onnx::TensorProto> weightTp, biasTp;
                const std::string &input0 = this->onnxNode.input(0);
                if (onnxWeights.find(input0) != onnxWeights.end() &&
                    this->onnxWeightReferCount[input0] == 1) {
                    weightTp.push_back(onnxWeights[this->onnxNode.input(0)]);
                }
                // update tensor
                if (onnxNodeType == "ScatterND" || onnxNodeType == "ScatterElements") {
                    const std::string &input2 = this->onnxNode.input(2);
                    if (onnxWeights.find(input2) != onnxWeights.end() &&
                        this->onnxWeightReferCount[input2] == 1) {
                        weightTp.push_back(onnxWeights[this->onnxNode.input(2)]);
                    }
                }
                assign_weight(weightSpec, opName, weightTp, biasTp);
                const std::string &input1 = this->onnxNode.input(1);
                if (onnxWeights.find(input1) != onnxWeights.end() &&
                    this->onnxWeightReferCount[input1] == 1) {
                    std::vector<int> index = get_int_vec_from_tensorProto(onnxWeights[input1]);
                    weightSpec.bytes_of_vec = sizeof(int) * index.size();
                    weightSpec.vec = (U8 *)mt_new_storage(weightSpec.bytes_of_vec);
                    memcpy(weightSpec.vec, index.data(), weightSpec.bytes_of_vec);
                }
                weightSpecVec.push_back(weightSpec);
            } else if (onnxNodeType == "GenerateProposals") {
                std::vector<onnx::TensorProto> weightTp = {onnxWeights[this->onnxNode.input(3)]};
                std::vector<onnx::TensorProto> biasTp;
                assign_weight(weightSpec, opName, weightTp, biasTp);
                weightSpecVec.push_back(weightSpec);
            }
        }
        ms->num_weight_specs = weightSpecVec.size();
        ms->ws = (WeightSpec *)mt_new_storage(sizeof(WeightSpec) * ms->num_weight_specs);
        memcpy(ms->ws, weightSpecVec.data(), sizeof(WeightSpec) * weightSpecVec.size());
        for (I32 i = 0; i < ms->num_weight_specs; i++) {
            ms->ws[i].num_quant_scale = 0;
            ms->ws[i].weight_scale = nullptr;
        }
        return SUCCESS;
    }

    void insert_shared_weight()
    {
        for (int i = 0; i < this->onnxNode.input_size(); i++) {
            const std::string &name = this->onnxNode.input(i);
            if (onnxWeights.find(name) != onnxWeights.end()) {
                this->sharedWeights.insert(name);
            }
        }
    }

    ParameterSpec adapt_SharedWeight() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        const onnx::TensorProto &data = onnxWeights[this->onnxNode.input(0)];
        const onnx::TensorProto &ind = onnxWeights[this->onnxNode.input(1)];
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
        const std::string &onnxNodeType = this->onnxNode.op_type();
        std::vector<int> reshapeInfo;
        if (onnxNodeType == "Flatten") {
            int axis = get_node_single_int_attribute_by_name(this->onnxNode, "axis", 1);
            for (int i = 0; i < axis; i++) {
                reshapeInfo.push_back(0);
            }
            reshapeInfo.push_back(-1);
        } else {
            if (this->onnxNode.input_size() == 1) {
                reshapeInfo = get_node_vector_ints_attribute_by_name(this->onnxNode, "shape");
            } else {
                reshapeInfo = get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(1)]);
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
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Resize") {
            for (int i = 0; i < this->onnxNode.input_size(); i++) {
                if (onnxWeights.find(this->onnxNode.input(i)) != onnxWeights.end()) {
                    auto curTp = onnxWeights[this->onnxNode.input(i)];
                    if (curTp.data_type() == onnx::TensorProto::FLOAT) {
                        scalesIndex = this->onnxNode.input(i);
                    } else if (curTp.data_type() == onnx::TensorProto::INT64) {
                        sizesIndex = this->onnxNode.input(i);
                    } else {
                        UNI_ERROR_LOG("can not process operator name:%s %s type attributes.\n",
                            this->onnxNode.name().c_str(),
                            onnx_data_type_string(curTp.data_type()).c_str());
                    }
                }
            }
        } else if (onnxNodeType == "Upsample") {
            scalesIndex = this->onnxNode.input(1);
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Resize.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        if (scalesIndex != "") {
            const onnx::TensorProto &scales = onnxWeights[scalesIndex];
            if (scales.dims(0) == 0 || scales.dims(0) == 4) {
                resizePs.num_scales = scales.dims(0);
                U8 *ptr = get_ptr_from_weight_obj(scales);
                memcpy(resizePs.scales, ptr, resizePs.num_scales * bytesOf(DT_F32));
            } else {
                UNI_ERROR_LOG("can not get scale information from operator name:%s type:%s.\n",
                    this->onnxNode.name().c_str(), onnxNodeType.c_str());
            }
        }
        if (sizesIndex != "") {
            const onnx::TensorProto &sizes = onnxWeights[sizesIndex];
            if (sizes.dims(0) == 0) {
            } else if (sizes.dims(0) == 4) {
                std::vector<int> ptr = get_int_vec_from_tensorProto(sizes);
                resizePs.num_sizes = 2;
                resizePs.sizes[0] = ptr[2];
                resizePs.sizes[1] = ptr[3];
            } else {
                UNI_ERROR_LOG("can not get resize information from operator name:%s "
                              "type:%s.\n",
                    this->onnxNode.name().c_str(), onnxNodeType.c_str());
            }
        }

        std::string mode = get_node_str_attribute_by_name(this->onnxNode, "mode", "nearest");
        std::string coordinate_transformation_mode = get_node_str_attribute_by_name(
            this->onnxNode, "coordinate_transformation_mode", "half_pixel");
        std::string nearest_mode =
            get_node_str_attribute_by_name(this->onnxNode, "nearest_mode", "round_prefer_floor");

        if (mode.compare("linear") == 0) {
            resizePs.mode = LINEAR;
        } else if (mode.compare("nearest") == 0) {
            resizePs.mode = NEAREST;
        } else if (mode.compare("cubic") == 0) {
            resizePs.mode = CUBIC;
        } else {
            UNI_ERROR_LOG("can not support mode:%s in operator name:%s type:%s.\n", mode.c_str(),
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }

        if (coordinate_transformation_mode.compare("align_corners") == 0) {
            resizePs.trans_mode = ALIGN_CORNERS;
        } else if (coordinate_transformation_mode.compare("half_pixel") == 0) {
            resizePs.trans_mode = HALF_PIXEL;
        } else if (coordinate_transformation_mode.compare("pytorch_half_pixel") == 0) {
            resizePs.trans_mode = PYTORCH_HALF_PIXEL;
        } else if (coordinate_transformation_mode.compare("asymmetric") == 0) {
            resizePs.trans_mode = ASYMMETRIC;
        } else {
            UNI_ERROR_LOG("can not support coordinate transformation mode:%s in operator name:%s "
                          "type:%s.\n",
                coordinate_transformation_mode.c_str(), this->onnxNode.name().c_str(),
                onnxNodeType.c_str());
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
        std::vector<int> transpose_info =
            get_node_vector_ints_attribute_by_name(this->onnxNode, "perm");
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
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Max") {
            clipParam.min = 0;
            clipParam.max = UNI_F16_MAX;
        } else if (onnxNodeType == "Min") {
            clipParam.min = -UNI_F16_MAX;
            clipParam.max = 1;
        } else {  // onnxNodeType == "Clip"
            if (this->onnxNode.input_size() == 1) {
                clipParam.min =
                    get_node_float_attribute_by_name(this->onnxNode, "min", -UNI_F16_MAX);
                clipParam.max = get_node_float_attribute_by_name(this->onnxNode, "max", UNI_F16_MAX);
            } else {
                if (this->onnxNode.input(1) == "") {
                    clipParam.min = -UNI_F16_MAX;
                } else {
                    clipParam.min =
                        getSinFloat_from_tensorProto(onnxWeights[this->onnxNode.input(1)]);
                }
                if (this->onnxNode.input(2) == "") {
                    clipParam.max = UNI_F16_MAX;
                } else {
                    clipParam.max =
                        getSinFloat_from_tensorProto(onnxWeights[this->onnxNode.input(2)]);
                }
            }
        }
        curPs.clip_spec = clipParam;
        return curPs;
    }

    ParameterSpec adapt_Conv() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ConvolutionParamSpec cps;
        memset(&cps, 0, sizeof(cps));
        std::vector<int> kernelShape =
            get_node_vector_ints_attribute_by_name(this->onnxNode, "kernel_shape");
        std::vector<int> dilations =
            get_node_vector_ints_attribute_by_name(this->onnxNode, "dilations");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(this->onnxNode, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(this->onnxNode, "pads");
        int group = get_node_single_int_attribute_by_name(this->onnxNode, "group", 1);

        const onnx::TensorProto &weight = onnxWeights[this->onnxNode.input(1)];
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
        const std::string &onnxNodeType = this->onnxNode.op_type();
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ConvolutionParamSpec cps;
        memset(&cps, 0, sizeof(cps));
        std::vector<int> kernelShape =
            get_node_vector_ints_attribute_by_name(this->onnxNode, "kernel_shape");
        std::vector<int> dilations =
            get_node_vector_ints_attribute_by_name(this->onnxNode, "dilations");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(this->onnxNode, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(this->onnxNode, "pads");
        int group = get_node_single_int_attribute_by_name(this->onnxNode, "group", 1);
        std::vector<int> output_shapes =
            get_node_vector_ints_attribute_by_name(this->onnxNode, "output_shape");

        const onnx::TensorProto &weight = onnxWeights[this->onnxNode.input(1)];
        cps.num_outputs = weight.dims(1);
        cps.kernel_t = 1;
        cps.kernel_h = 1;
        cps.kernel_w = 1;
        if (kernelShape.size() == 2) {
            cps.kernel_h = kernelShape[0];
            cps.kernel_w = kernelShape[1];
        } else if (kernelShape.size() == 1) {
            cps.kernel_h = kernelShape[0];
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Deconvolution.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
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
        if (onnxValues.find(this->onnxNode.input(0)) != onnxValues.end() &&
            output_shapes.size() > 0) {
            auto shape = onnxValues[this->onnxNode.input(0)].type().tensor_type().shape();
            if (shape.dim().size() > 2) {
                int ih = shape.dim(2).dim_value();
                if (output_shapes[0] == (int)cps.stride_h * ih) {
                    if (output_shapes.size() > 1) {
                        int iw = shape.dim(2).dim_value();
                        if (output_shapes[1] == (int)cps.stride_w * iw) {
                            cps.rm = TF_SAME;
                        }
                    } else {
                        cps.rm = TF_SAME;
                    }
                }
            }
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
        std::string autoPad =
            get_node_str_attribute_by_name(this->onnxNode, "auto_pad");  // deprecated
        std::vector<int> kernelShape =
            get_node_vector_ints_attribute_by_name(this->onnxNode, "kernel_shape");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(this->onnxNode, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(this->onnxNode, "pads");

        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "AveragePool" || onnxNodeType == "ReduceMean" ||
            onnxNodeType == "GlobalAveragePool") {
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
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Gemm") {
            matmulPs.transpose_a =
                get_node_single_int_attribute_by_name(this->onnxNode, "transA", 0);
            matmulPs.transpose_b =
                get_node_single_int_attribute_by_name(this->onnxNode, "transB", 0);
        }
        curPs.matmul_spec = matmulPs;
        insert_shared_weight();
        return curPs;
    }

    ParameterSpec adapt_Fc() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        FullyConnectedParamSpec fcParamSpec;
        memset(&fcParamSpec, 0, sizeof(fcParamSpec));
        fcParamSpec.num_outputs = -1;

        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "MatMul") {
            const onnx::TensorProto &matmulTp = onnxWeights[this->onnxNode.input(1)];
            if (matmulTp.dims_size() == 2) {
                fcParamSpec.num_outputs = matmulTp.dims(1);
            } else {
                UNI_ERROR_LOG("can not map operator name:%s type:%s to FullyConnected.\n",
                    this->onnxNode.name().c_str(), onnxNodeType.c_str());
            }
        } else if (onnxNodeType == "Linear") {
            const onnx::TensorProto &matmulTp = onnxWeights[this->onnxNode.input(1)];
            if (matmulTp.dims_size() == 2) {
                fcParamSpec.num_outputs = matmulTp.dims(0);
            } else {
                UNI_ERROR_LOG("can not map operator name:%s type:%s to FullyConnected.\n",
                    this->onnxNode.name().c_str(), onnxNodeType.c_str());
            }
        } else {
            float alpha = get_node_float_attribute_by_name(this->onnxNode, "alpha", 1.f);
            float beta = get_node_float_attribute_by_name(this->onnxNode, "beta", 1.f);
            int transA = get_node_single_int_attribute_by_name(this->onnxNode, "transA", 0);
            int transB = get_node_single_int_attribute_by_name(this->onnxNode, "transB", 0);
            auto weightTp = onnxWeights[this->onnxNode.input(1)];
            if (transB == 1.0) {
                fcParamSpec.num_outputs = weightTp.dims(0);
            } else {
                fcParamSpec.num_outputs = weightTp.dims(1);
            }
            if (!(alpha == 1.f && beta == 1.f && transA == 0)) {
                UNI_ERROR_LOG("can not map operator name:%s type:%s to FullyConnected.\n",
                    this->onnxNode.name().c_str(), onnxNodeType.c_str());
            }
        }
        fcParamSpec.num_slices = 1;
        fcParamSpec.slice_point[0] = fcParamSpec.num_outputs;
        curPs.fc_spec = fcParamSpec;
        return curPs;
    }

    ParameterSpec adapt_BatchNorm() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        BatchNormParamSpec bnPs;
        memset(&bnPs, 0, sizeof(bnPs));
        bnPs.eps = get_node_float_attribute_by_name(this->onnxNode, "epsilon", 1e-5f);
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "BatchNormalization") {
            bnPs.axis = 1;
        } else if (onnxNodeType == "BatchNorm") {
            bnPs.axis = -1;
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to BatchNorm.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        bnPs.gama = 1;
        bnPs.momentum = get_node_float_attribute_by_name(this->onnxNode, "momentum", 0.9);
        curPs.bn_spec = bnPs;
        return curPs;
    }

    ParameterSpec adapt_InstanceNorm() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        InstanceNormParamSpec inPs;
        memset(&inPs, 0, sizeof(inPs));
        inPs.eps = get_node_float_attribute_by_name(this->onnxNode, "epsilon", 1e-5f);
        inPs.axis = 1;
        inPs.axis_dim = -1;
        curPs.in_spec = inPs;
        return curPs;
    }

    ParameterSpec adapt_Eltwise() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        EltwiseParamSpec eps;
        memset(&eps, 0, sizeof(eps));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Add" || onnxNodeType == "Sum") {
            eps.elt_mode = ELTWISE_SUM;
            eps.elt_sum_spec.coeff_size = 2;
            for (I32 j = 0; j < eps.elt_sum_spec.coeff_size; j++) {
                eps.elt_sum_spec.coeff_values[j] = 1.0;
            }
        } else if (onnxNodeType == "Mul") {
            eps.elt_mode = ELTWISE_PROD;
        } else if (onnxNodeType == "Sub") {
            eps.elt_mode = ELTWISE_SUB;
        } else if (onnxNodeType == "Div") {
            eps.elt_mode = ELTWISE_DIV;
        } else if (onnxNodeType == "And") {
            eps.elt_mode = ELTWISE_AND;
        } else if (onnxNodeType == "Or") {
            eps.elt_mode = ELTWISE_OR;
        } else if (onnxNodeType == "Xor") {
            eps.elt_mode = ELTWISE_XOR;
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Eltwise.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        eps.activation_type = ACTIVATION_NULL;
        curPs.eltwise_spec = eps;

        insert_shared_weight();
        return curPs;
    }

    void handle_Constant()
    {
        for (int i = 0; i < this->onnxNode.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = this->onnxNode.attribute(i);
            if (attribute.name() == "value") {
                CHECK_REQUIREMENT(4 == attribute.type());
                const onnx::TensorProto &tp = attribute.t();
                const auto &output_name = this->onnxNode.output(0);
                this->onnxWeights[output_name] = tp;
                this->onnxWeightReferCount[output_name] = -INT_MAX;
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
        std::string padModeStr = get_node_str_attribute_by_name(this->onnxNode, "mode");
        std::vector<int> padVec = get_node_vector_ints_attribute_by_name(this->onnxNode, "pads");
        F32 padValue = get_node_float_attribute_by_name(this->onnxNode, "value", 0.f);
        if (padModeStr == "constant" || padModeStr.length() == 0) {
            padPs.pad_mode = Pad_Constant;
        } else if (padModeStr == "edge") {
            padPs.pad_mode = Pad_Edge;
        } else if (padModeStr == "reflect") {
            padPs.pad_mode = Pad_Reflect;
        }

        padPs.front = 0;
        padPs.back = 0;
        padPs.before = 0;
        padPs.after = 0;
        U32 padSize = padVec.size();
        if (padSize == 0) {
            const onnx::TensorProto &padsTp = onnxWeights[this->onnxNode.input(1)];
            padVec = get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(1)]);
            padSize = padVec.size();
        }
        if (padSize == 8) {  // NCHW
            padPs.front = padVec[1];
            padPs.top = padVec[2];
            padPs.left = padVec[3];
            padPs.back = padVec[5];
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
                this->onnxNode.name().c_str(), this->onnxNode.op_type().c_str());
        }
        padPs.constant_value = padValue;
        curPs.pad_spec = padPs;
        return curPs;
    }

    ParameterSpec adapt_Gather() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        GatherParamSpec p;
        memset(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Gather" || onnxNodeType == "GatherElements") {
            p.axis = get_node_single_int_attribute_by_name(this->onnxNode, "axis", 0);
        } else {
            p.axis = INT_MAX;
        }
        if (onnxNodeType == "GatherElements") {
            p.element_level = true;
        } else {
            p.element_level = false;
        }
        if (onnxNodeType == "GatherND") {
            p.batch_dims = get_node_single_int_attribute_by_name(this->onnxNode, "batch_dims", 0);
        } else {
            p.batch_dims = 0;
        }
        for (int i = 0; i < 2; i++) {
            TensorDesc *desc;
            switch (i) {
                case 0:
                    desc = &(p.data_desc);
                    break;
                case 1:
                    desc = &(p.index_desc);
                    break;
                default:
                    break;
            }
            const std::string &input_name = this->onnxNode.input(i);
            if (onnxWeights.find(input_name) == onnxWeights.end()) {
                *desc = tensor0d();
            } else {
                const onnx::TensorProto &tp = onnxWeights[input_name];
                TensorDesc tmp = genDescFromTp(tp);
                if (onnxNodeType == "Gather" && i == 1 && tmp.nDims == 0) {
                    p.index_scalar = true;
                }
                int num = get_data_size_from_tensor_proto(tp);
                if (tmp.nDims == 0 && num > 0) {
                    tmp.nDims = 1;
                    tmp.dims[0] = num;
                }
                if (onnxWeightReferCount[input_name] > 1) {
                    *desc = tensor0d();
                } else {
                    *desc = tmp;
                }
            }
        }
        curPs.gather_spec = p;
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

        if (this->onnxNode.input_size() == 1) {
            startsInfo = get_node_vector_ints_attribute_by_name(this->onnxNode, "starts");
            endsInfo = get_node_vector_ints_attribute_by_name(this->onnxNode, "ends");
            axesInfo = get_node_vector_ints_attribute_by_name(this->onnxNode, "axes");
        } else {
            startsInfo = get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(1)]);
            endsInfo = get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(2)]);
            if (this->onnxNode.input_size() >= 4) {
                axesInfo = get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(3)]);
            }
            if (this->onnxNode.input_size() >= 5) {
                stepInfo = get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(4)]);
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
        std::vector<int> splitInfo = get_node_vector_ints_attribute_by_name(this->onnxNode, "split");
        slice_ps.axis = get_node_single_int_attribute_by_name(this->onnxNode, "axis", 0);
        // Split equally by default. Set all slice_points to 0
        if (0 == splitInfo.size()) {
            slice_ps.slice_size = (int)this->onnxNode.output_size();
            memset(slice_ps.slice_points, 0, slice_ps.slice_size * sizeof(I32));
        } else {
            slice_ps.slice_size = splitInfo.size();
            slice_ps.slice_points[0] = splitInfo[0];
            for (U32 i = 1; i < slice_ps.slice_size; i++) {
                slice_ps.slice_points[i] = slice_ps.slice_points[i - 1] + splitInfo[i];
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
        std::string embed_weight_name = this->onnxNode.input(0);
        if (onnxWeights.find(this->onnxNode.input(0)) == onnxWeights.end()) {
            return curPs;
        }
        auto tensor_proto = onnxWeights[embed_weight_name];
        int size_of_dims = tensor_proto.dims_size();
        if (size_of_dims != 2) {
            UNI_ERROR_LOG("can not process operator name:%s type:%s attributes.\n",
                this->onnxNode.name().c_str(), this->onnxNode.op_type().c_str());
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
        std::vector<int> squeezeAxes;
        if (this->onnxNode.input_size() > 1) {
            squeezeAxes = get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(1)]);
        } else {
            squeezeAxes = get_node_vector_ints_attribute_by_name(this->onnxNode, "axes");
        }
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
        std::vector<int> unsqueezeAxes;
        if (this->onnxNode.input_size() > 1) {
            unsqueezeAxes = get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(1)]);
        } else {
            unsqueezeAxes = get_node_vector_ints_attribute_by_name(this->onnxNode, "axes");
        }
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
        if (this->onnxNode.input_size() == 2 &&
            onnxWeights.find(this->onnxNode.input(1)) != onnxWeights.end()) {
            cast_to = (get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(1)]))[0];
        } else {
            cast_to = get_node_single_int_attribute_by_name(this->onnxNode, "to", 0);
        }

        if (cast_to == onnx::TensorProto::FLOAT) {
            castPs.targetDt = DT_F32;
        } else if (cast_to == onnx::TensorProto::FLOAT16) {
            castPs.targetDt = DT_F16;
        } else if (cast_to == onnx::TensorProto::INT16 || cast_to == onnx::TensorProto::INT32 ||
            cast_to == onnx::TensorProto::INT64) {
            castPs.targetDt = DT_I32;
        } else if (cast_to == onnx::TensorProto::BOOL) {
            castPs.targetDt = DT_U8;
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
        concatPs.axis = get_node_single_int_attribute_by_name(this->onnxNode, "axis", 1);
        curPs.concat_spec = concatPs;

        insert_shared_weight();
        return curPs;
    }

    ParameterSpec adapt_Softmax() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        SoftmaxParamSpec softmaxPs;
        memset(&softmaxPs, 0, sizeof(softmaxPs));
        softmaxPs.axis = get_node_single_int_attribute_by_name(this->onnxNode, "axis", -1);
        curPs.softmax_spec = softmaxPs;
        return curPs;
    }

    ParameterSpec adapt_Relu() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ReLUParamSpec reluPs;
        memset(&reluPs, 0, sizeof(reluPs));
        reluPs.neg_slope = get_node_float_attribute_by_name(this->onnxNode, "alpha", 0.0);
        curPs.relu_spec = reluPs;
        return curPs;
    }

    ParameterSpec adapt_RNN() override
    {
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Scan") {
            return adapt_Scan();
        }
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        RNNParamSpec rnnPs;
        memset(&rnnPs, 0, sizeof(rnnPs));
        if (onnxNodeType == "RNN") {
            rnnPs.mode = RNN_RNN;
        } else if (onnxNodeType == "LSTM") {
            rnnPs.mode = RNN_LSTM;
        } else if (onnxNodeType == "GRU") {
            int linear_before_reset =
                get_node_single_int_attribute_by_name(this->onnxNode, "linear_before_reset", 0);
            if (linear_before_reset == 0) {
                rnnPs.mode = RNN_GRU;
            } else {
                rnnPs.mode = RNN_GRU_LBR;
            }
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to RNN.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        rnnPs.numOutput = get_node_single_int_attribute_by_name(this->onnxNode, "hidden_size", 1);
        rnnPs.biDirection = get_node_str_attribute_by_name(
                                this->onnxNode, "direction", "forward") == "bidirectional"
            ? true
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
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Pow" || onnxNodeType == "Mul" || onnxNodeType == "Div" ||
            onnxNodeType == "Add" || onnxNodeType == "Sub") {
            std::vector<int> indexes = getOperatorWeightInputIndex(this->onnxNode);
            CHECK_REQUIREMENT(indexes.size() == 1);
            index = indexes[0];
            const onnx::TensorProto &tp = onnxWeights[this->onnxNode.input(index)];
            value = getSinFloat_from_tensorProto(tp);
        }
        if (onnxNodeType == "Pow") {
            powerPs.power = value;
        } else if (onnxNodeType == "Mul") {
            powerPs.scale = value;
        } else if (onnxNodeType == "Div") {
            powerPs.scale = 1 / value;
            if (index == 0) {
                powerPs.power = -1;
            }
        } else if (onnxNodeType == "Add") {
            powerPs.shift = value;
        } else if (onnxNodeType == "Sub") {
            if (index == 0) {
                powerPs.scale = -1;
                powerPs.shift = value;
            } else {
                powerPs.shift = -1 * value;
            }
        } else if (onnxNodeType == "Sqrt") {
            powerPs.power = 0.5;
        } else if (onnxNodeType == "Scale") {
            powerPs.scale = get_node_float_attribute_by_name(this->onnxNode, "scale", 1.0);
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Power.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        curPs.power_spec = powerPs;
        return curPs;
    }

    ParameterSpec adapt_Scale() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ScaleParamSpec scale_ps;
        memset(&scale_ps, 0, sizeof(scale_ps));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "Add" || onnxNodeType == "Sub" || onnxNodeType == "Mul" ||
            onnxNodeType == "Div") {
            const auto &tensor = onnxWeights[this->onnxNode.input(1)];
            if (tensor.dims_size() > 1) {
                for (int idx = 0; idx < tensor.dims_size(); ++idx) {
                    if (tensor.dims(idx) > 1) {
                        scale_ps.axis = idx;
                        break;
                    }
                }
            } else {
                scale_ps.axis = -1;
            }
        } else {
            scale_ps.axis = 1;
        }
        curPs.scale_spec = scale_ps;
        return curPs;
    }

    ParameterSpec adapt_Space2Depth() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        Space2DepthParamSpec s2dPs;
        memset(&s2dPs, 0, sizeof(s2dPs));
        s2dPs.blockSize = get_node_single_int_attribute_by_name(this->onnxNode, "blocksize", 1);
        curPs.space2depth_spec = s2dPs;
        return curPs;
    }

    ParameterSpec adapt_Depth2Space() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        Depth2SpaceParamSpec d2sPs;
        memset(&d2sPs, 0, sizeof(d2sPs));
        d2sPs.blockSize = get_node_single_int_attribute_by_name(this->onnxNode, "blocksize", 1);
        std::string d2s_mode = get_node_str_attribute_by_name(this->onnxNode, "mode", "DCR");
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
        std::vector<int> axesInfo = get_node_vector_ints_attribute_by_name(this->onnxNode, "axes");
        if (axesInfo.size() == 0 && this->onnxNode.input_size() > 1) {
            axesInfo = get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(1)]);
        }
        int keepdimsInfo = get_node_single_int_attribute_by_name(this->onnxNode, "keepdims", 1);
        rsPs.axes_num = axesInfo.size();
        for (int i = 0; i < rsPs.axes_num; i++) {
            rsPs.axes[i] = axesInfo[i];
        }
        rsPs.keep_dim = keepdimsInfo == 0 ? false : true;
        rsPs.coeff = 1.0;
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "ReduceSum") {
            rsPs.reduction_mode = REDUCTION_SUM;
        } else if (onnxNodeType == "ReduceMean") {
            rsPs.reduction_mode = REDUCTION_MEAN;
        } else if (onnxNodeType == "ReduceMax") {
            rsPs.reduction_mode = REDUCTION_MAX;
        } else if (onnxNodeType == "ReduceMin") {
            rsPs.reduction_mode = REDUCTION_MIN;
        } else if (onnxNodeType == "ReduceL2") {
            rsPs.reduction_mode = REDUCTION_L2;
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Reduction.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
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
        amPs.axis = get_node_single_int_attribute_by_name(this->onnxNode, "axis", -1);
        curPs.argmax_spec = amPs;
        return curPs;
    }

    ParameterSpec adapt_PRelu() override
    {
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
        std::vector<int> tileInfo =
            get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(1)]);
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (tileInfo.size() > 0 && tileInfo.size() <= 8) {
            tilePs.dimsSize = tileInfo.size();
        } else {
            UNI_ERROR_LOG("can not process operator name:%s type:%s attributes.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        }
        for (U32 i = 0; i < tileInfo.size(); i++) {
            tilePs.repeatsInfo[i] = tileInfo[i];
        }
        curPs.tile_spec = tilePs;
        return curPs;
    }

    ParameterSpec adapt_Splice() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        SpliceParamSpec splicePs;
        memset(&splicePs, 0, sizeof(splicePs));
        std::vector<int> context = get_node_vector_ints_attribute_by_name(this->onnxNode, "context");
        std::vector<int> indexes =
            get_node_vector_ints_attribute_by_name(this->onnxNode, "forward_indexes");
        splicePs.num_context = context.size();
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (splicePs.num_context == 0) {
            UNI_ERROR_LOG("can not process operator name:%s type:%s attributes.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
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
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        TdnnParamSpec tdnnPs;
        memset(&tdnnPs, 0, sizeof(tdnnPs));
        const onnx::TensorProto &context = onnxWeights[this->onnxNode.input(1)];
        const onnx::TensorProto &params = onnxWeights[this->onnxNode.input(2)];
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
        p.axis = get_node_single_int_attribute_by_name(this->onnxNode, "axis", -1);
        p.largest = get_node_single_int_attribute_by_name(this->onnxNode, "largest", 1);
        p.sorted = get_node_single_int_attribute_by_name(this->onnxNode, "sorted", 1);
        if (this->onnxNode.input_size() == 1) {
            p.topk = get_node_single_int_attribute_by_name(this->onnxNode, "k", 1);
        } else {
            p.topk = get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(1)])[0];
        }
        curPs.topk_spec = p;
        return curPs;
    }

    ParameterSpec adapt_Where() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        WhereParamSpec wherePs;
        memset(&wherePs, 0, sizeof(wherePs));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxWeights.find(this->onnxNode.input(0)) == onnxWeights.end()) {
            UNI_WARNING_LOG("not find condition initializer in operator name:%s type:%s "
                            "attributes.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        } else {
            const onnx::TensorProto &conditionTp = onnxWeights[this->onnxNode.input(0)];
            wherePs.conditionDesc = genDescFromTp(conditionTp);
        }

        if (onnxWeights.find(this->onnxNode.input(2)) == onnxWeights.end()) {
            UNI_WARNING_LOG("not find y initializer in operator name:%s type:%s attributes.\n",
                this->onnxNode.name().c_str(), onnxNodeType.c_str());
        } else {
            const onnx::TensorProto &yTp = onnxWeights[this->onnxNode.input(2)];
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
        for (int i = 0; i < this->onnxNode.attribute_size(); i++) {
            const onnx::AttributeProto &attribute = this->onnxNode.attribute(i);
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
                    if (onnxWeights.find(curNode.input(j)) != onnxWeights.end()) {
                        auto hidWeightTp = onnxWeights[curNode.input(j)];
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

    ParameterSpec adapt_Expand() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ExpandParamSpec expandPs;
        memset(&expandPs, 0, sizeof(expandPs));
        std::vector<int> expandInfo;
        if (this->onnxNode.input_size() == 1) {
            expandInfo = get_node_vector_ints_attribute_by_name(this->onnxNode, "shape");
        } else {
            expandInfo = get_int_vec_from_tensorProto(onnxWeights[this->onnxNode.input(1)]);
        }
        expandPs.shape_size = expandInfo.size();
        memcpy(expandPs.shape_dims, expandInfo.data(), expandPs.shape_size * sizeof(I32));
        curPs.expand_spec = expandPs;
        return curPs;
    }

    ParameterSpec adapt_Scatter() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ScatterParamSpec p;
        memset(&p, 0, sizeof(p));
        const std::string &onnxNodeType = this->onnxNode.op_type();
        if (onnxNodeType == "ScatterElements") {
            p.axis = get_node_single_int_attribute_by_name(this->onnxNode, "axis", 0);
        } else {
            p.axis = INT_MAX;
        }
        for (int i = 0; i < 3; i++) {
            TensorDesc *desc;
            switch (i) {
                case 0:
                    desc = &(p.data_desc);
                    break;
                case 1:
                    desc = &(p.index_desc);
                    break;
                case 2:
                    desc = &(p.update_desc);
                    break;
                default:
                    break;
            }
            if (onnxWeights.find(this->onnxNode.input(i)) == onnxWeights.end()) {
                *desc = tensor0d();
            } else {
                const onnx::TensorProto &tp = onnxWeights[this->onnxNode.input(i)];
                *desc = genDescFromTp(tp);
            }
        }
        curPs.scatter_spec = p;
        return curPs;
    }

    ParameterSpec adapt_RoIAlign() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        RoIAlignParamSpec rps;
        memset(&rps, 0, sizeof(rps));
        std::string coordinateTransformationMode = get_node_str_attribute_by_name(
            this->onnxNode, "coordinate_transformation_mode", "NO_SET");
        if (coordinateTransformationMode == "NO_SET") {
            int aligned = get_node_single_int_attribute_by_name(this->onnxNode, "aligned", 1);
            if (aligned <= 0) {
                coordinateTransformationMode = "output_half_pixel";
            } else {
                coordinateTransformationMode = "half_pixel";
            }
        }
        if (coordinateTransformationMode == "half_pixel") {
            rps.coordinateTransformationMode = ROIALIGN_HALF_PIXEL;
        } else if (coordinateTransformationMode == "output_half_pixel") {
            rps.coordinateTransformationMode = ROIALIGN_OUTPUT_HALF_PIXEL;
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }

        std::string poolingMode = get_node_str_attribute_by_name(this->onnxNode, "mode", "avg");
        if (poolingMode == "avg") {
            rps.mode = POOLING_MEAN;
        } else if (poolingMode == "max") {
            rps.mode = POOLING_MAX;
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        rps.output_w = get_node_single_int_attribute_by_name(this->onnxNode, "pooled_w", 1);
        rps.output_h = get_node_single_int_attribute_by_name(this->onnxNode, "pooled_h", 1);
        if (rps.output_w == 1) {
            rps.output_w = get_node_single_int_attribute_by_name(this->onnxNode, "output_width", 1);
        }
        if (rps.output_h == 1) {
            rps.output_h = get_node_single_int_attribute_by_name(this->onnxNode, "output_height", 1);
        }
        rps.sampling_ratio =
            get_node_single_int_attribute_by_name(this->onnxNode, "sampling_ratio", 0);
        rps.spatial_scale = get_node_float_attribute_by_name(this->onnxNode, "spatial_scale", 1.0);
        curPs.roialign_spec = rps;
        return curPs;
    }

    ParameterSpec adapt_GenerateProposals() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        GenerateProposalsParamSpec gpps;
        memset(&gpps, 0, sizeof(gpps));
        gpps.angle_bound_hi =
            get_node_single_int_attribute_by_name(this->onnxNode, "angle_bound_hi", 0);
        gpps.angle_bound_lo =
            get_node_single_int_attribute_by_name(this->onnxNode, "angle_bound_lo", 0);
        gpps.angle_bound_on =
            get_node_single_int_attribute_by_name(this->onnxNode, "angle_bound_on", 0);
        gpps.clip_angle_thresh =
            get_node_float_attribute_by_name(this->onnxNode, "clip_angle_thresh", 0.0);
        gpps.legacy_plus_one =
            get_node_single_int_attribute_by_name(this->onnxNode, "legacy_plus_one", 0);
        gpps.min_size = get_node_float_attribute_by_name(this->onnxNode, "min_size", 0.0);
        gpps.nms_thresh = get_node_float_attribute_by_name(this->onnxNode, "nms_thresh", 0.0);
        gpps.post_nms_topN =
            get_node_single_int_attribute_by_name(this->onnxNode, "post_nms_topN", 0);
        gpps.pre_nms_topN = get_node_single_int_attribute_by_name(this->onnxNode, "pre_nms_topN", 0);
        gpps.spatial_scale = get_node_float_attribute_by_name(this->onnxNode, "spatial_scale", 0.0);
        curPs.generate_proposals_spec = gpps;
        return curPs;
    }

private:
    int removePreprocessOpNum;
    bool useBNN;

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
