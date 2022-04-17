// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSORFLOWADAPTEE
#define _H_TENSORFLOWADAPTEE
#include <json/json.h>
#include <sstream>

#include "model_adaptee.h"

class TensorflowAdaptee : public ModelAdaptee {
public:
    TensorflowAdaptee()
    {
        this->modelInputLayerNum = 0;
        this->entityOpCount = 0;
        this->weightNumber = 0;
        this->curInDegree = 0;
    }

    ~TensorflowAdaptee()
    {}

protected:
    std::string cleanRedundantC(std::string str)
    {
        std::string result = "";
        for (int i = 1; i < (int)(str.length() - 1); i++) {
            if (str.at(i) == '\\') {
                continue;
            } else {
                result += str.at(i);
            }
        }
        return result;
    }

    OperatorType convert_tensorflow_type(std::string tfType)
    {
        std::map<std::string, OperatorType> operatorMap = {
            {"FusedBatchNorm", OT_BatchNorm},
            {"Relu6", OT_Relu6},
            {"DepthwiseConv2dNative", OT_Conv},
            {"MaxPool", OT_Pooling},
            {"ConcatV2", OT_Concat},
            {"Relu", OT_Relu},
            {"ResizeBilinear", OT_Resize},
            {"ArgMax", OT_ArgMax},
            {"ExpandDims", OT_Unsqueeze},
            {"Pad", OT_Pad},
            {"PadV2", OT_Pad},
            {"Transpose", OT_Transpose},
            {"BiasAdd", OT_FC},
            {"Conv2DBackpropInput", OT_Conv},
            {"Conv2D", OT_Conv},
            {"Cast", OT_Cast},
            {"Reshape", OT_Reshape},
            {"Rsqrt", OT_Power},
            {"Squeeze", OT_Squeeze},
            {"Sigmoid", OT_Sigmoid},
            {"Softmax", OT_Softmax},
            {"AvgPool", OT_Pooling},
            {"Mean", OT_Reduction},
            {"Shape", OT_Shape},
        };
        if (operatorMap.find(tfType) != operatorMap.end()) {
            return operatorMap[tfType];
        }
        if (tfType == "Mul" || tfType == "Sub" || tfType == "Add" || tfType == "RealDiv") {
            if (curInDegree == 1) {
                return OT_Power;
            } else {
                return OT_Eltwise;
            }
        } else if (tfType == "MatMul") {
            if (this->curInDegree == 1) {
                return OT_FC;
            } else {
                return OT_MatMul;
            }
        } else {
            UNI_ERROR_LOG("operator name:%s type:%s not supported.\n", this->layerName.c_str(),
                tfType.c_str());
            return OT_None;
        }
    }

    EE parse_file(std::string dir, std::string mfn) override
    {
        EE ret;
        std::string tfSuffix = ".json";
        this->modelName = mfn;
        std::string modelAbsPath = dir + "/" + mfn + tfSuffix;
        std::string::size_type idx;
        std::ifstream inFile;
        inFile.open(modelAbsPath);
        if (!inFile.is_open()) {
            UNI_ERROR_LOG("can not open tensorflow model file %s.\n", modelAbsPath.c_str());
        }
        std::stringstream strStream;
        strStream << inFile.rdbuf();
        std::string strValueFromPy = strStream.str();
        std::string strValue = cleanRedundantC(strValueFromPy);
        std::string tailStr = strValue.substr(strValue.length() - 18, 18);
        newStrValue = "";
        idx = tailStr.find("library");
        if (idx == std::string::npos) {
            newStrValue = strValue;
        } else {
            newStrValue = strValue.substr(0, strValue.length() - 16) + "}";
        }
        Json::Reader reader;
        Json::Value value;
        if (reader.parse(newStrValue, value)) {
            this->entityOpCount = value["node"].size();
            for (int i = 0; i < (int)(value["node"].size()); i++) {
                if ((value["node"][i]["op"].asString()).compare("Const") == 0) {
                    constId[value["node"][i]["name"].asString()] = i;
                    this->entityOpCount = this->entityOpCount - 1;
                } else if ((value["node"][i]["op"].asString()).compare("Identity") == 0) {
                    idenConst[value["node"][i]["name"].asString()] =
                        value["node"][i]["input"][0].asString();
                    this->entityOpCount = this->entityOpCount - 1;
                } else if ((value["node"][i]["op"].asString()).compare("Placeholder") == 0) {
                    this->modelInputLayerNum = this->modelInputLayerNum + 1;
                    this->entityOpCount = this->entityOpCount - 1;
                }
            }
            ret = SUCCESS;
        } else {
            ret = FILE_ERROR;
        }
        return ret;
    }

    template <typename T>
    void shiftRight(T *array, int length, int left, int right)
    {
        // only transpose 4-dim parameter
        if (length >= 4) {
            T data = array[right];
            for (int i = right; i > left; i--) {
                array[i] = array[i - 1];
            }
            array[left] = data;
        }
    }

    EE adapt_operators(ModelSpec *ms) override
    {
        EE ret = SUCCESS;
        ms->dt = DT_F32;
        str_copy(ms->model_name, modelName.c_str(), modelName.length());
        ms->num_inputs = this->modelInputLayerNum;
        ms->input_names = (I8 **)mt_malloc(ms->num_inputs * sizeof(I8 *));
        ms->input_dims = (TensorDesc *)mt_malloc(sizeof(TensorDesc) * ms->num_inputs);
        int traverseInputLayerIndex = 0;

        ms->num_operator_specs = this->entityOpCount;
        OperatorSpec *opsPtr =
            (OperatorSpec *)mt_malloc(sizeof(OperatorSpec) * ms->num_operator_specs);
        ms->ops = opsPtr;
        int traverseEntityOpIndex = 0;

        std::map<std::string, int> unmapOps;
        Json::Reader reader;
        Json::Value value;
        if (reader.parse(newStrValue, value)) {
            this->ttValue = value;
            for (int i = 0; i < (int)(value["node"].size()); i++) {
                layerName = value["node"][i]["name"].asString();
                UNI_DEBUG_LOG("process operator name:%s parameter.\n", layerName.c_str());
                this->opType = value["node"][i]["op"].asString();
                if (opType.compare("Placeholder") == 0) {
                    ms->input_names[traverseInputLayerIndex] =
                        (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
                    str_copy(ms->input_names[traverseInputLayerIndex], layerName.c_str(),
                        layerName.length());
                    int placeholder_shape_size =
                        value["node"][i]["attr"]["shape"]["shape"]["dim"].size();
                    if (placeholder_shape_size == 0) {
                        UNI_ERROR_LOG("model input %s dimensions are not specific.\n",
                            ms->input_names[traverseInputLayerIndex]);
                    }
                    std::vector<int> inputShape(placeholder_shape_size);
                    for (int j = 0; j < placeholder_shape_size; j++) {
                        inputShape[j] = std::stoi(
                            value["node"][i]["attr"]["shape"]["shape"]["dim"][j]["size"].asString());
                    }
                    shiftRight<int>(inputShape.data(), inputShape.size(), 1, inputShape.size() - 1);
                    ms->input_dims[traverseInputLayerIndex].nDims = inputShape.size();
                    ms->input_dims[traverseInputLayerIndex].dt = DT_F32;
                    ms->input_dims[traverseInputLayerIndex].df =
                        getTensorDefaultDataFormat(ms->input_dims[traverseInputLayerIndex].nDims);
                    for (U32 j = 0; j < ms->input_dims[traverseInputLayerIndex].nDims; j++) {
                        ms->input_dims[traverseInputLayerIndex]
                            .dims[ms->input_dims[traverseInputLayerIndex].nDims - 1 - j] =
                            inputShape[j];
                    }
                    traverseInputLayerIndex++;
                } else if (opType.compare("Const") == 0) {
                    auto shape = value["node"][i]["attr"]["value"]["tensor"]["tensorShape"]["dim"];
                    int tensorDimSize = shape.size();
                    std::vector<int> tensorDims;
                    int tensorDimsNum = 1;
                    for (int j = 0; j < tensorDimSize; j++) {
                        tensorDims.push_back(std::stoi(shape[j]["size"].asString()));
                        tensorDimsNum *= tensorDims[j];
                    }
                } else if (opType.compare("Identity") != 0) {
                    std::vector<std::string> inList;
                    std::vector<std::string> constList;

                    this->node = value["node"][i];
                    ParameterSpec tmpPs;

                    str_copy(
                        opsPtr[traverseEntityOpIndex].name, layerName.c_str(), layerName.length());

                    if (opType.compare("Conv2DBackpropInput") == 0) {
                        UNI_WARNING_LOG("Filter the input0_size\n");
                    } else if (opType.compare("FusedBatchNorm") == 0 &&
                        value["node"][i]["input"].size() != 5) {  // To collect more special cases
                        constList.push_back(value["node"][i]["input"][0].asString());
                    } else {
                        inList.push_back(value["node"][i]["input"][0].asString());
                    }
                    for (int k = 1; k < (int)(value["node"][i]["input"].size()); k++) {
                        std::string curIn = value["node"][i]["input"][k].asString();
                        if (idenConst.find(curIn) == idenConst.end() &&
                            constId.find(curIn) == constId.end()) {
                            inList.push_back(curIn);
                        } else {
                            if (constId.find(idenConst[curIn]) == constId.end() &&
                                constId.find(curIn) == constId.end()) {
                                inList.push_back(curIn);
                            } else {
                                constList.push_back(curIn);
                            }
                        }
                    }

                    if (constList.size() > 0) {
                        weightConstInput[layerName] = constList;

                        if (opType != "Mul" && opType != "Sub" && opType != "Add" &&
                            opType != "RealDiv" && opType != "ConcatV2" && opType != "PadV2" &&
                            opType != "ArgMax" && opType != "Transpose" && opType != "Pad" &&
                            opType != "ExpandDims" && opType != "ResizeBilinear" &&
                            opType != "Reshape" && opType != "Mean") {  // TODO: expand more cases
                            weightIds.push_back(i);
                            this->weightNumber = this->weightNumber + 1;
                        }
                    }

                    opsPtr[traverseEntityOpIndex].num_inputs = inList.size();
                    opsPtr[traverseEntityOpIndex].input_tensors_name =
                        (I8 **)mt_malloc(opsPtr[traverseEntityOpIndex].num_inputs * sizeof(I8 *));
                    for (int k = 0; k < (int)(opsPtr[traverseEntityOpIndex].num_inputs); k++) {
                        opsPtr[traverseEntityOpIndex].input_tensors_name[k] =
                            (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
                        str_copy(opsPtr[traverseEntityOpIndex].input_tensors_name[k],
                            inList[k].c_str(), inList[k].length());
                    }
                    opsPtr[traverseEntityOpIndex].num_outputs = 1;
                    opsPtr[traverseEntityOpIndex].output_tensors_name =
                        (I8 **)mt_malloc(opsPtr[traverseEntityOpIndex].num_outputs * sizeof(I8 *));
                    for (int k = 0; k < (int)(opsPtr[traverseEntityOpIndex].num_outputs); k++) {
                        opsPtr[traverseEntityOpIndex].output_tensors_name[k] =
                            (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
                        str_copy(opsPtr[traverseEntityOpIndex].output_tensors_name[k],
                            layerName.c_str(), layerName.length());
                    }
                    opsPtr[traverseEntityOpIndex].tensor_positions = nullptr;
                    opsPtr[traverseEntityOpIndex].num_quant_feature = 0;
                    opsPtr[traverseEntityOpIndex].feature_scale = nullptr;

                    this->curInDegree = inList.size();
                    OperatorType curOpType = convert_tensorflow_type(opType);
                    opsPtr[traverseEntityOpIndex].type = curOpType;
                    CHECK_STATUS(adapt_operator(curOpType, &tmpPs));
                    opsPtr[traverseEntityOpIndex].ps = tmpPs;

                    traverseEntityOpIndex++;
                }
            }
        } else {
            UNI_ERROR_LOG("can not read tensorflow model.\n");
        }
        return ret;
    }

    std::vector<int> get_weight_ids(const Json::Value &node)
    {
        std::vector<int> ids;
        std::string name = node["name"].asString();
        std::vector<std::string> all = this->weightConstInput[name];
        for (U32 i = 0; i < all.size(); i++) {
            int id;
            if (constId.find(all[i]) != constId.end()) {
                id = constId[all[i]];
            } else {
                id = constId[idenConst[all[i]]];
            }
            ids.push_back(id);
        }
        return ids;
    }

    int get_length(const Json::Value &tensor)
    {
        auto data = tensor["attr"]["value"]["tensor"]["tensorContent"];
        return data.size();
    }

    std::vector<float> get_floats(const Json::Value &tensor)
    {
        auto data = tensor["attr"]["value"]["tensor"]["tensorContent"];
        int size = data.size();
        std::vector<float> ret(size);
        for (int i = 0; i < size; i++) {
            ret[i] = std::stof(data[i].asString());
        }
        return ret;
    }

    std::vector<int> get_shape(const Json::Value &tensor)
    {
        auto shape = tensor["attr"]["value"]["tensor"]["tensorShape"]["dim"];
        int size = shape.size();
        std::vector<int> ret(size);
        for (int i = 0; i < size; i++) {
            ret[i] = std::stoi(shape[i].asString());
        }
        return ret;
    }

    std::vector<int> get_ints(const Json::Value &tensor)
    {
        auto data = tensor["attr"]["value"]["tensor"]["tensorContent"];
        int size = data.size();
        std::vector<int> ret(size);
        for (int i = 0; i < size; i++) {
            ret[i] = std::stoi(data[i].asString());
        }
        return ret;
    }

    std::vector<int> get_ints(const Json::Value &node, const char *attributeName)
    {
        auto attribute = node["attr"][attributeName]["list"]["i"];
        int size = attribute.size();
        std::vector<int> ret(size);
        for (int i = 0; i < size; i++) {
            ret[i] = std::stoi(attribute[i].asString());
        }
        return ret;
    }

    void copy_tensors(std::vector<Json::Value> tensors, U8 *ptr)
    {
        for (U32 i = 0; i < tensors.size(); i++) {
            std::vector<float> data = get_floats(tensors[i]);
            int bytes = sizeof(float) * data.size();
            UNI_MEMCPY(ptr, data.data(), bytes);
            ptr += bytes;
        }
    }

    WeightSpec convert_weight(
        std::string operatorName, std::vector<Json::Value> weight, std::vector<Json::Value> bias)
    {
        DataType wdt = DT_F32;
        U32 bytes0 = 0, bytes1 = 0;
        for (U32 i = 0; i < weight.size(); i++) {
            bytes0 += get_length(weight[i]) * bytesOf(wdt);
        }
        for (U32 i = 0; i < bias.size(); i++) {
            bytes1 += get_length(bias[i]) * bytesOf(wdt);
        }
        WeightSpec w = mt_create_weight(operatorName.c_str(), wdt, bytes0, bytes1, 0);
        copy_tensors(weight, w.weight);
        copy_tensors(bias, w.vec);
        return w;
    }

    EE adapt_weights(ModelSpec *ms) override
    {
        ms->num_weight_specs = weightNumber;
        WeightSpec *ws = (WeightSpec *)mt_malloc(sizeof(WeightSpec) * ms->num_weight_specs);
        ms->ws = ws;
        Json::Reader reader;
        Json::Value value;
        if (!reader.parse(newStrValue, value)) {
            return NOT_SUPPORTED;
        }
        for (int j = 0; j < ms->num_weight_specs; j++) {
            this->node = value["node"][weightIds[j]];
            std::string name = this->node["name"].asString();
            UNI_DEBUG_LOG("process operator name:%s weight.\n", name.c_str());
            std::vector<int> ids = get_weight_ids(this->node);
            std::string type = this->node["op"].asString();
            std::vector<std::string> constList;
            if (type == "Conv2D" || type == "Conv2DBackpropInput" || type == "MatMul" ||
                type == "DepthwiseConv2dNative") {
                ws[j] = convert_weight(name, {value["node"][ids[0]]}, {});
            } else if (type.compare("BiasAdd") == 0) {
                ws[j] = convert_weight(name, {}, {value["node"][ids[0]]});
            } else if (type.compare("FusedBatchNorm") == 0) {
                U32 bytes = get_length(value["node"][ids[0]]) * sizeof(float);
                ws[j] = mt_create_weight(name.c_str(), DT_F32, bytes, bytes, 0);
                std::vector<float> scale = get_floats(value["node"][ids[0]]);
                std::vector<float> bias = get_floats(value["node"][ids[1]]);
                std::vector<float> mean = get_floats(value["node"][ids[2]]);
                std::vector<float> var = get_floats(value["node"][ids[3]]);
                for (U32 i = 0; i < scale.size(); i++) {
                    float a = mean[i] - bias[i] * sqrt(var[i] / powf(scale[i], 2));
                    float b = var[i] / (powf(scale[i], 2));
                    mean[i] = a;
                    var[i] = b;
                }
                UNI_MEMCPY(ws[j].weight, mean.data(), mean.size() * sizeof(float));
                UNI_MEMCPY(ws[j].vec, var.data(), var.size() * sizeof(float));
            }
        }
        return SUCCESS;
    }

    ParameterSpec adapt_Eltwise() override
    {
        ParameterSpec ps;
        EltwiseParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        if (opType == "Add") {
            p.mode = ELTWISE_SUM;
        } else if (opType == "Sub") {
            p.mode = ELTWISE_SUB;
        }
        p.activation_type = ACTIVATION_NULL;
        ps.eltwise_spec = p;
        return ps;
    }

    ParameterSpec adapt_ArgMax() override
    {
        ParameterSpec ps;
        ArgMaxParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = 1;
        ps.argmax_spec = p;
        return ps;
    }

    ParameterSpec adapt_Conv() override
    {
        ParameterSpec ps;
        ConvolutionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.kernel_t = 1;
        p.stride_t = 1;
        p.pad_before = 0;
        p.pad_after = 0;
        p.dilatedRate_t = 1;

        std::vector<int> dilations(4, 1);
        if (opType.compare("DepthwiseConv2dNative") == 0) {
            dilations = get_ints(this->node, "dilations");
        }
        std::vector<int> strides = get_ints(this->node, "strides");
        ;
        p.dilatedRate_h = dilations[1];
        p.dilatedRate_w = dilations[2];
        p.stride_h = strides[1];
        p.stride_w = strides[2];

        int id = get_weight_ids(this->node)[0];
        std::vector<int> kernels = get_shape(this->ttValue["node"][id]);
        if (kernels.size() < 4) {
            UNI_ERROR_LOG("can not process operator name:%s kernel.\n", this->layerName.c_str());
        }
        if (opType.compare("DepthwiseConv2dNative") == 0) {
            p.num_outputs = kernels[2];
        } else {
            p.num_outputs = kernels[3];
        }
        p.kernel_h = kernels[0];
        p.kernel_w = kernels[1];

        // choose one of VALID/SAME
        std::string tfPaddingMode = this->node["attr"]["padding"]["s"].asString();
        if (tfPaddingMode.at(0) == 'V') {
            tfPaddingMode = "VALID";
            p.pad_top = 0;
            p.pad_bottom = 0;
            p.pad_left = 0;
            p.pad_right = 0;
        } else {
            tfPaddingMode = "SAME";
            p.pad_top = (U32)INT_MAX;
            p.pad_bottom = (U32)INT_MAX;
            p.pad_left = (U32)INT_MAX;
            p.pad_right = (U32)INT_MAX;
        }

        p.group = 1;
        p.dw_activation_type = ACTIVATION_NULL;
        p.pw_activation_type = ACTIVATION_NULL;

        if (p.group != 1 && p.group == p.num_outputs) {
            p.convolution_type = CONVOLUTION_DEPTHWISE;
        } else {
            p.convolution_type = CONVOLUTION_POINTWISE;
        }

        if (opType.compare("DepthwiseConv2dNative") == 0) {
            p.convolution_type = CONVOLUTION_DEPTHWISE;
        }
        ps.conv_spec = p;
        return ps;
    }

    ParameterSpec adapt_BatchNorm() override
    {
        ParameterSpec ps;
        BatchNormParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = 0;
        p.eps = this->node["attr"]["epsilon"]["f"].asFloat();
        p.gama = 0;
        p.momentum = 0;
        ps.bn_spec = p;
        return ps;
    }

    ParameterSpec adapt_Fc() override
    {
        ParameterSpec ps;
        FullyConnectedParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        int id = get_weight_ids(this->node)[0];
        std::vector<int> kernels = get_shape(this->ttValue["node"][id]);
        p.num_outputs = kernels[kernels.size() - 1];
        p.num_slices = 1;
        ps.fc_spec = p;
        return ps;
    }

    ParameterSpec adapt_Pooling() override
    {
        ParameterSpec ps;
        PoolingParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> kernels = get_ints(this->node, "ksize");
        std::vector<int> strides = get_ints(this->node, "strides");
        p.kernel_t = 1;
        p.kernel_h = kernels[1];
        p.kernel_w = kernels[2];
        p.stride_t = 1;
        p.stride_h = strides[1];
        p.stride_w = strides[2];
        p.pad_before = 0;
        p.pad_after = 0;
        p.pad_top = 0;
        p.pad_bottom = 0;
        p.pad_left = 0;
        p.pad_right = 0;
        p.round_mode = ROUND_CEIL;
        if (opType.compare("MaxPool") == 0) {
            p.mode = POOLING_MAX;
        } else {
            p.mode = POOLING_MEAN;
        }
        p.count_include_pad = false;
        ps.pooling_spec = p;
        return ps;
    }

    ParameterSpec adapt_Reduction() override
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        ReductionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        if (opType.compare("Mean") == 0) {
            p.mode = REDUCTION_MEAN;
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Reduction.\n",
                this->layerName.c_str(), opType.c_str());
        }
        int id = get_weight_ids(this->node)[0];
        std::vector<int> dims = get_ints(this->ttValue["node"][id]);
        p.num_axes = dims.size();
        for (int i = 0; i < p.num_axes; i++) {
            p.axes[i] = dims[i];
        }
        ps.reduction_spec = p;
        return ps;
    }

    ParameterSpec adapt_Pad() override
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        PadParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        int id = get_weight_ids(this->node)[0];
        std::vector<int> pad = get_ints(this->ttValue["node"][id]);
        p.before = 0;
        p.after = 0;
        p.top = pad[2];
        p.bottom = pad[3];
        p.left = pad[4];
        p.right = pad[5];
        p.constant_value = 0;
        p.pad_mode = PAD_CONSTANT;
        ps.pad_spec = p;
        return ps;
    }

    ParameterSpec adapt_Concat() override
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        ConcatParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = std::stoi(this->node["attr"]["N"]["i"].asString());
        ps.concat_spec = p;
        return ps;
    }

    ParameterSpec adapt_Resize() override
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        ResizeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));

        int id = get_weight_ids(this->node)[0];
        p.num_sizes = 2;
        std::vector<int> sizes = get_ints(this->ttValue["node"][id]);
        for (U32 i = 0; i < p.num_sizes; i++) {
            p.sizes[i] = sizes[i];
        }
        p.num_scales = 0;
        ps.resize_spec = p;
        return ps;
    }

    ParameterSpec adapt_Power() override
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        PowerParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.scale = 1.0;
        p.shift = 0.0;
        p.power = 1.0;
        if (opType.compare("Rsqrt") == 0) {
            p.power = 0.5;
        } else {
            int id = get_weight_ids(this->node)[0];
            std::vector<float> data = get_floats(this->ttValue["node"][id]);
            if (opType.compare("Mul") == 0) {
                p.scale = data[0];
            } else if (opType.compare("Sub") == 0) {
                p.shift = -1 * data[0];
            } else if (opType.compare("RealDiv") == 0) {
                p.scale = 1 / data[0];
            }
        }
        ps.power_spec = p;
        return ps;
    }

    ParameterSpec adapt_Transpose() override
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        TransposeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        int id = get_weight_ids(this->node)[0];
        std::vector<int> dims = get_ints(this->ttValue["node"][id]);
        p.num_axes = dims.size();
        for (U32 i = 0; i < p.num_axes; i++) {
            p.axes[i] = dims[i];
        }
        ps.transpose_spec = p;
        return ps;
    }

    ParameterSpec adapt_Reshape() override
    {
        ParameterSpec ps;
        ReshapeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        int id = get_weight_ids(this->node)[0];
        std::vector<int> shape = get_ints(this->ttValue["node"][id]);
        p.num_shape = shape.size();
        for (int i = 0; i < p.num_shape; i++) {
            p.shape[i] = shape[i];
        }
        p.axis = 8;
        p.num_axes = -1;
        ps.reshape_spec = p;
        return ps;
    }

    ParameterSpec adapt_Squeeze() override
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        SqueezeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> dims = get_ints(this->node, "squeeze_dims");
        p.num_axes = dims.size();
        for (int i = 0; i < p.num_axes; i++) {
            p.axes[i] = dims[i];
        }
        ps.squeeze_spec = p;
        return ps;
    }

    ParameterSpec adapt_Unsqueeze() override
    {
        ParameterSpec ps;
        UnsqueezeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        int id = get_weight_ids(this->node)[0];
        std::vector<int> dims = get_ints(this->ttValue["node"][id]);
        p.num_axes = dims.size();
        for (int i = 0; i < p.num_axes; i++) {
            p.axes[i] = dims[i];
        }
        ps.unsqueeze_spec = p;
        return ps;
    }

    ParameterSpec adapt_Cast() override
    {
        ParameterSpec ps;
        CastParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.dt = DT_F32;
        ps.cast_spec = p;
        return ps;
    }

private:
    int modelInputLayerNum;
    int entityOpCount;

    std::string modelName;
    std::string newStrValue;
    Json::Value node;
    std::string opType;
    std::string layerName;

    Json::Value ttValue;

    std::map<std::string, int> constId;
    std::map<std::string, std::string> idenConst;
    std::map<std::string, std::vector<std::string>> weightConstInput;
    std::vector<int> weightIds;

    int weightNumber;
    int curInDegree;
};
#endif
