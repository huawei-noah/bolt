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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>

#include "model_adaptee.h"

class TensorflowAdaptee : public ModelAdaptee {
public:
    TensorflowAdaptee()
    {
        this->modelInputLayerNum = 0;
        this->entityOpCount = 0;
        this->weightOpNum = 0;
        this->curInDegree = 0;
        this->curNodeIndex = 0;
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
        if (tfType.compare("Mul") == 0 || tfType.compare("Sub") == 0 ||
            tfType.compare("Add") == 0 || tfType.compare("RealDiv") == 0) {
            if (curInDegree == 1) {
                return OT_Power;
            } else {
                return OT_Eltwise;
            }
        } else if (tfType.compare("FusedBatchNorm") == 0) {
            return OT_BatchNorm;
        } else if (tfType.compare("Relu6") == 0) {
            return OT_Relu6;
        } else if (tfType.compare("DepthwiseConv2dNative") == 0) {
            return OT_Conv;
        } else if (tfType.compare("MaxPool") == 0) {
            return OT_Pooling;
        } else if (tfType.compare("ConcatV2") == 0) {
            return OT_Concat;
        } else if (tfType.compare("Relu") == 0) {
            return OT_Relu;
        } else if (tfType.compare("ResizeBilinear") == 0) {
            return OT_Resize;
        } else if (tfType.compare("ArgMax") == 0) {
            return OT_ArgMax;
        } else if (tfType.compare("ExpandDims") == 0) {
            return OT_Unsqueeze;
        } else if (tfType.compare("Pad") == 0 || tfType.compare("PadV2") == 0) {
            return OT_Pad;
        } else if (tfType.compare("Transpose") == 0) {
            return OT_Transpose;
        } else if (tfType.compare("BiasAdd") == 0) {
            return OT_FC;
        } else if (tfType.compare("Conv2DBackpropInput") == 0 || tfType.compare("Conv2D") == 0) {
            return OT_Conv;
        } else if (tfType.compare("Cast") == 0) {
            return OT_Cast;
        } else if (tfType.compare("Reshape") == 0) {
            return OT_Reshape;
        } else if (tfType.compare("Rsqrt") == 0) {
            return OT_Power;
        } else if (tfType.compare("Squeeze") == 0) {
            return OT_Squeeze;
        } else if (tfType.compare("Sigmoid") == 0) {
            return OT_Sigmoid;
        } else if (tfType.compare("MatMul") == 0) {
            if (this->curInDegree == 1) {
                return OT_FC;
            } else {
                return OT_MatMul;
            }
        } else if (tfType.compare("Softmax") == 0) {
            return OT_Softmax;
        } else if (tfType.compare("AvgPool") == 0) {
            return OT_Pooling;
        } else if (tfType.compare("Mean") == 0) {
            return OT_Reduction;
        } else if (tfType.compare("Shape") == 0) {
            return OT_Shape;
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
        ms->input_names = (I8 **)mt_new_storage(ms->num_inputs * sizeof(I8 *));
        ms->input_dims = (TensorDesc *)mt_new_storage(sizeof(TensorDesc) * ms->num_inputs);
        int traverseInputLayerIndex = 0;

        ms->num_operator_specs = this->entityOpCount;
        OperatorSpec *opsPtr =
            (OperatorSpec *)mt_new_storage(sizeof(OperatorSpec) * ms->num_operator_specs);
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
                        (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
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
                    int tensorDimSize =
                        value["node"][i]["attr"]["value"]["tensor"]["tensorShape"]["dim"].size();
                    std::vector<int> tensorDims;
                    int tensorDimsNum = 1;
                    for (int j = 0; j < tensorDimSize; j++) {
                        tensorDims.push_back(std::stoi(
                            value["node"][i]["attr"]["value"]["tensor"]["tensorShape"]["dim"][j]["s"
                                                                                                 "i"
                                                                                                 "z"
                                                                                                 "e"]
                                .asString()));
                        tensorDimsNum *= tensorDims[j];
                    }
                } else if (opType.compare("Identity") != 0) {
                    std::vector<std::string> inList;
                    std::vector<std::string> constList;

                    this->nodeV = value["node"][i];
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
                            this->weightOpNum = this->weightOpNum + 1;
                        }
                    }

                    opsPtr[traverseEntityOpIndex].num_inputs = inList.size();
                    opsPtr[traverseEntityOpIndex].input_tensors_name = (I8 **)mt_new_storage(
                        opsPtr[traverseEntityOpIndex].num_inputs * sizeof(I8 *));
                    for (int k = 0; k < (int)(opsPtr[traverseEntityOpIndex].num_inputs); k++) {
                        opsPtr[traverseEntityOpIndex].input_tensors_name[k] =
                            (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                        str_copy(opsPtr[traverseEntityOpIndex].input_tensors_name[k],
                            inList[k].c_str(), inList[k].length());
                    }
                    opsPtr[traverseEntityOpIndex].num_outputs = 1;
                    opsPtr[traverseEntityOpIndex].output_tensors_name = (I8 **)mt_new_storage(
                        opsPtr[traverseEntityOpIndex].num_outputs * sizeof(I8 *));
                    for (int k = 0; k < (int)(opsPtr[traverseEntityOpIndex].num_outputs); k++) {
                        opsPtr[traverseEntityOpIndex].output_tensors_name[k] =
                            (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
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

    EE adapt_weights(ModelSpec *ms) override
    {
        ms->num_weight_specs = weightOpNum;
        WeightSpec *wsPtr = (WeightSpec *)mt_new_storage(sizeof(WeightSpec) * ms->num_weight_specs);
        for (int j = 0; j < ms->num_weight_specs; j++) {
            wsPtr[j].num_quant_scale = 0;
            wsPtr[j].weight_scale = nullptr;
        }
        ms->ws = wsPtr;
        Json::Reader reader;
        Json::Value value;
        if (reader.parse(newStrValue, value)) {
            for (int j = 0; j < ms->num_weight_specs; j++) {
                int curWeightIndex = weightIds[j];
                std::string weightOpType = value["node"][curWeightIndex]["op"].asString();
                std::string weightOpName = value["node"][curWeightIndex]["name"].asString();
                str_copy(wsPtr[j].op_name, weightOpName.c_str(), weightOpName.length());
                std::vector<std::string> constList = weightConstInput[weightOpName];
                UNI_DEBUG_LOG("process operator name:%s weight.\n", weightOpName.c_str());
                if (weightOpType.compare("Conv2D") == 0 ||
                    weightOpType.compare("Conv2DBackpropInput") == 0 ||
                    weightOpType.compare("MatMul") == 0 ||
                    weightOpType.compare("DepthwiseConv2dNative") == 0) {  // To collect more op

                    if (constList.size() == 1) {
                        std::string curIdenStr = constList[0];
                        std::string curConstStr = idenConst[curIdenStr];
                        int curConstIndex = constId[curConstStr];
                        if (constId.find(curIdenStr) != constId.end()) {
                            curConstIndex = constId[curIdenStr];
                        }
                        int tensorContentSize =
                            value["node"][curConstIndex]["attr"]["value"]["tensor"]["tensorContent"]
                                .size();
                        wsPtr[j].mdt = DT_F32;
                        wsPtr[j].bytes_of_weight = tensorContentSize * sizeof(float);
                        float *fp32Ptr = (float *)mt_new_storage(wsPtr[j].bytes_of_weight);
                        for (int k = 0; k < tensorContentSize; k++) {
                            fp32Ptr[k] = std::stof(
                                value["node"][curConstIndex]["attr"]["value"]["tensor"]["tensorCont"
                                                                                        "ent"][k]
                                    .asString());
                        }
                        wsPtr[j].weight = (U8 *)fp32Ptr;
                        wsPtr[j].bytes_of_vec = 0;
                        wsPtr[j].vec = nullptr;
                    } else {
                        CHECK_STATUS(NOT_IMPLEMENTED);
                    }
                } else if (weightOpType.compare("BiasAdd") == 0) {
                    if (constList.size() == 1) {
                        std::string curIdenStr = constList[0];
                        std::string curConstStr = idenConst[curIdenStr];
                        int curConstIndex = constId[curConstStr];

                        int tensorContentSize =
                            value["node"][curConstIndex]["attr"]["value"]["tensor"]["tensorContent"]
                                .size();
                        wsPtr[j].mdt = DT_F32;
                        wsPtr[j].bytes_of_weight = 0;
                        wsPtr[j].weight = nullptr;
                        wsPtr[j].bytes_of_vec = tensorContentSize * sizeof(float);
                        float *fp32Ptr = (float *)mt_new_storage(wsPtr[j].bytes_of_vec);
                        for (int k = 0; k < tensorContentSize; k++) {
                            fp32Ptr[k] = std::stof(
                                value["node"][curConstIndex]["attr"]["value"]["tensor"]["tensorCont"
                                                                                        "ent"][k]
                                    .asString());
                        }
                        wsPtr[j].vec = (U8 *)fp32Ptr;
                    } else {
                        CHECK_STATUS(NOT_IMPLEMENTED);
                    }

                } else if (weightOpType.compare("FusedBatchNorm") == 0) {
                    if (constList.size() == 4) {
                        std::string curScaleIdenStr = constList[0];
                        std::string curScaleConstStr = idenConst[curScaleIdenStr];
                        int curScaleConstIndex = constId[curScaleConstStr];
                        if (constId.find(curScaleIdenStr) != constId.end()) {
                            curScaleConstIndex = constId[curScaleIdenStr];
                        }

                        std::string curOffsetIdenStr = constList[1];
                        std::string curOffsetConstStr = idenConst[curOffsetIdenStr];
                        int curOffsetConstIndex = constId[curOffsetConstStr];
                        if (constId.find(curOffsetIdenStr) != constId.end()) {
                            curOffsetConstIndex = constId[curOffsetIdenStr];
                        }

                        std::string curMeanIdenStr = constList[2];
                        std::string curMeanConstStr = idenConst[curMeanIdenStr];
                        int curMeanConstIndex = constId[curMeanConstStr];
                        if (constId.find(curMeanIdenStr) != constId.end()) {
                            curMeanConstIndex = constId[curMeanIdenStr];
                        }

                        std::string curVarianceIdenStr = constList[3];
                        std::string curVarianceConstStr = idenConst[curVarianceIdenStr];
                        int curVarianceConstIndex = constId[curVarianceConstStr];
                        if (constId.find(curVarianceIdenStr) != constId.end()) {
                            curVarianceConstIndex = constId[curVarianceIdenStr];
                        }

                        int iterSize =
                            value["node"][curScaleConstIndex]["attr"]["value"]["tensor"]["tensorCon"
                                                                                         "tent"]
                                .size();
                        wsPtr[j].mdt = DT_F32;
                        wsPtr[j].bytes_of_weight = iterSize * sizeof(float);
                        float *fp32FirPtr = (float *)mt_new_storage(wsPtr[j].bytes_of_weight);
                        wsPtr[j].weight = (U8 *)fp32FirPtr;
                        wsPtr[j].bytes_of_vec = iterSize * sizeof(float);
                        float *fp32SecPtr = (float *)mt_new_storage(wsPtr[j].bytes_of_vec);
                        wsPtr[j].vec = (U8 *)fp32SecPtr;

                        for (int k = 0; k < iterSize; k++) {
                            float tmpScale = std::stof(
                                value["node"][curScaleConstIndex]["attr"]["value"]["tensor"]["tenso"
                                                                                             "rCont"
                                                                                             "ent"][0]
                                    .asString());
                            float tmpOffset = std::stof(
                                value["node"][curOffsetConstIndex]["attr"]["value"]["tensor"]["tens"
                                                                                              "orCo"
                                                                                              "nten"
                                                                                              "t"][0]
                                    .asString());
                            float tmpMean = std::stof(
                                value["node"][curMeanConstIndex]["attr"]["value"]["tensor"]["tensor"
                                                                                            "Conten"
                                                                                            "t"][0]
                                    .asString());
                            float tmpVariance = std::stof(
                                value["node"][curVarianceConstIndex]["attr"]["value"]["tensor"]["te"
                                                                                                "ns"
                                                                                                "or"
                                                                                                "Co"
                                                                                                "nt"
                                                                                                "en"
                                                                                                "t"][0]
                                    .asString());

                            float tmpNewMean =
                                tmpMean - tmpOffset * sqrt(tmpVariance / powf(tmpScale, 2));
                            float tmpNewVariance = tmpVariance / (powf(tmpScale, 2));
                            fp32FirPtr[k] = tmpNewMean;
                            fp32SecPtr[k] = tmpNewVariance;
                        }
                    } else {
                        CHECK_STATUS(NOT_IMPLEMENTED);
                    }
                }
            }
        }
        return SUCCESS;
    }

    ParameterSpec adapt_Eltwise() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        EltwiseParamSpec eps;
        memset(&eps, 0, sizeof(eps));
        if (opType == "Add") {
            eps.elt_mode = ELTWISE_SUM;
            eps.activation_type = ACTIVATION_NULL;
        } else if (opType == "Sub") {
            eps.elt_mode = ELTWISE_SUB;
            eps.activation_type = ACTIVATION_NULL;
        }
        curPs.eltwise_spec = eps;
        return curPs;
    }

    ParameterSpec adapt_ArgMax() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ArgMaxParamSpec aps;
        memset(&aps, 0, sizeof(aps));
        aps.axis = 1;  // TODO
        curPs.argmax_spec = aps;
        return curPs;
    }

    ParameterSpec adapt_Conv() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ConvolutionParamSpec convPs;
        memset(&convPs, 0, sizeof(convPs));
        convPs.kernel_t = 1;
        convPs.stride_t = 1;
        convPs.padding_before = 0;
        convPs.padding_after = 0;
        convPs.dilatedRate_t = 1;

        std::string conv_op = nodeV["name"].asString();
        int dilationsInfo[4] = {0, 0, 0, 0};
        int stridesInfo[4] = {0, 0, 0, 0};
        if (opType.compare("DepthwiseConv2dNative") == 0) {
            for (int i = 0; i < (int)(nodeV["attr"]["dilations"]["list"]["i"].size()); i++) {
                dilationsInfo[i] = 1;
            }
        } else {
            dilationsInfo[0] = 1;
            dilationsInfo[1] = 1;
        }
        for (int i = 0; i < (int)(nodeV["attr"]["strides"]["list"]["i"].size()); i++) {
            stridesInfo[i] = std::stoi(
                nodeV["attr"]["strides"]["list"]["i"][i].asString());  // TODO extract real data
        }
        convPs.dilatedRate_h = dilationsInfo[1];  // atten
        convPs.dilatedRate_w = dilationsInfo[2];
        convPs.stride_h = stridesInfo[1];
        convPs.stride_w = stridesInfo[2];

        std::vector<std::string> curConvIdens = this->weightConstInput[conv_op];
        int curConstId = -1;
        if (constId.find(curConvIdens[0]) != constId.end()) {
            curConstId = constId[curConvIdens[0]];
        } else {
            curConstId = constId[idenConst[curConvIdens[0]]];
        }
        std::string constOpName = this->ttValue["node"][curConstId]["name"].asString();
        std::vector<int> convWeightKernels;
        for (int k = 0; k <
             (int)(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tensorShape"]["di"
                                                                                               "m"]
                       .size());
             k++) {
            convWeightKernels.push_back(
                std::stoi(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]
                                       ["tensorShape"]["dim"][k]["size"]
                                           .asString()));
        }

        if (convWeightKernels.size() < 4) {
            UNI_ERROR_LOG("can not process operator name:%s kernel.\n", this->layerName.c_str());
        }
        if (opType.compare("DepthwiseConv2dNative") == 0) {
            convPs.num_outputs = convWeightKernels[2];
        } else {
            convPs.num_outputs = convWeightKernels[3];
        }
        convPs.kernel_h = convWeightKernels[0];
        convPs.kernel_w = convWeightKernels[1];

        std::string tfPaddingMode =
            nodeV["attr"]["padding"]["s"].asString();  // choose one of VALID/SAME
        if (tfPaddingMode.at(0) == 'V') {
            tfPaddingMode = "VALID";
            convPs.padding_top = 0;
            convPs.padding_bottom = 0;
            convPs.padding_left = 0;
            convPs.padding_right = 0;
        } else {
            tfPaddingMode = "SAME";
            convPs.padding_top = (U32)INT_MAX;
            convPs.padding_bottom = (U32)INT_MAX;
            convPs.padding_left = (U32)INT_MAX;
            convPs.padding_right = (U32)INT_MAX;
        }

        convPs.group = 1;
        convPs.dw_activation_type = ACTIVATION_NULL;
        convPs.pw_activation_type = ACTIVATION_NULL;

        if (convPs.group != 1 && convPs.group == convPs.num_outputs) {
            convPs.convolution_type = Convolution_Depthwise;
        } else {
            if (convPs.dilatedRate_h > 1 || convPs.dilatedRate_w > 1) {
                convPs.convolution_type = Convolution_Dilation;
            } else {
                convPs.convolution_type = Convolution_Pointwise;
            }
        }

        if (opType.compare("DepthwiseConv2dNative") == 0) {
            convPs.convolution_type = Convolution_Depthwise;
        }
        curPs.conv_spec = convPs;
        return curPs;
    }

    ParameterSpec adapt_BatchNorm() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        BatchNormParamSpec bps;
        memset(&bps, 0, sizeof(bps));
        bps.axis = 0;
        bps.eps = nodeV["attr"]["epsilon"]["f"].asFloat();
        bps.gama = 0;
        bps.momentum = 0;
        curPs.bn_spec = bps;
        return curPs;
    }

    ParameterSpec adapt_Fc() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        FullyConnectedParamSpec fps;
        memset(&fps, 0, sizeof(fps));
        // to locate the const weight op
        std::string curOpName = nodeV["name"].asString();
        std::vector<std::string> curConvIdens = this->weightConstInput[curOpName];
        int curConstId = -1;
        if (constId.find(curConvIdens[0]) != constId.end()) {
            curConstId = constId[curConvIdens[0]];
        } else {
            curConstId = constId[idenConst[curConvIdens[0]]];
        }
        int dimLengthIndex =
            this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tensorShape"].size() - 1;
        fps.num_outputs =
            std::stoi(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tensorShape"]
                                   ["dim"][dimLengthIndex]["size"]
                                       .asString());  // fc_dimSize is static two-dimension
        fps.num_slices = 1;
        curPs.fc_spec = fps;
        return curPs;
    }

    ParameterSpec adapt_Pooling() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        PoolingParamSpec pps;
        memset(&pps, 0, sizeof(pps));
        std::vector<int> kernelSize;  // ihwo
        std::vector<int> stridesInfo;
        for (int i = 0; i < (int)(nodeV["attr"]["ksize"]["list"]["i"].size()); i++) {
            kernelSize.push_back(std::stoi(nodeV["attr"]["ksize"]["list"]["i"][i].asString()));
        }
        for (int i = 0; i < (int)(nodeV["attr"]["strides"]["list"]["i"].size()); i++) {
            stridesInfo.push_back(std::stoi(nodeV["attr"]["strides"]["list"]["i"][i].asString()));
        }
        pps.kernel_t = 1;
        pps.kernel_h = kernelSize[1];
        pps.kernel_w = kernelSize[2];
        pps.stride_t = 1;
        pps.stride_h = 1;
        pps.stride_w = 1;
        pps.padding_before = 0;
        pps.padding_after = 0;
        pps.padding_top = 0;
        pps.padding_bottom = 0;
        pps.padding_left = 0;
        pps.padding_right = 0;
        pps.rm = CEIL;
        if (opType.compare("MaxPool") == 0) {
            pps.mode = POOLING_MAX;
        } else {  // refer to "AvgPool"
            pps.mode = POOLING_MEAN;
        }
        curPs.pooling_spec = pps;
        return curPs;
    }

    ParameterSpec adapt_Reduction() override
    {
        // Mapping to <Mean>
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ReductionParamSpec reductionPs;
        memset(&reductionPs, 0, sizeof(reductionPs));
        if (opType.compare("Mean") == 0) {
            reductionPs.reduction_mode = REDUCTION_MEAN;
        } else {
            UNI_ERROR_LOG("can not map operator name:%s type:%s to Reduction.\n",
                this->layerName.c_str(), opType.c_str());
        }
        std::string reductionOpName = nodeV["name"].asString();
        std::vector<std::string> constInputs = weightConstInput[reductionOpName];
        int constReductionOpIndex = -1;
        if (constId.find(constInputs[0]) != constId.end()) {
            constReductionOpIndex = constId[constInputs[0]];
        } else {
            constReductionOpIndex = constId[idenConst[constInputs[0]]];
        }
        reductionPs.axes_num =
            this->ttValue["node"][constReductionOpIndex]["attr"]["value"]["tensor"]["tensorContent"]
                .size();
        for (int i = 0; i < reductionPs.axes_num; i++) {
            reductionPs.axes[i] = std::stoi(
                this->ttValue["node"][constReductionOpIndex]["attr"]["value"]["tensor"]["tensorCont"
                                                                                        "ent"][i]
                    .asString());
        }
        curPs.reduction_spec = reductionPs;
        return curPs;
    }

    ParameterSpec adapt_Pad() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        PadParamSpec padPs;
        memset(&padPs, 0, sizeof(padPs));

        std::string curOpName = nodeV["name"].asString();
        std::vector<std::string> curConvIdens = this->weightConstInput[curOpName];
        int curConstId = -1;
        if (constId.find(curConvIdens[0]) != constId.end()) {
            curConstId = constId[curConvIdens[0]];
        } else {
            curConstId = constId[idenConst[curConvIdens[0]]];
        }

        std::vector<int> padInfos;
        for (int i = 0; i < (int)(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["ten"
                                                                                               "sor"
                                                                                               "Con"
                                                                                               "ten"
                                                                                               "t"]
                                      .size());
             i++) {
            padInfos.push_back(
                std::stoi(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tensorConte"
                                                                                       "nt"][i]
                              .asString()));
        }
        padPs.before = 0;
        padPs.after = 0;
        padPs.top = padInfos[2];
        padPs.bottom = padInfos[3];
        padPs.left = padInfos[4];
        padPs.right = padInfos[5];
        padPs.constant_value = 0;  // TODO: for PadV2
        padPs.pad_mode = Pad_Constant;
        curPs.pad_spec = padPs;
        return curPs;
    }

    ParameterSpec adapt_Concat() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ConcatParamSpec concatPs;
        memset(&concatPs, 0, sizeof(concatPs));
        concatPs.axis = std::stoi(nodeV["attr"]["N"]["i"].asString());
        curPs.concat_spec = concatPs;
        return curPs;
    }

    ParameterSpec adapt_Resize() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ResizeParamSpec resizePs;
        memset(&resizePs, 0, sizeof(resizePs));

        std::string curOpName = nodeV["name"].asString();
        std::vector<std::string> curConvIdens = this->weightConstInput[curOpName];
        int curConstId = -1;
        if (constId.find(curConvIdens[0]) != constId.end()) {
            curConstId = constId[curConvIdens[0]];
        } else {
            curConstId = constId[idenConst[curConvIdens[0]]];
        }
        resizePs.num_sizes = 2;
        resizePs.num_scales = 0;
        for (int k = 0; k < (int)(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["ten"
                                                                                               "sor"
                                                                                               "Con"
                                                                                               "ten"
                                                                                               "t"]
                                      .size());
             k++) {
            resizePs.sizes[k] =
                std::stoi(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tensorConte"
                                                                                       "nt"][k]
                              .asString());
        }

        curPs.resize_spec = resizePs;
        return curPs;
    }

    ParameterSpec adapt_Power() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        PowerParamSpec powerPs;
        memset(&curPs, 0, sizeof(powerPs));
        float curScale = 1.0;
        float curShift = 0.0;

        if (opType.compare("Rsqrt") == 0) {
            powerPs.power = 0.5;
            curPs.power_spec = powerPs;
            return curPs;
        }

        std::string curOpName = nodeV["name"].asString();
        std::vector<std::string> curConvIdens = this->weightConstInput[curOpName];
        int curConstId = -1;
        if (constId.find(curConvIdens[0]) != constId.end()) {
            curConstId = constId[curConvIdens[0]];
        } else {
            curConstId = constId[idenConst[curConvIdens[0]]];
        }

        if (opType.compare("Mul") == 0) {
            curScale = std::stof(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tens"
                                                                                              "orCo"
                                                                                              "nten"
                                                                                              "t"][0]
                                     .asString());
        } else if (opType.compare("Sub") == 0) {
            curShift = -1 *
                std::stof(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tensorConte"
                                                                                       "nt"][0]
                              .asString());
        } else if (opType.compare("RealDiv") == 0) {
            curScale = 1.0 /
                std::stof(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tensorConte"
                                                                                       "nt"][0]
                              .asString());
        }
        powerPs.scale = curScale;
        powerPs.shift = curShift;
        powerPs.power = 1;
        curPs.power_spec = powerPs;
        return curPs;
    }

    ParameterSpec adapt_Transpose() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        TransposeParamSpec transPs;
        memset(&transPs, 0, sizeof(transPs));
        // extract the perm info from the const input
        std::string curOpName = nodeV["name"].asString();
        std::vector<std::string> curConvIdens = this->weightConstInput[curOpName];
        int curConstId = -1;
        if (constId.find(curConvIdens[0]) != constId.end()) {
            curConstId = constId[curConvIdens[0]];
        } else {
            curConstId = constId[idenConst[curConvIdens[0]]];
        }

        transPs.trans_size =
            this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tensorContent"].size();
        for (int i = 0; i < (int)(transPs.trans_size); i++) {
            transPs.trans_dims[i] =
                std::stoi(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tensorConte"
                                                                                       "nt"][i]
                              .asString());
            ;
        }
        curPs.transpose_spec = transPs;
        return curPs;
    }

    ParameterSpec adapt_Reshape() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ReshapeParamSpec reshapePs;
        memset(&reshapePs, 0, sizeof(reshapePs));

        std::string curOpName = nodeV["name"].asString();
        std::vector<std::string> curConvIdens = this->weightConstInput[curOpName];
        if (curConvIdens.size() == 0) {
            return curPs;
        }
        int curConstId = -1;
        if (constId.find(curConvIdens[0]) != constId.end()) {
            curConstId = constId[curConvIdens[0]];
        } else {
            curConstId = constId[idenConst[curConvIdens[0]]];
        }
        reshapePs.shape_size =
            this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tensorContent"].size();
        for (int k = 0; k < reshapePs.shape_size; k++) {
            reshapePs.shape_dims[k] =
                std::stoi(this->ttValue["node"][curConstId]["attr"]["value"]["tensor"]["tensorConte"
                                                                                       "nt"][k]
                              .asString());
        }
        reshapePs.axis = 8;
        reshapePs.num_axes = -1;
        curPs.reshape_spec = reshapePs;
        return curPs;
    }

    ParameterSpec adapt_Squeeze() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        SqueezeParamSpec squeezePs;
        memset(&squeezePs, 0, sizeof(squeezePs));
        std::vector<int> squeezeDimsInfo;
        squeezePs.axes_num = nodeV["attr"]["squeeze_dims"]["list"]["i"].size();
        for (int i = 0; i < (int)(nodeV["attr"]["squeeze_dims"]["list"]["i"].size()); i++) {
            squeezePs.axes[i] = std::stoi(nodeV["attr"]["squeeze_dims"]["list"]["i"][i].asString());
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
        std::string unsqueeze_op = nodeV["name"].asString();
        int expandDimIndex = constId[idenConst[weightConstInput[unsqueeze_op][0]]];
        unsqueezePs.axes_num =
            this->ttValue["node"][expandDimIndex]["attr"]["value"]["tensor"]["tensorContent"].size();
        for (int k = 0; k < unsqueezePs.axes_num; k++) {
            unsqueezePs.axes[k] = std::stoi(
                this->ttValue["node"][expandDimIndex]["attr"]["value"]["tensor"]["tensorContent"][k]
                    .asString());
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
        castPs.targetDt = DT_F32;
        curPs.cast_spec = castPs;
        return curPs;
    }

private:
    int modelInputLayerNum;
    int entityOpCount;

    std::string modelName;
    std::string newStrValue;
    Json::Value nodeV;
    int curNodeIndex;
    std::string opType;
    std::string layerName;

    Json::Value ttValue;

    std::map<std::string, int> constId;
    std::map<std::string, std::string> idenConst;
    std::map<std::string, std::vector<std::string>> weightConstInput;
    std::vector<int> weightIds;

    int weightOpNum;
    int curInDegree;
};
#endif
