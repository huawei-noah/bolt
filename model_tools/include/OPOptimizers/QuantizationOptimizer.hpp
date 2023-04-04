// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_QUANTIZATIONOPTIMIZER
#define _H_QUANTIZATIONOPTIMIZER

// ptr->feature_scale[0].scale[0] 0: no quant
// ptr->feature_scale[0].scale[0] -1: quant, input int8, output int8 (x86->u8, arm->i8)
// ptr->feature_scale[0].scale[0] -2: quant, input int8, output float32/float16
// ptr->feature_scale[0].scale[0] -3: quant, input float32/float16, output float32/float16
// ptr->feature_scale[0].scale[0] -5: quant, input int8, output int8 (x86->i8, arm->i8)
#include <set>
#include <unordered_map>
#include "OPOptimizer.hpp"
#include <sstream>
#include <fstream>
#include <json/json.h>

const static std::set<OperatorType> NoQuantOP = {OT_HSwish, OT_HSigmoid, OT_Sigmoid, OT_Clip,
    OT_Gelu, OT_TanH, OT_Resize, OT_LayerNorm, OT_HSwishNoDiv, OT_Eltwise, OT_PRelu,
    OT_Softmax, OT_LogSoftmax, OT_DetectionOutput, OT_Scale, OT_SharedWeight, OT_Concat,
    OT_Swish, OT_OneHot, OT_Where, OT_Cast, OT_TopK, OT_Round, OT_Range, OT_NonMaxSuppression, OT_Exp,
    OT_Log, OT_Floor, OT_Reduction}; // 0
const static std::set<OperatorType> QuantOP = {OT_Conv, OT_MatMul, OT_FC, OT_Deconvolution}; // -1 or -2
const static std::set<OperatorType> FQuantOP = {OT_RNN}; // -3
const static std::set<OperatorType> DependPreOP = {OT_Gather, OT_Embedding};
const static std::set<OperatorType> C8OP = {OT_Conv};

class QuantizationOptimizer : public OPOptimizer {
public:
    QuantizationOptimizer()
    {
        this->actFP = false;
        this->scaleFile = nullptr;
        this->clipVal = -1;
    }

    QuantizationOptimizer(bool actFP, const char *scaleFile, F32 clipVal)
    {
        this->actFP = actFP;
        this->scaleFile = scaleFile;
        this->clipVal = clipVal;
    }

    void SetC8Flag(ModelSpec *spec)
    {
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type != OT_QuantizeLinear) {
                continue;
            }
            std::string curOut = spec->ops[i].output_tensors_name[0];
            auto nextIndex =
                searchOperatorIndexByInput(spec, curOut, i + 1, spec->num_operator_specs);
            bool trans = true;
            for (auto next : nextIndex) {
                if (!C8OP.count(spec->ops[next.first].type)) {
                    trans = false;
                }
            }
            // spec->ops[i].ps.quant_spec.trans = trans;
        }
    }

    void mergeQuantizeLinear(ModelSpec *spec)
    {
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_None || spec->ops[i].num_outputs > 1) {
                continue;
            }
            std::string curOut = spec->ops[i].output_tensors_name[0];
            auto nextIndex =
                searchOperatorIndexByInput(spec, curOut, i + 1, spec->num_operator_specs);
            if (nextIndex.size() < 2) {
                continue;
            }
            std::set<int> quantizeIdx;
            for (auto next : nextIndex) {
                if (spec->ops[next.first].type == OT_QuantizeLinear) {
                    quantizeIdx.insert(next.first);
                }
            }
            if (quantizeIdx.size() <= 1) {
                continue;
            }
            std::unordered_map<float, std::vector<int>> m;
            for (int idx : quantizeIdx) {
                m[spec->ops[idx].feature_scale[0].scale[0]].push_back(idx);
            }
            for (auto ele : m) {
                if (ele.second.size() > 1) {
                    // merge these Quantize OP
                    std::string inputName = spec->ops[ele.second[0]].output_tensors_name[0];
                    for (unsigned j = 1; j < ele.second.size(); ++j) {
                        curOut = spec->ops[ele.second[j]].output_tensors_name[0];
                        nextIndex = searchOperatorIndexByInput(
                            spec, curOut, i + 1, spec->num_operator_specs);
                        if (nextIndex.size() > 1) {
                            CHECK_STATUS(NOT_SUPPORTED);
                        }
                        str_copy(
                            spec->ops[nextIndex[0].first].input_tensors_name[nextIndex[0].second],
                            inputName.data(), strlen(inputName.data()));
                        setOperatorInvalid(spec, ele.second[j], true);
                    }
                }
            }
        }
    }

    void insertQuantizeLinearKernel(ModelSpec *spec,
        int insertIdx,
        int nextIdx,
        int nextInputIdx,
        std::string inputName,
        std::string outputName)
    {
        OperatorSpec quantizeOperator =
            mt_create_operator(outputName.c_str(), OT_QuantizeLinear, 1, 1);
        quantizeOperator.ps.quant_spec.axis = 0;
        quantizeOperator.ps.quant_spec.dt = DT_U8_Q;
        // quantizeOperator.ps.quant_spec.trans = false;
        str_copy(
            quantizeOperator.output_tensors_name[0], outputName.data(), strlen(outputName.data()));
        str_copy(quantizeOperator.input_tensors_name[0], inputName.data(), strlen(inputName.data()));
        str_copy(spec->ops[nextIdx].input_tensors_name[nextInputIdx], outputName.data(),
            strlen(outputName.data()));
        mt_insert_operator(spec, insertIdx, quantizeOperator);

        nextIdx += 1;
        if (spec->ops[nextIdx].type == OT_MatMul && nextInputIdx == 1) {
            spec->ops[insertIdx].ps.quant_spec.dt = DT_I8;
        }

        // set the output scale
        if (spec->ops[nextIdx].num_quant_feature ==
            (spec->ops[nextIdx].num_inputs + spec->ops[nextIdx].num_outputs)) {
            spec->ops[insertIdx].num_quant_feature = 1;
            spec->ops[insertIdx].feature_scale = (QuantSpec *)mt_malloc(sizeof(QuantSpec));
            U32 numScale = spec->ops[nextIdx].feature_scale[nextInputIdx].num_scale;
            spec->ops[insertIdx].feature_scale[0].num_scale = numScale;
            spec->ops[insertIdx].feature_scale[0].scale = (F32 *)mt_malloc(sizeof(F32) * numScale);
            UNI_MEMCPY(spec->ops[insertIdx].feature_scale[0].scale,
                spec->ops[nextIdx].feature_scale[nextInputIdx].scale, sizeof(F32) * numScale);
        } else {
            label_OP(spec->ops + insertIdx, -1);
        }
    }

    void insertQuantizeLinear(ModelSpec *spec)
    {
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_None) {
                continue;
            }
            CHECK_REQUIREMENT(spec->ops[i].num_quant_feature > 0);
            CHECK_REQUIREMENT(spec->ops[i].feature_scale[0].num_scale > 0);
            if ((spec->ops[i].feature_scale[0].scale[0] == 0) ||
                (spec->ops[i].feature_scale[0].scale[0] == -3) ||
                ((spec->ops[i].num_quant_feature > spec->ops[i].num_inputs) &&
                 (spec->ops[i].feature_scale[spec->ops[i].num_inputs].scale[0] == -3)))
            {
                continue;
            }
            for (unsigned k = 0; k < spec->ops[i].num_inputs; ++k) {
                std::string curIn = spec->ops[i].input_tensors_name[k];
                std::vector<std::pair<int, int>> prevIndex =
                    searchOperatorIndexByOutput(spec, curIn, 0, i);

                int prevNumQuant = 0;
                int prevOutputScale = 0;
                int insertIdx = i;
                if (!prevIndex.empty()) {
                    prevNumQuant = spec->ops[prevIndex[0].first].num_quant_feature;
                    prevOutputScale =
                        spec->ops[prevIndex[0].first].feature_scale[prevNumQuant - 1].scale[0];
                    insertIdx = prevIndex[0].first + 1;
                }
                if (prevOutputScale != -1) {
                    std::string quantizeName = allocName("Quantize_" + curIn + std::to_string(i));
                    insertQuantizeLinearKernel(spec, insertIdx, i, k, curIn, quantizeName);
                    ++i;
                }
            }
        }
    }

    std::pair<int, std::vector<int>> FindPathToNextLabeledOP(ModelSpec *spec, int i)
    {
        std::vector<int> paths(1, i);
        std::vector<std::pair<int, int>> nextIndex(1, std::make_pair(i, 0));
        while (nextIndex.size() > 0) {
            int nextId = nextIndex[0].first;
            if (nextIndex.size() > 1 || spec->ops[nextId].num_outputs > 1) {
                return std::make_pair(-2, paths);
            }
            if (spec->ops[nextId].num_quant_feature > 0) {
                if (spec->ops[nextId].feature_scale[0].scale[0] == -3) {
                    return std::make_pair(-2, paths);
                }
                return std::make_pair(nextId, paths);
            }
            paths.push_back(nextId);
            std::string curOut = spec->ops[nextId].output_tensors_name[0];
            nextIndex =
                searchOperatorIndexByInput(spec, curOut, nextId + 1, spec->num_operator_specs);
        }
        return std::make_pair(-1, paths);
    }

    bool isNotNaiveRelu(OperatorSpec *op)
    {
        return ((op->type == OT_Relu) && (op->ps.relu_spec.neg_slope != 0));
    }

    bool isAvgPooling(OperatorSpec *op)
    {
        return ((op->type == OT_Pooling) && (op->ps.pooling_spec.mode == POOLING_MEAN));
    }

    bool isDepthWiseConv(OperatorSpec *op)
    {
        return ((op->type == OT_Conv) && (op->ps.conv_spec.convolution_type == CONVOLUTION_DEPTHWISE));
    }

    bool isDepthWiseDeConv(OperatorSpec *op)
    {
        return ((op->type == OT_Deconvolution) && (op->ps.conv_spec.num_outputs_origin == op->ps.conv_spec.group));
    }

    bool isNoQuantOp(OperatorSpec *op)
    {
        return (NoQuantOP.count(op->type) || isDepthWiseConv(op) || isAvgPooling(op) ||
            isNotNaiveRelu(op) || isDepthWiseDeConv(op));
    }

    bool isQuantOp(OperatorSpec *op)
    {
        return (QuantOP.count(op->type) && !isNoQuantOp(op));
    }

    void parseAndPreLabel(ModelSpec *spec, Json::Value value)
    {
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_None) {
                continue;
            }
            if (OT_Conv == spec->ops[i].type) {
                std::string curIn = spec->ops[i].input_tensors_name[0];
                // input is model input
                if (this->is_kin_to_model_input(spec, curIn, i)) {
                    this->label_OP_as_no_quant(spec->ops + i);
                    continue;
                }
            }
            if (QuantOP.count(spec->ops[i].type)) {
                // All quantized nodes are labeled float output at first.
                label_OP_as_quant_float(spec->ops + i);
            }

            if (FQuantOP.count(spec->ops[i].type)) {
                label_OP_as_float_quant_float(spec->ops + i);
            }

            bool isOutputOp = false;
            for (unsigned j = 0; j < spec->ops[i].num_outputs; ++j) {
                if (outputNames.count(std::string(spec->ops[i].output_tensors_name[j]))) {
                    isOutputOp = true;
                    break;
                }
            }
            if (isNoQuantOp(spec->ops + i) ||
                (isOutputOp && !QuantOP.count(spec->ops[i].type) && !FQuantOP.count(spec->ops[i].type))) {
                label_OP_as_no_quant(spec->ops + i);
            }
        }

        if (value != Json::Value::null &&
            (value.get("quantization", Json::Value::null).asString() == "dynamic"))
        {
            auto quanOpsVal = value.get("quant_ops", Json::Value::null);
            auto noQuantOpsVal = value.get("no_quant_ops", Json::Value::null);
            std::set<std::string> quantOps;
            std::set<std::string> noQuantOps;
            if (quanOpsVal != Json::Value::null) {
                for (auto v: quanOpsVal) {
                    quantOps.insert(v.asString());
                }
            }
            if (noQuantOpsVal != Json::Value::null) {
                for (auto v: noQuantOpsVal) {
                    noQuantOps.insert(v.asString());
                }
            }
            for (int i = 0; i < spec->num_operator_specs; i++) {
                if (spec->ops[i].type == OT_None) {
                    continue;
                }
                if (quantOps.count(std::string(spec->ops[i].name))) {
                    if (QuantOP.count(spec->ops[i].type)) {
                        // All quantized nodes are labeled float output at first.
                        label_OP_as_quant_float(spec->ops + i);
                    }

                    if (FQuantOP.count(spec->ops[i].type)) {
                        label_OP_as_float_quant_float(spec->ops + i);
                    }
                }
                if (noQuantOps.count(std::string(spec->ops[i].name))) {
                    label_OP_as_no_quant(spec->ops + i);
                }
            }
        }
    }

    float setToTheNextScale(ModelSpec *spec, int i)
    {
        std::string curOut = spec->ops[i].output_tensors_name[0];
        auto nextIndex = searchOperatorIndexByInput(spec, curOut, i + 1, spec->num_operator_specs);
        if (nextIndex.empty()) {
            return 0;
        }
        float scale = -99;
        for (auto index : nextIndex) {
            float tmp = 0;
            if (spec->ops[index.first].num_quant_feature == 0) {
                tmp = setToTheNextScale(spec, index.first);
            } else {
                tmp = spec->ops[index.first]
                          .feature_scale[spec->ops[index.first].num_quant_feature - 1]
                          .scale[0];
                if (tmp < -1) {
                    tmp += 1;
                }
            }
            if (scale != -99 && tmp != scale) {
                scale = 0;
                break;
            }
            scale = tmp;
        }
        if (nextIndex.empty()) {
            scale = 0;
        }
        label_OP(spec->ops + i, scale);
        return scale;
    }

    void labelContinuousPath(ModelSpec *spec)
    {
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].num_quant_feature > 0 || spec->ops[i].type == OT_None) {
                continue;
            }
            if (DependPreOP.count(spec->ops[i].type)) {
                label_OP(spec->ops + i, 0);
                continue;
            }
            std::pair<int, std::vector<int>> path = FindPathToNextLabeledOP(spec, i);
            int flag = 0;
            if (path.first > 0 && spec->ops[path.first].feature_scale[0].scale[0] < 0) {
                flag = -1;
            }
            if (path.first != -2) {
                for (int id : path.second) {
                    label_OP(spec->ops + id, flag);
                }
            }
        }
        for (int i = 0; i < spec->num_operator_specs; i++) {  // handle the multi-branch case
            if (spec->ops[i].type == OT_None || NoQuantOP.count(spec->ops[i].type) ||
                QuantOP.count(spec->ops[i].type) || spec->ops[i].num_quant_feature > 0) {
                continue;
            }
            setToTheNextScale(spec, i);
        }
    }

    void SetQuantOutput(ModelSpec *spec)
    {
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_None) {
                continue;
            }
            if (isQuantOp(spec->ops + i)) {
                std::string curOut = spec->ops[i].output_tensors_name[0];
                std::vector<std::pair<int, int>> nextIndex =
                    searchOperatorIndexByInput(spec, curOut, i + 1, spec->num_operator_specs);
                int flag = -1;
                if (nextIndex.empty() || outputNames.count(curOut)) {
                    continue;
                }
                for (auto nextNode : nextIndex) {
                    if (spec->ops[nextNode.first].feature_scale[0].scale[0] != -1 &&
                        (!isQuantOp(spec->ops + nextNode.first))) {
                        flag = -2;
                        break;
                    }
                }
                if (flag == -1 && spec->ops[i].feature_scale[0].scale[0] != 0) {
                    label_OP_as_quant_int8(spec->ops + i);
                }
            }
        }
    }

    void clip_weight(ModelSpec *spec, int idx, F32 clipVal)
    {
        int weightIdx = searchWeightIndex(spec, spec->ops[idx].name);
        if (weightIdx < 0) {
            return;
        }
        CHECK_REQUIREMENT(DT_F32 == spec->ws[weightIdx].mdt);

        UNI_INFO_LOG("Clipping the weight of %s\n", spec->ops[idx].name);

        F32 clipMax = clipVal;
        F32 clipMin = -1 * clipMax;
        U32 len = spec->ws[weightIdx].bytes_of_weight / bytesOf(DT_F32);
        F32 *w = (F32 *)mt_malloc(spec->ws[weightIdx].bytes_of_weight);
        UNI_MEMCPY(w, spec->ws[weightIdx].weight, spec->ws[weightIdx].bytes_of_weight);
        for (U32 j = 0; j < len; j++) {
            if (w[j] > clipMax) {
                w[j] = clipMax;
            } else if (w[j] < clipMin) {
                w[j] = clipMin;
            }
        }
        mt_free(spec->ws[weightIdx].weight, spec);
        spec->ws[weightIdx].weight = (U8 *)w;
    }

    void label_OP(OperatorSpec *ptr, float scale)
    {
        switch (ptr->num_quant_feature) {
            case 0: {
                ptr->num_quant_feature = 1;
                ptr->feature_scale = (QuantSpec *)mt_malloc(sizeof(QuantSpec));
                ptr->feature_scale[0].num_scale = 1;
                ptr->feature_scale[0].scale = (F32 *)mt_malloc(sizeof(F32));
                ptr->feature_scale[0].scale[0] = scale;
                break;
            }
            case 1: {
                CHECK_REQUIREMENT(1 == ptr->feature_scale[0].num_scale);
                ptr->feature_scale[0].scale[0] = scale;
                break;
            }
            default: {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
    }

    void label_OP_as_no_quant(OperatorSpec *ptr)
    {
        label_OP(ptr, 0);
    }

    void label_OP_as_quant_float(OperatorSpec *ptr)
    {
        label_OP(ptr, -2);
    }

    void label_OP_as_quant_int8(OperatorSpec *ptr)
    {
        label_OP(ptr, -1);
    }

    void label_OP_as_float_quant_float(OperatorSpec *ptr)
    {
        label_OP(ptr, -3);
    }

    bool is_kin_to_model_input(ModelSpec *spec, std::string name, int bound)
    {
        if (0 == bound) {
            return true;
        }
        std::vector<std::pair<int, int>> prevIndices =
            searchOperatorIndexByOutput(spec, name, 0, bound, false);
        if (0 == prevIndices.size()) {
            return true;
        }
        int prevIndex = prevIndices[prevIndices.size() - 1].first;
        OperatorType ot = spec->ops[prevIndex].type;
        if (OT_Conv == spec->ops[bound].type &&
            (spec->ops[bound].ps.conv_spec.convolution_type == CONVOLUTION_POINTWISE ||
                spec->ops[bound].ps.conv_spec.convolution_type == CONVOLUTION_DILATION)) {
            if (spec->ops[bound].ps.conv_spec.num_outputs % 8 != 0) {
                return true;
            }
            int weightIdx = searchWeightIndex(spec, spec->ops[bound].name);
            CHECK_REQUIREMENT(weightIdx >= 0);
            int ic = spec->ws[weightIdx].bytes_of_weight / bytesOf(spec->ws[weightIdx].mdt) /
                spec->ops[bound].ps.conv_spec.num_outputs;
            if (spec->ops[bound].ps.conv_spec.kernel_t > 0) {
                ic /= spec->ops[bound].ps.conv_spec.kernel_t;
            }
            if (spec->ops[bound].ps.conv_spec.kernel_h > 0) {
                ic /= spec->ops[bound].ps.conv_spec.kernel_h;
            }
            if (spec->ops[bound].ps.conv_spec.kernel_w > 0) {
                ic /= spec->ops[bound].ps.conv_spec.kernel_w;
            }
            if (ic % 8 != 0) {
                return true;
            }
        }
        if (OT_Deconvolution == ot || OT_Conv == ot || OT_FC == ot || OT_MatMul == ot) {
            return false;
        }
        for (U32 i = 0; i < spec->ops[prevIndex].num_inputs; i++) {
            if (!is_kin_to_model_input(spec, spec->ops[prevIndex].input_tensors_name[i], prevIndex)) {
                return false;
            }
        }
        return true;
    }

    bool optimizeNormal(ModelSpec *spec, Json::Value value)
    {
        // Label the key quant and no-quant OP
        parseAndPreLabel(spec, value);

        // Label the other OP, depending on the graph
        // If a continuous path ends at a quantized node, these nodes will be labeled as quantized nodes with int8 output.
        // However, if the path ends at a non-quantized node, then these nodes will be marked as non-quantized nodes with float output.
        labelContinuousPath(spec);

        // Modify some quantized OP output to int8.
        SetQuantOutput(spec);

        return true;
    }

    bool optimizeActFP(ModelSpec *spec)
    {
        // Label the quant and no-quant OP
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_None) {
                continue;
            }
            if (OT_Conv == spec->ops[i].type) {
                std::string curIn = spec->ops[i].input_tensors_name[0];
                if (this->is_kin_to_model_input(spec, curIn, i)) {  // input is model input
                    this->label_OP_as_no_quant(spec->ops + i);
                    continue;
                }
            }
            if (isQuantOp(spec->ops + i)) {
                // All quantized nodes are labeled float output.
                label_OP_as_quant_float(spec->ops + i);
            } else {
                label_OP_as_no_quant(spec->ops + i);
            }
        }
        return true;
    }

    bool optimizeQATWithScale(ModelSpec *spec, Json::Value value)
    {
        parseAndPreLabel(spec, Json::Value::null);

        for (I32 i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_None) {
                continue;
            }
            if ((spec->ops[i].type != OT_SharedWeight) && 
                (spec->ops[i].num_quant_feature == 1) && 
                (spec->ops[i].feature_scale[0].scale[0] == 0)) {
                continue;
            }
            std::string layerName = std::string(spec->ops[i].name);

            // only quantize the layer in the scale file
            if (!value[layerName].isObject()) {
                label_OP_as_no_quant(spec->ops + i);
                continue;
            }

            if (spec->ops[i].num_quant_feature == 0) {
                UNI_WARNING_LOG("The %s Layer may not be quantized.\n", layerName.c_str());
            }

            std::vector<std::vector<F32>> scales;

            // all nodes are set to F32 default
            U32 inputNum = spec->ops[i].num_inputs;
            U32 outputNum = spec->ops[i].num_outputs;
            F32 scaleVal = -2;
            if (FQuantOP.count(spec->ops[i].type)) {
                scaleVal = -3;
            }
            for (U32 j = 0; j < inputNum; j++) {
                scales.push_back({scaleVal});
            }
            for (U32 j = 0; j < outputNum; j++) {
                scales.push_back({scaleVal});
            }
            if (value[layerName]["inputs"].isObject()) {
                for (U32 j = 0; j < inputNum; j++) {
                    // only support 1 clip value now
                    std::string inputName = std::string(spec->ops[i].input_tensors_name[j]);
                    if (value[layerName]["inputs"][inputName].isDouble()) {
                        scales[j] = {127.0f / value[layerName]["inputs"][inputName].asFloat()};
                    }
                }
            }

            if (value[layerName]["outputs"].isObject()) {
                for (U32 j = 0; j < outputNum; j++) {
                    // only support 1 clip value now
                    std::string outputName = std::string(spec->ops[i].output_tensors_name[j]);
                    if (value[layerName]["outputs"][outputName].isDouble()) {
                        scales[j] = {127.0f / value[layerName]["outputs"][outputName].asFloat()};
                    }
                }
            }

            // weight clip value
            if (value[layerName]["weights"].isObject() && value[layerName]["weights"].size() >= 1) {
                CHECK_REQUIREMENT(value[layerName]["weights"].size() == 1);
                Json::Value::Members members = value[layerName]["weights"].getMemberNames();
                float val = 0;
                if (value[layerName]["weights"][members[0]].isDouble()) {
                    val = value[layerName]["weights"][members[0]].asFloat();
                } else {
                    // val = value[layerName]["weights"][members[0]][0].asFloat();
                // } else {
                    CHECK_STATUS(NOT_SUPPORTED);
                }
                clip_weight(spec, i, val);
                if (spec->ops[i].type == OT_SharedWeight) {
                    scales[0] = {127.0f / value[layerName]["weights"][members[0]].asFloat()};
                }
            }

            // Store scales into result model
            if (nullptr != spec->ops[i].feature_scale) {  // Could be labelled with -2
                for (U32 k = 0; k < spec->ops[i].num_quant_feature; k++) {
                    mt_free(spec->ops[i].feature_scale[k].scale);
                }
                mt_free(spec->ops[i].feature_scale);
            }

            spec->ops[i].num_quant_feature = scales.size();
            spec->ops[i].feature_scale = (QuantSpec *)mt_malloc(scales.size() * sizeof(QuantSpec));

            for (U32 k = 0; k < scales.size(); k++) {
                spec->ops[i].feature_scale[k].num_scale = scales[k].size();
                U32 scaleBytes = scales[k].size() * sizeof(F32);
                spec->ops[i].feature_scale[k].scale = (F32 *)mt_malloc(scaleBytes);
                UNI_MEMCPY(spec->ops[i].feature_scale[k].scale, scales[k].data(), scaleBytes);
            }
        }
        return true;
    }

    bool optimizeQATWithGlobalClip(ModelSpec *spec)
    {
        // Label the quant and no-quant OP
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_None) {
                continue;
            }
            if (OT_Conv == spec->ops[i].type) {
                std::string curIn = spec->ops[i].input_tensors_name[0];
                if (this->is_kin_to_model_input(spec, curIn, i)) {  // input is model input
                    this->label_OP_as_no_quant(spec->ops + i);
                    continue;
                }
            }
            if (isQuantOp(spec->ops + i)) {
                // All quantized nodes are labeled float output.
                // label_OP_as_quant_float(spec->ops + i);
                clip_weight(spec, i, this->clipVal);
                U32 scaleNum = spec->ops[i].num_inputs + spec->ops[i].num_outputs;
                spec->ops[i].feature_scale = (QuantSpec *)mt_malloc(scaleNum * sizeof(QuantSpec));
                for (U32 j = 0; j < scaleNum; ++j) {
                    spec->ops[i].feature_scale[j].num_scale = 1;
                    spec->ops[i].feature_scale[j].scale = (F32 *)mt_malloc(sizeof(F32));
                    if (j < spec->ops[i].num_inputs) {
                        spec->ops[i].feature_scale[j].scale[0] = this->clipVal;
                    } else {
                        spec->ops[i].feature_scale[j].scale[0] = -2;
                    }
                }
            } else {
                label_OP_as_no_quant(spec->ops + i);
            }
        }
        return true;
    }

    bool optimize(ModelSpec *spec)
    {
        for (int i = 0; i < spec->num_outputs; ++i) {
            outputNames.insert(std::string(spec->output_names[i]));
        }
        if (this->scaleFile != nullptr) {
            std::fstream file(std::string(this->scaleFile), std::ios::in);
            Json::Value value;
            Json::Reader reader;
            if (!reader.parse(file, value)) {
                UNI_ERROR_LOG("%s is not a valid JSON file.", scaleFile);
            }
            file.close();
            if ("static" == value.get("quantization", Json::Value("static")).asString()) {
                optimizeQATWithScale(spec, value);
            } else if ("dynamic" == value["quantization"].asString()) {
                optimizeNormal(spec, value);
            } else {
                UNI_ERROR_LOG("\"quantization\" must be static or dynamic, now it is %s\n",
                    value["quantization"].asString().c_str());
            }
        } else if (this->clipVal > 0) {
            optimizeQATWithGlobalClip(spec);
        } else if (this->actFP) {
            optimizeActFP(spec);
        } else {
            optimizeNormal(spec, Json::Value::null);
        }

        // Insert QuantizeLinearOP between the OP with float output and the quantized OP.
        insertQuantizeLinear(spec);
        mergeQuantizeLinear(spec);
        return true;
    }

private:
    bool actFP;
    const char *scaleFile;
    F32 clipVal;
    std::set<std::string> outputNames;
};
#endif
