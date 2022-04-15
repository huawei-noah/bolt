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
// ptr->feature_scale[0].scale[0] -1: quant, output int8
// ptr->feature_scale[0].scale[0] -2: quant, output float32/float16
#include <set>
#include <unordered_map>
#include "OPOptimizer.hpp"
#include <sstream>
#include <fstream>
#include <json/json.h>

const static std::set<OperatorType> NoQuantOP = {OT_HSwish, OT_HSigmoid, OT_Sigmoid, OT_Clip,
    OT_Gelu, OT_TanH, OT_Resize, OT_LayerNorm, OT_Deconvolution, OT_HSwishNoDiv, OT_Eltwise,
    OT_Softmax, OT_DetectionOutput, OT_Scale, OT_SharedWeight, OT_Concat, OT_Swish};
const static std::set<OperatorType> QuantOP = {OT_Conv, OT_MatMul, OT_FC};
const static std::set<OperatorType> IntegerOP = {OT_Gather, OT_Embedding};
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
            if (spec->ops[i].feature_scale[0].scale[0] == 0) {
                continue;
            }
            for (unsigned k = 0; k < spec->ops[i].num_inputs; ++k) {
                std::string curIn = spec->ops[i].input_tensors_name[k];
                std::vector<std::pair<int, int>> prevIndex =
                    searchOperatorIndexByOutput(spec, curIn, 0, i);

                int prevNumQuant = 0;
                int prevOutputScale = 0;
                if (!prevIndex.empty()) {
                    prevNumQuant = spec->ops[prevIndex[0].first].num_quant_feature;
                    prevOutputScale =
                        spec->ops[prevIndex[0].first].feature_scale[prevNumQuant - 1].scale[0];
                }
                if (IntegerOP.count(spec->ops[i].type)) {
                    continue;
                }
                if (prevOutputScale == 0 || prevOutputScale == -2) {
                    std::string quantizeName = "Quantize_" + curIn + std::to_string(i);
                    insertQuantizeLinearKernel(spec, i, i, k, curIn, quantizeName);
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
            if (spec->ops[nextId].num_quant_feature > 0) {
                return std::make_pair(nextId, paths);
            }
            paths.push_back(nextId);
            if (nextIndex.size() > 1 || spec->ops[nextId].num_outputs > 1) {
                return std::make_pair(-2, paths);
            }
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
        return ((op->type == OT_Conv) && (op->ps.conv_spec.num_outputs == op->ps.conv_spec.group));
    }

    bool isNoQuantOp(OperatorSpec *op)
    {
        return (NoQuantOP.count(op->type) || isDepthWiseConv(op) || isAvgPooling(op) ||
            isNotNaiveRelu(op));
    }

    bool isQuantOp(OperatorSpec *op)
    {
        return (QuantOP.count(op->type) && !isNoQuantOp(op));
    }

    void parseAndPreLabel(ModelSpec *spec)
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
            if (isNoQuantOp(spec->ops + i)) {
                label_OP_as_no_quant(spec->ops + i);
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
        float scale = -2;
        for (auto index : nextIndex) {
            float tmp = 0;
            if (spec->ops[index.first].num_quant_feature == 0) {
                tmp = setToTheNextScale(spec, index.first);
            } else {
                tmp = spec->ops[index.first]
                          .feature_scale[spec->ops[index.first].num_quant_feature - 1]
                          .scale[0];
                if (tmp != 0) {
                    tmp = -1;
                }
            }
            if (scale != -2 && tmp != scale) {
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
                if (nextIndex.empty()) {
                    flag = -2;
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
        if (OT_Conv == ot || OT_FC == ot || OT_MatMul == ot) {
            return false;
        }
        for (U32 i = 0; i < spec->ops[prevIndex].num_inputs; i++) {
            if (!is_kin_to_model_input(spec, spec->ops[prevIndex].input_tensors_name[i], prevIndex)) {
                return false;
            }
        }
        return true;
    }

    bool optimizeNormal(ModelSpec *spec)
    {
        // Label the key quant and no-quant OP
        parseAndPreLabel(spec);

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

    bool optimizeQATWithScale(ModelSpec *spec)
    {
        std::fstream file(std::string(this->scaleFile), std::ios::in);
        Json::Value value;
        Json::Reader reader;
        if (!reader.parse(file, value)) {
            UNI_ERROR_LOG("%s is not a valid JSON file.", scaleFile);
        }
        file.close();

        parseAndPreLabel(spec);

        for (I32 i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_None) {
                continue;
            }
            if (spec->ops[i].num_quant_feature == 1 && spec->ops[i].feature_scale[0].scale[0] == 0) {
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
            for (U32 j = 0; j < inputNum; j++) {
                scales.push_back({-2});
            }
            for (U32 j = 0; j < outputNum; j++) {
                scales.push_back({-2});
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
                CHECK_REQUIREMENT(value[layerName]["weights"][members[0]].isDouble());
                clip_weight(spec, i, value[layerName]["weights"][members[0]].asFloat());
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
        if (this->scaleFile != nullptr) {
            optimizeQATWithScale(spec);
        } else if (this->clipVal > 0) {
            optimizeQATWithGlobalClip(spec);
        } else if (this->actFP) {
            optimizeActFP(spec);
        } else {
            optimizeNormal(spec);
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
};
#endif
