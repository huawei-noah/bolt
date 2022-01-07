// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <map>
#include <sstream>
#include <fstream>
#include <json/json.h>
#include "model_quantization.h"
#include "model_common.h"
#include "OPOptimizers/OPOptimizer.hpp"

void add_scale_from_file(ModelSpec *ms, const char *scaleFile)
{
    std::fstream file(std::string(scaleFile), std::ios::in);
    Json::Value value;
    Json::Reader reader;
    if (!reader.parse(file, value)) {
        UNI_ERROR_LOG("%s is not a valid JSON file.", scaleFile);
    }
    file.close();

    for (I32 i = 0; i < ms->num_operator_specs; i++) {
        if (isDeprecatedOp(ms->ops[i].type)) {
            continue;
        }
        if (ms->ops[i].num_quant_feature == 1 && ms->ops[i].feature_scale[0].scale[0] == 0) {
            UNI_WARNING_LOG("%s cannot be quantized.\n", ms->ops[i].name);
            continue;
        }
        std::string layerName = std::string(ms->ops[i].name);

        // only quantize the layer in the scale file
        if (!value[layerName].isObject()) {
            CHECK_REQUIREMENT(ms->ops[i].num_quant_feature == 0);
            ms->ops[i].num_quant_feature = 1;
            ms->ops[i].feature_scale = (QuantSpec *)mt_new_storage(sizeof(QuantSpec));
            ms->ops[i].feature_scale[0].num_scale = 1;
            ms->ops[i].feature_scale[0].scale = (F32 *)mt_new_storage(sizeof(F32));
            ms->ops[i].feature_scale[0].scale[0] = 0;
            continue;
        }

        std::vector<std::vector<F32>> scales;

        // all nodes are set to F32 default
        U32 inputNum = ms->ops[i].num_inputs;
        U32 outputNum = ms->ops[i].num_outputs;
        for (U32 j = 0; j < inputNum; j++) {
            scales.push_back({-2});
        }
        for (U32 j = 0; j < outputNum; j++) {
            scales.push_back({-2});
        }
        if (value[layerName]["inputs"].isObject()) {
            for (U32 j = 0; j < inputNum; j++) {
                // only support 1 clip value now
                std::string inputName = std::string(ms->ops[i].input_tensors_name[j]);
                if (value[layerName]["inputs"][inputName].isDouble()) {
                    scales[j] = {127.0f / value[layerName]["inputs"][inputName].asFloat()};
                }
            }
        }

        if (value[layerName]["outputs"].isObject()) {
            for (U32 j = 0; j < outputNum; j++) {
                // only support 1 clip value now
                std::string outputName = std::string(ms->ops[i].output_tensors_name[j]);
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
            int weightIdx = OPOptimizer::searchWeightIndex(ms, ms->ops[i].name);
            CHECK_REQUIREMENT(-1 != weightIdx);
            CHECK_REQUIREMENT(DT_F32 == ms->ws[weightIdx].mdt);
            UNI_INFO_LOG("Clipping the weight of %s\n", ms->ops[i].name);
            F32 clipMax = value[layerName]["weights"][members[0]].asFloat();
            F32 clipMin = -1 * clipMax;
            U32 len = ms->ws[weightIdx].bytes_of_weight / bytesOf(DT_F32);
            F32 *w = (F32 *)mt_new_storage(ms->ws[weightIdx].bytes_of_weight);
            memcpy(w, ms->ws[weightIdx].weight, ms->ws[weightIdx].bytes_of_weight);
            for (U32 j = 0; j < len; j++) {
                if (w[j] > clipMax) {
                    w[j] = clipMax;
                } else if (w[j] < clipMin) {
                    w[j] = clipMin;
                }
            }
            if (ms->ws[weightIdx].weight != nullptr) {
                if (outOfFileMapRange(ms->ws[weightIdx].weight, ms->mfd)) {
                    delete ms->ws[weightIdx].weight;
                }
                ms->ws[weightIdx].weight = nullptr;
            }
            ms->ws[weightIdx].weight = (U8 *)w;
        }

        // Store scales into result model
        if (nullptr != ms->ops[i].feature_scale) {  // Could be labelled with -2
            for (U32 k = 0; k < ms->ops[i].num_quant_feature; k++) {
                if (nullptr != ms->ops[i].feature_scale[k].scale) {
                    delete ms->ops[i].feature_scale[k].scale;
                }
            }
            delete ms->ops[i].feature_scale;
        }

        ms->ops[i].num_quant_feature = scales.size();
        ms->ops[i].feature_scale = (QuantSpec *)mt_new_storage(scales.size() * sizeof(QuantSpec));

        for (U32 k = 0; k < scales.size(); k++) {
            ms->ops[i].feature_scale[k].num_scale = scales[k].size();
            U32 scaleBytes = scales[k].size() * sizeof(F32);
            ms->ops[i].feature_scale[k].scale = (F32 *)mt_new_storage(scaleBytes);
            memcpy(ms->ops[i].feature_scale[k].scale, scales[k].data(), scaleBytes);
        }
    }
}
