// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_WEIGHTSCALEOPTIMIZER
#define _H_WEIGHTSCALEOPTIMIZER

#include "OPOptimizer.hpp"

class WeightScaleOptimizer : public OPOptimizer {
public:
    WeightScaleOptimizer(bool PTQ = false)
    {
        this->ops = {OT_Scale};
        if (!PTQ) {
            this->ops.insert(OT_Conv);
            this->ops.insert(OT_FC);
        }
    }

    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        hasOptimized |= optimize_power(spec);
        hasOptimized |= optimize_scale(spec);
        hasOptimized |= optimize_power_scale(spec);
        return hasOptimized;
    }

    bool optimize_power(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (OT_Conv == spec->ops[i].type || OT_FC == spec->ops[i].type ||
                OT_Scale == spec->ops[i].type) {
                int prevOpIndex = i;
                if (OT_Conv == spec->ops[prevOpIndex].type) {
                    if (ACTIVATION_NULL != spec->ops[prevOpIndex].ps.conv_spec.dw_activation_type ||
                        ACTIVATION_NULL != spec->ops[prevOpIndex].ps.conv_spec.pw_activation_type) {
                        continue;
                    }
                }
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[prevOpIndex].output_tensors_name[0], prevOpIndex + 1,
                    spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 || OT_Power != spec->ops[nextOpIndexes[0].first].type) {
                    continue;
                }
                int powerOpIndex = nextOpIndexes[0].first;
                if (spec->ops[powerOpIndex].ps.power_spec.power != 1) {
                    UNI_WARNING_LOG(
                        "encounter unoptimize Power layer(pow > 1): %s\n", spec->ops[i].name);
                    continue;
                }

                int convWeightIndex = searchWeightIndex(spec, spec->ops[prevOpIndex].name);
                CHECK_REQUIREMENT(convWeightIndex >= 0);
                if (spec->ws[convWeightIndex].mdt == DT_BIN01 ||
                    spec->ws[convWeightIndex].mdt == DT_BIN11) {
                    continue;
                }

                if (spec->ws[convWeightIndex].weight == nullptr ||
                    spec->ws[convWeightIndex].bytes_of_weight == 0) {
                    spec->ws[convWeightIndex].bytes_of_weight =
                        spec->ws[convWeightIndex].bytes_of_vec;
                    spec->ws[convWeightIndex].weight =
                        (U8 *)mt_malloc(spec->ws[convWeightIndex].bytes_of_weight);
                    F32 *ptr = (F32 *)spec->ws[convWeightIndex].weight;
                    for (U32 m = 0; m < spec->ws[convWeightIndex].bytes_of_weight /
                             bytesOf(spec->ws[convWeightIndex].mdt);
                         m++) {
                        ptr[m] = 1;
                    }
                }
                F32 *weightTemp = (F32 *)mt_malloc(spec->ws[convWeightIndex].bytes_of_weight);
                UNI_MEMCPY(weightTemp, spec->ws[convWeightIndex].weight,
                    spec->ws[convWeightIndex].bytes_of_weight);
                if (spec->ws[convWeightIndex].vec == nullptr ||
                    spec->ws[convWeightIndex].bytes_of_vec == 0) {
                    if (OT_Conv == spec->ops[i].type) {
                        spec->ws[convWeightIndex].bytes_of_vec =
                            spec->ops[i].ps.conv_spec.num_outputs * sizeof(F32);
                    } else if (OT_FC == spec->ops[i].type) {
                        spec->ws[convWeightIndex].bytes_of_vec =
                            spec->ops[i].ps.fc_spec.num_outputs * sizeof(F32);
                    } else if (OT_Scale == spec->ops[i].type) {
                        spec->ws[convWeightIndex].bytes_of_vec =
                            spec->ws[convWeightIndex].bytes_of_weight;
                    } else {
                        continue;
                    }
                    spec->ws[convWeightIndex].vec =
                        (U8 *)mt_malloc(spec->ws[convWeightIndex].bytes_of_vec);
                    UNI_MEMSET(
                        spec->ws[convWeightIndex].vec, 0, spec->ws[convWeightIndex].bytes_of_vec);
                }
                F32 *vecTemp = (F32 *)mt_malloc(spec->ws[convWeightIndex].bytes_of_vec);
                UNI_MEMCPY(
                    vecTemp, spec->ws[convWeightIndex].vec, spec->ws[convWeightIndex].bytes_of_vec);
                for (U32 m = 0; m < spec->ws[convWeightIndex].bytes_of_weight /
                         bytesOf(spec->ws[convWeightIndex].mdt);
                     m++) {
                    weightTemp[m] *= spec->ops[powerOpIndex].ps.power_spec.scale;
                }
                for (U32 m = 0; m < spec->ws[convWeightIndex].bytes_of_vec /
                         bytesOf(spec->ws[convWeightIndex].mdt);
                     m++) {
                    vecTemp[m] = vecTemp[m] * spec->ops[powerOpIndex].ps.power_spec.scale +
                        spec->ops[powerOpIndex].ps.power_spec.shift;
                }

                mt_free(spec->ws[convWeightIndex].vec, spec);
                mt_free(spec->ws[convWeightIndex].weight, spec);

                spec->ws[convWeightIndex].vec = (U8 *)vecTemp;
                spec->ws[convWeightIndex].weight = (U8 *)weightTemp;

                setOperatorInvalid(spec, powerOpIndex, true);
                hasOptimized = true;
                i--;
            }
        }
        return hasOptimized;
    }

    bool optimize_scale(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (this->ops.find(spec->ops[i].type) != this->ops.end()) {
                int prevOpIndex = i;
                if (OT_Conv == spec->ops[prevOpIndex].type) {
                    if (ACTIVATION_NULL != spec->ops[prevOpIndex].ps.conv_spec.dw_activation_type ||
                        ACTIVATION_NULL != spec->ops[prevOpIndex].ps.conv_spec.pw_activation_type) {
                        continue;
                    }
                }
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[prevOpIndex].output_tensors_name[0], prevOpIndex + 1,
                    spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 || OT_Scale != spec->ops[nextOpIndexes[0].first].type) {
                    continue;
                }
                int scaleOpIndex = nextOpIndexes[0].first;
                if (spec->ops[scaleOpIndex].num_inputs > 1) {
                    UNI_WARNING_LOG(
                        "encounter unoptimize Scale layer(multi-inputs): %s\n", spec->ops[i].name);
                    continue;
                }

                // scale
                int scaleWeightIndex = searchWeightIndex(spec, spec->ops[scaleOpIndex].name);
                CHECK_REQUIREMENT(scaleWeightIndex >= 0);
                CHECK_REQUIREMENT(spec->ws[scaleWeightIndex].mdt == DT_F32);
                U32 channelAlpha = spec->ws[scaleWeightIndex].bytes_of_weight /
                    bytesOf(spec->ws[scaleWeightIndex].mdt);
                U32 channelBeta = spec->ws[scaleWeightIndex].bytes_of_vec /
                    bytesOf(spec->ws[scaleWeightIndex].mdt);
                U32 channelCur = UNI_MAX(channelAlpha, channelBeta);
                F32 *alphaPtr = (F32 *)spec->ws[scaleWeightIndex].weight;
                F32 *betaPtr = (F32 *)spec->ws[scaleWeightIndex].vec;

                if (spec->ws[scaleWeightIndex].bytes_of_weight == 4 ||
                    spec->ws[scaleWeightIndex].bytes_of_vec == 4) {
                    continue;
                }

                int convWeightIndex = searchWeightIndex(spec, spec->ops[prevOpIndex].name);
                CHECK_REQUIREMENT(convWeightIndex >= 0);
                // mdt can now be DT_BIN01 or DT_BIN11
                U32 isBNN = 0;
                if (spec->ws[convWeightIndex].mdt == DT_BIN01 ||
                    spec->ws[convWeightIndex].mdt == DT_BIN11) {
                    isBNN = 1;
                }

                // scale + scale
                if (spec->ws[convWeightIndex].weight == nullptr ||
                    spec->ws[convWeightIndex].bytes_of_weight == 0) {
                    spec->ws[convWeightIndex].bytes_of_weight = channelCur * sizeof(F32);
                    spec->ws[convWeightIndex].weight =
                        (U8 *)mt_malloc(spec->ws[convWeightIndex].bytes_of_weight);
                    F32 *ptr = (F32 *)spec->ws[convWeightIndex].weight;
                    for (U32 m = 0; m < channelCur; m++) {
                        ptr[m] = 1;
                    }
                }
                F32 *weightTemp = (F32 *)mt_malloc(spec->ws[convWeightIndex].bytes_of_weight);
                UNI_MEMCPY(weightTemp, spec->ws[convWeightIndex].weight,
                    spec->ws[convWeightIndex].bytes_of_weight);
                if (spec->ws[convWeightIndex].vec == nullptr) {
                    spec->ws[convWeightIndex].bytes_of_vec = channelCur * sizeof(F32);
                    if (isBNN == 1) {
                        spec->ws[convWeightIndex].bytes_of_vec *= 2;
                    }
                    spec->ws[convWeightIndex].vec =
                        (U8 *)mt_malloc(spec->ws[convWeightIndex].bytes_of_vec);
                    if (isBNN == 1) {
                        F32 *scale = (F32 *)spec->ws[convWeightIndex].vec;
                        F32 *bias = scale + channelCur;
                        for (U32 m = 0; m < channelCur; m++) {
                            scale[m] = 1;
                            bias[m] = 0;
                        }
                    } else {
                        UNI_MEMSET(spec->ws[convWeightIndex].vec, 0,
                            spec->ws[convWeightIndex].bytes_of_vec);
                    }
                }
                F32 *vecTemp = (F32 *)mt_malloc(spec->ws[convWeightIndex].bytes_of_vec);
                UNI_MEMCPY(
                    vecTemp, spec->ws[convWeightIndex].vec, spec->ws[convWeightIndex].bytes_of_vec);
                if (isBNN == 1) {
                    F32 *scale = vecTemp;
                    F32 *bias = vecTemp + channelCur;
                    for (U32 m = 0; m < channelCur; m++) {
                        if (alphaPtr != nullptr) {
                            scale[m] *= alphaPtr[m];
                            bias[m] *= alphaPtr[m];
                        }
                        if (betaPtr != nullptr) {
                            bias[m] += betaPtr[m];
                        }
                    }
                } else {
                    int weightDataSize = spec->ws[convWeightIndex].bytes_of_weight /
                        bytesOf(spec->ws[convWeightIndex].mdt);
                    int weightPerChannel = weightDataSize / channelCur;
                    // NCHW
                    for (U32 m = 0; m < channelCur; m++) {
                        F32 *convWeightPerChannel = weightTemp + weightPerChannel * m;
                        if (alphaPtr != nullptr) {
                            for (int n = 0; n < weightPerChannel; n++) {
                                convWeightPerChannel[n] *= alphaPtr[m];
                            }
                            vecTemp[m] = alphaPtr[m] * vecTemp[m];
                        }
                        if (betaPtr != nullptr) {
                            vecTemp[m] += betaPtr[m];
                        }
                    }
                }

                mt_free(spec->ws[convWeightIndex].weight, spec);
                mt_free(spec->ws[convWeightIndex].vec, spec);
                spec->ws[convWeightIndex].weight = (U8 *)weightTemp;
                spec->ws[convWeightIndex].vec = (U8 *)vecTemp;

                // if this op is output op, we shoule keep the origin output name
                std::vector<int> outputIdxes = searchString(spec->output_names, spec->num_outputs,
                        spec->ops[scaleOpIndex].output_tensors_name[0]);

                setOperatorInvalid(spec, scaleOpIndex, true);

                if (!outputIdxes.empty()) {
                    std::string originOutput =
                        spec->ops[scaleOpIndex].output_tensors_name[0];
                    str_copy(spec->output_names[outputIdxes[0]],
                            originOutput.c_str(), NAME_LEN);
                    std::vector<std::pair<int, int>> operatorIndexes0 = searchOperatorIndexByInput(
                        spec, spec->ops[prevOpIndex].output_tensors_name[0], prevOpIndex + 1,
                        spec->num_operator_specs);
                    for (U32 j = 0; j < operatorIndexes0.size(); j++) {
                        str_copy(spec->ops[operatorIndexes0[j].first]
                                     .input_tensors_name[operatorIndexes0[j].second],
                            originOutput.c_str(), NAME_LEN);
                    }
                    str_copy(spec->ops[prevOpIndex].output_tensors_name[0],
                             originOutput.c_str(), NAME_LEN);

                }
                hasOptimized = true;
                i--;
            }
        }
        return hasOptimized;
    }

    bool optimize_power_scale(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (OT_Power == spec->ops[i].type && 1 == spec->ops[i].ps.power_spec.power) {
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 || OT_Scale != spec->ops[nextOpIndexes[0].first].type) {
                    continue;
                }
                int scaleOpIndex = nextOpIndexes[0].first;
                if (spec->ops[scaleOpIndex].num_inputs > 1) {
                    UNI_WARNING_LOG(
                        "encounter unoptimize Scale layer(multi-inputs): %s\n", spec->ops[i].name);
                    continue;
                }

                // scale
                int scaleWeightIndex = searchWeightIndex(spec, spec->ops[scaleOpIndex].name);
                CHECK_REQUIREMENT(scaleWeightIndex >= 0);
                CHECK_REQUIREMENT(spec->ws[scaleWeightIndex].mdt == DT_F32);
                U32 channelAlpha = spec->ws[scaleWeightIndex].bytes_of_weight /
                    bytesOf(spec->ws[scaleWeightIndex].mdt);
                U32 channelBeta = spec->ws[scaleWeightIndex].bytes_of_vec /
                    bytesOf(spec->ws[scaleWeightIndex].mdt);
                U32 channelCur = UNI_MAX(channelAlpha, channelBeta);
                F32 *alpha0 = (F32 *)spec->ws[scaleWeightIndex].weight;
                F32 *beta0 = (F32 *)spec->ws[scaleWeightIndex].vec;

                spec->ws[scaleWeightIndex].bytes_of_weight =
                    channelCur * bytesOf(spec->ws[scaleWeightIndex].mdt);
                spec->ws[scaleWeightIndex].bytes_of_vec = spec->ws[scaleWeightIndex].bytes_of_weight;
                spec->ws[scaleWeightIndex].weight =
                    (U8 *)mt_malloc(spec->ws[scaleWeightIndex].bytes_of_weight);
                spec->ws[scaleWeightIndex].vec =
                    (U8 *)mt_malloc(spec->ws[scaleWeightIndex].bytes_of_vec);
                F32 *alpha1 = (F32 *)spec->ws[scaleWeightIndex].weight;
                F32 *beta1 = (F32 *)spec->ws[scaleWeightIndex].vec;
                for (U32 m = 0; m < channelCur; m++) {
                    F32 beta = spec->ops[i].ps.power_spec.shift;
                    if (alpha0 == nullptr) {
                        alpha1[m] = spec->ops[i].ps.power_spec.scale;
                    } else {
                        alpha1[m] = alpha0[m] * spec->ops[i].ps.power_spec.scale;
                        beta *= alpha0[m];
                    }
                    if (beta0 == nullptr) {
                        beta1[m] = beta;
                    } else {
                        beta1[m] = beta + beta0[m];
                    }
                }
                mt_free(alpha0, spec);
                mt_free(beta0, spec);
                setOperatorInvalid(spec, i, true);
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }

private:
    std::set<OperatorType> ops;
};
#endif
