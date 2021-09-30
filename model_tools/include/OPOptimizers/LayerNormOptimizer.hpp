// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_LayerNormOPTIMIZER
#define _H_LayerNormOPTIMIZER

#include "OPOptimizer.hpp"

class LayerNormOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 6; i++) {
            int k = i;
            int reduce_mean = 0;
            if (k < spec->num_operator_specs && spec->ops[k].type == OT_Reduction &&
                spec->ops[k].ps.reduction_spec.reduction_mode == REDUCTION_MEAN) {
                if (spec->ops[k].ps.reduction_spec.axes_num != 1) {
                    continue;
                }
                reduce_mean++;
                k++;
            }
            if (k < spec->num_operator_specs && spec->ops[k].type == OT_Pooling &&
                spec->ops[k].ps.pooling_spec.mode == POOLING_MEAN) {
                reduce_mean++;
                k++;
            }
            if (reduce_mean != 1) {
                continue;
            }
            // var = sum(x - u) ^ 2 / n
            if (k < spec->num_operator_specs && spec->ops[k].type == OT_Eltwise &&
                spec->ops[k].num_inputs == 2 &&
                spec->ops[k].ps.eltwise_spec.elt_mode == ELTWISE_SUB) {
                k++;
                k = skipInvalidOperator(spec, k);
                if (k < spec->num_operator_specs && spec->ops[k].type == OT_Cast) {
                    k++;
                }
                if (k < spec->num_operator_specs && spec->ops[k].type == OT_Power &&
                    spec->ops[k].ps.power_spec.scale == 1 &&
                    spec->ops[k].ps.power_spec.shift == 0 && spec->ops[k].ps.power_spec.power == 2) {
                    k++;
                } else {
                    continue;
                }
                if (k < spec->num_operator_specs && spec->ops[k].type == OT_Reduction &&
                    spec->ops[k].ps.reduction_spec.reduction_mode == REDUCTION_MEAN) {
                    if (spec->ops[k].ps.reduction_spec.axes_num != 1) {
                        continue;
                    }
                    reduce_mean++;
                    k++;
                }
                if (k < spec->num_operator_specs && spec->ops[k].type == OT_Pooling &&
                    spec->ops[k].ps.pooling_spec.mode == POOLING_MEAN) {
                    reduce_mean++;
                    k++;
                }
                if (reduce_mean != 2) {
                    continue;
                }
                // var = sum(x ^ 2) / n  - u ^ 2
            } else if (k < spec->num_operator_specs && spec->ops[k].type == OT_Power &&
                spec->ops[k].ps.power_spec.scale == 1 && spec->ops[k].ps.power_spec.shift == 0 &&
                spec->ops[k].ps.power_spec.power == 2) {
                k++;
                if (k < spec->num_operator_specs && spec->ops[k].type == OT_Reduction &&
                    spec->ops[k].ps.reduction_spec.reduction_mode == REDUCTION_MEAN) {
                    if (spec->ops[k].ps.reduction_spec.axes_num != 1) {
                        continue;
                    }
                    reduce_mean++;
                    k++;
                }
                if (k < spec->num_operator_specs && spec->ops[k].type == OT_Pooling &&
                    spec->ops[k].ps.pooling_spec.mode == POOLING_MEAN) {
                    reduce_mean++;
                    k++;
                }
                if (reduce_mean != 2) {
                    continue;
                }
                if (k < spec->num_operator_specs && spec->ops[k].type == OT_Power &&
                    spec->ops[k].ps.power_spec.scale == 1 &&
                    spec->ops[k].ps.power_spec.shift == 0 && spec->ops[k].ps.power_spec.power == 2) {
                    k++;
                } else {
                    continue;
                }
                if (k < spec->num_operator_specs && spec->ops[k].type == OT_Eltwise &&
                    spec->ops[k].num_inputs == 2 &&
                    spec->ops[k].ps.eltwise_spec.elt_mode == ELTWISE_SUB) {
                    k++;
                } else {
                    continue;
                }
                if (k < spec->num_operator_specs && spec->ops[k].type == OT_Eltwise &&
                    spec->ops[k].num_inputs == 2 &&
                    spec->ops[k].ps.eltwise_spec.elt_mode == ELTWISE_SUB) {
                    k++;
                } else {
                    continue;
                }
            } else {
                continue;
            }
            k = skipInvalidOperator(spec, k);
            if (k < spec->num_operator_specs && spec->ops[k].type == OT_Power &&
                spec->ops[k].ps.power_spec.scale == 1 && spec->ops[k].ps.power_spec.power == 1 &&
                UNI_ABS(spec->ops[k].ps.power_spec.shift) < 0.0001) {
                k++;
            }
            if (k < spec->num_operator_specs && spec->ops[k].type == OT_Power &&
                spec->ops[k].ps.power_spec.scale == 1 && spec->ops[k].ps.power_spec.shift == 0 &&
                UNI_ABS(spec->ops[k].ps.power_spec.power) == 0.5) {
                k++;
            } else {
                continue;
            }
            k = skipInvalidOperator(spec, k);
            if (k < spec->num_operator_specs && spec->ops[k].type == OT_Eltwise &&
                spec->ops[k].num_inputs == 2 &&
                spec->ops[k].ps.eltwise_spec.elt_mode == ELTWISE_DIV) {
                k++;
                k = skipInvalidOperator(spec, k);
                if (k < spec->num_operator_specs && spec->ops[k].type == OT_Scale) {
                    for (int j = i; j < k; j++) {
                        setOperatorInvalid(spec, j, true);
                    }
                    spec->ops[k].type = OT_LayerNorm;
                    i = k;
                    hasOptimized = true;
                }
            } else if (k < spec->num_operator_specs - 4 && spec->ops[k].type == OT_Scale &&
                spec->ops[k + 1].type == OT_Eltwise && spec->ops[k + 2].type == OT_Eltwise &&
                spec->ops[k + 2].ps.eltwise_spec.elt_mode == ELTWISE_SUB &&
                spec->ops[k + 3].type == OT_Eltwise && spec->ops[k + 4].type == OT_Eltwise) {
                char *subWeightName = spec->ops[k + 2].input_tensors_name[0];
                int scaleWeightIndex = searchWeightIndex(spec, spec->ops[k].name);
                int subWeightIndex = searchWeightIndex(spec, subWeightName);
                CHECK_REQUIREMENT(scaleWeightIndex >= 0);
                CHECK_REQUIREMENT(subWeightIndex >= 0);
                if (spec->ws[scaleWeightIndex].bytes_of_vec != 0) {
                    continue;
                }
                int sub0Index;
                for (sub0Index = 0; sub0Index < i; sub0Index++) {
                    if (spec->ops[sub0Index].name == std::string(subWeightName)) {
                        break;
                    }
                }
                spec->ws[scaleWeightIndex].bytes_of_vec = spec->ws[subWeightIndex].bytes_of_weight;
                spec->ws[scaleWeightIndex].vec =
                    (U8 *)mt_new_storage(spec->ws[scaleWeightIndex].bytes_of_vec);
                memcpy(spec->ws[scaleWeightIndex].vec, spec->ws[subWeightIndex].weight,
                    spec->ws[scaleWeightIndex].bytes_of_vec);

                for (int j = i; j < k; j++) {
                    setOperatorInvalid(spec, j, true);
                }
                strcpy(spec->ops[k].output_tensors_name[0], spec->ops[k + 4].output_tensors_name[0]);
                for (int j = 1; j < 5; j++) {
                    setOperatorInvalid(spec, k + j, false);
                }
                setOperatorInvalid(spec, sub0Index, true);
                spec->ops[k].type = OT_LayerNorm;
                i = k + 5;
                hasOptimized = true;
            } else {
            }
        }
        return hasOptimized;
    }
};
#endif
