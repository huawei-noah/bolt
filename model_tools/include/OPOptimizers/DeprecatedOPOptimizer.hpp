// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_DEPRECATEDOPOPTIMIZER
#define _H_DEPRECATEDOPOPTIMIZER

#include <map>
#include "OPOptimizer.hpp"

class DeprecatedOPOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;

        std::map<std::string, int> operatorMap, weightMap;
        for (int i = 0; i < spec->num_weight_specs; i++) {
            weightMap[spec->ws[i].op_name] = i;
        }
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Pad) {
                if (spec->ops[i].ps.pad_spec.before == 0 && spec->ops[i].ps.pad_spec.after == 0 &&
                    spec->ops[i].ps.pad_spec.top == 0 && spec->ops[i].ps.pad_spec.bottom == 0 &&
                    spec->ops[i].ps.pad_spec.left == 0 && spec->ops[i].ps.pad_spec.right == 0 &&
                    spec->ops[i].ps.pad_spec.front == 0 && spec->ops[i].ps.pad_spec.back == 0) {
                    setOperatorInvalid(spec, i, true);
                    hasOptimized = true;
                    continue;
                }
            }
            if (spec->ops[i].type == OT_Input) {
                spec->ops[i].type = OT_None;
                hasOptimized = true;
                continue;
            }
            if (spec->ops[i].type == OT_Pooling) {
                if (spec->ops[i].ps.pooling_spec.kernel_t == 1 &&
                    spec->ops[i].ps.pooling_spec.kernel_h == 1 &&
                    spec->ops[i].ps.pooling_spec.kernel_w == 1 &&
                    spec->ops[i].ps.pooling_spec.stride_t == 1 &&
                    spec->ops[i].ps.pooling_spec.stride_h == 1 &&
                    spec->ops[i].ps.pooling_spec.stride_w == 1 &&
                    spec->ops[i].ps.pooling_spec.padding_before == 0 &&
                    spec->ops[i].ps.pooling_spec.padding_after == 0 &&
                    spec->ops[i].ps.pooling_spec.padding_top == 0 &&
                    spec->ops[i].ps.pooling_spec.padding_bottom == 0 &&
                    spec->ops[i].ps.pooling_spec.padding_left == 0 &&
                    spec->ops[i].ps.pooling_spec.padding_right == 0) {
                    setOperatorInvalid(spec, i, true);
                    hasOptimized = true;
                    continue;
                }
            }
            if (spec->ops[i].type == OT_Split || spec->ops[i].type == OT_Dropout) {
                setOperatorInvalid(spec, i, true);
                hasOptimized = true;
                continue;
            }
            if (spec->ops[i].type == OT_Slice && spec->ops[i].num_inputs == 1 &&
                spec->ops[i].num_outputs == 1) {
                setOperatorInvalid(spec, i, true);
                hasOptimized = true;
                continue;
            }
            if (spec->ops[i].type == OT_Concat && spec->ops[i].num_inputs == 1) {
                setOperatorInvalid(spec, i, true);
                hasOptimized = true;
                continue;
            }
            if (spec->ops[i].type != OT_Input && spec->ops[i].type != OT_None &&
                spec->ops[i].type != OT_PreAllocatedMemory) {
                std::string key = std::string(OperatorTypeName()[spec->ops[i].type]);
                for (U32 j = 0; j < spec->ops[i].num_inputs; j++) {
                    key += spec->ops[i].input_tensors_name[j] + std::string(",");
                }
                key += copyBuffer<1024>(&(spec->ops[i].ps), sizeof(ParameterSpec));
                if (weightMap.find(spec->ops[i].name) != weightMap.end()) {
                    int weightId = weightMap[spec->ops[i].name];
                    if (spec->ws[weightId].bytes_of_weight > 0) {
                        key += copyBuffer<1024>(
                            spec->ws[weightId].weight, spec->ws[weightId].bytes_of_weight);
                    }
                    if (spec->ws[weightId].bytes_of_vec > 0) {
                        key += copyBuffer<1024>(
                            spec->ws[weightId].vec, spec->ws[weightId].bytes_of_vec);
                    }
                }

                if (operatorMap.find(key) == operatorMap.end()) {
                    operatorMap[key] = i;
                } else {
                    CHECK_REQUIREMENT(spec->ops[i].num_outputs == 1);
                    char *lastOut = spec->ops[operatorMap[key]].output_tensors_name[0];
                    std::string curOut = spec->ops[i].output_tensors_name[0];
                    auto nextOpIndexes = searchOperatorIndexByInput(
                        spec, curOut, i + 1, spec->num_operator_specs, false);

                    setOperatorInvalid(spec, i, false);
                    for (U32 j = 0; j < nextOpIndexes.size(); j++) {
                        str_copy(spec->ops[nextOpIndexes[j].first]
                                     .input_tensors_name[nextOpIndexes[j].second],
                            lastOut, NAME_LEN);
                    }
                    for (int j = 0; j < spec->num_outputs; j++) {
                        if (spec->output_names[j] == curOut) {
                            str_copy(spec->output_names[j], lastOut, NAME_LEN);
                        }
                    }
                    continue;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
