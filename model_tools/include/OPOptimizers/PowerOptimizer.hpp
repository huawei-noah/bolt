// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_POWEROPTIMIZER
#define _H_POWEROPTIMIZER

#include "OPOptimizer.hpp"

class PowerOptimizer : public OPOptimizer {
    // scale = 1, shift = 0, pow = 1
    bool optimizeUnusedPower(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Power &&
                UNI_ABS(spec->ops[i].ps.power_spec.scale - 1) < eps &&
                UNI_ABS(spec->ops[i].ps.power_spec.shift) < eps &&
                UNI_ABS(spec->ops[i].ps.power_spec.power - 1) < eps) {
                setOperatorInvalid(spec, i, true);
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }

    bool optimizePowerPower(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Power &&
                UNI_ABS(spec->ops[i].ps.power_spec.power - 1) < eps) {
                int powerIndex0 = i;
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[powerIndex0].output_tensors_name[0], powerIndex0 + 1,
                    spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 || OT_Power != spec->ops[nextOpIndexes[0].first].type ||
                    UNI_ABS(spec->ops[nextOpIndexes[0].first].ps.power_spec.power - 1) >= eps) {
                    continue;
                }
                int powerIndex1 = nextOpIndexes[0].first;
                float scale = spec->ops[powerIndex1].ps.power_spec.scale *
                    spec->ops[powerIndex0].ps.power_spec.scale;
                float shift = spec->ops[powerIndex1].ps.power_spec.scale *
                        spec->ops[powerIndex0].ps.power_spec.shift +
                    spec->ops[powerIndex1].ps.power_spec.shift;
                spec->ops[powerIndex1].ps.power_spec.scale = scale;
                spec->ops[powerIndex1].ps.power_spec.shift = shift;
                setOperatorInvalid(spec, i, true);
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }

    // power + shared weight
    bool optimizePowerEltwise(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 2; i++) {
            if (spec->ops[i].type == OT_Power && UNI_ABS(spec->ops[i].ps.power_spec.shift) < eps &&
                UNI_ABS(spec->ops[i].ps.power_spec.power - 1) < eps) {
                int powerIndex0 = i;
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[powerIndex0].output_tensors_name[0], powerIndex0 + 1,
                    spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 ||
                    OT_Eltwise != spec->ops[nextOpIndexes[0].first].type ||
                    spec->ops[nextOpIndexes[0].first].num_inputs != 2 ||
                    spec->ops[nextOpIndexes[0].first].ps.eltwise_spec.mode != ELTWISE_SUM) {
                    continue;
                }
                int eltwiseIndex = nextOpIndexes[0].first;

                std::vector<std::pair<int, int>> prevOpIndexes = searchOperatorIndexByOutput(spec,
                    spec->ops[eltwiseIndex].input_tensors_name[1 - nextOpIndexes[0].second], 0,
                    eltwiseIndex);
                if (prevOpIndexes.size() != 1 || OT_Power != spec->ops[prevOpIndexes[0].first].type ||
                    std::string(spec->ops[powerIndex0].input_tensors_name[0]) !=
                        std::string(spec->ops[prevOpIndexes[0].first].input_tensors_name[0]) ||
                    UNI_ABS(spec->ops[prevOpIndexes[0].first].ps.power_spec.shift) >= eps ||
                    UNI_ABS(spec->ops[prevOpIndexes[0].first].ps.power_spec.power - 1) >= eps ||
                    UNI_ABS(spec->ops[powerIndex0].ps.power_spec.scale +
                        spec->ops[prevOpIndexes[0].first].ps.power_spec.scale - 1) >= eps) {
                    continue;
                }
                int powerIndex1 = prevOpIndexes[0].first;
                nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[powerIndex1].output_tensors_name[0], powerIndex1 + 1,
                    spec->num_operator_specs);
                if (nextOpIndexes.size() != 1) {
                    continue;
                }

                setOperatorInvalid(spec, powerIndex0, true);
                setOperatorInvalid(spec, powerIndex1, true);
                if (spec->ops[eltwiseIndex].ps.eltwise_spec.activation_type == ACTIVATION_NULL) {
                    setOperatorInvalid(spec, eltwiseIndex, true);
                } else if (spec->ops[eltwiseIndex].ps.eltwise_spec.activation_type ==
                    ACTIVATION_RELU) {
                    spec->ops[eltwiseIndex].num_inputs = 1;
                    str_copy(spec->ops[eltwiseIndex].input_tensors_name[0],
                        spec->ops[i].input_tensors_name[0], NAME_LEN);
                    mt_free(spec->ops[eltwiseIndex].input_tensors_name[1]);
                    ReLUParamSpec reluParam =
                        spec->ops[eltwiseIndex].ps.eltwise_spec.activation_spec.relu_spec;
                    spec->ops[eltwiseIndex].ps.relu_spec = reluParam;
                } else {
                    UNI_ERROR_LOG("not support not-relu eltwise to merge with Power\n");
                }
                i = eltwiseIndex;
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }

    bool optimizeSquareSqrt(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Power && spec->ops[i].ps.power_spec.shift == 0) {
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 || OT_Power != spec->ops[nextOpIndexes[0].first].type) {
                    continue;
                }
                int nextPower = nextOpIndexes[0].first;
                if (spec->ops[nextPower].ps.power_spec.scale == 1 &&
                    spec->ops[nextPower].ps.power_spec.shift == 0) {
                    spec->ops[i].ps.power_spec.power *= spec->ops[nextPower].ps.power_spec.power;
                    setOperatorInvalid(spec, nextPower, true);
                    i = nextPower;
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }

    bool optimizeSquare(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Eltwise && spec->ops[i].num_inputs == 2 &&
                spec->ops[i].ps.eltwise_spec.mode == ELTWISE_PROD &&
                std::string(spec->ops[i].input_tensors_name[0]) ==
                    spec->ops[i].input_tensors_name[1]) {
                spec->ops[i].type = OT_Power;
                spec->ops[i].ps.power_spec.scale = 1;
                spec->ops[i].ps.power_spec.shift = 0;
                spec->ops[i].ps.power_spec.power = 2;
                spec->ops[i].num_inputs = 1;
                mt_free(spec->ops[i].input_tensors_name[1]);
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }

    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        hasOptimized |= optimizeUnusedPower(spec);
        hasOptimized |= optimizeSquareSqrt(spec);
        hasOptimized |= optimizePowerPower(spec);
        hasOptimized |= optimizeUnusedPower(spec);
        hasOptimized |= optimizePowerEltwise(spec);
        hasOptimized |= optimizeSquare(spec);
        return hasOptimized;
    }

private:
    float eps = 1e-16;
};
#endif
