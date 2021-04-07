// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_HSwishOPTIMIZER
#define _H_HSwishOPTIMIZER

#include "OPOptimizer.hpp"

class HSwishOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs - 2; i++) {
            if (spec->ops[i].type == OT_Relu6) {
                int relu6Index = i;
                int add3Index = relu6Index - 1;
                int mulIndex = relu6Index + 1;
                if (spec->ops[add3Index].type == OT_Power &&
                    spec->ops[add3Index].ps.power_spec.scale == 1 &&
                    spec->ops[add3Index].ps.power_spec.shift == 3 &&
                    spec->ops[add3Index].ps.power_spec.power == 1 &&
                    std::string(spec->ops[add3Index].output_tensors_name[0]) ==
                        std::string(spec->ops[relu6Index].input_tensors_name[0])) {
                    auto tmpVec = searchOperatorIndexByInput(spec,
                        spec->ops[add3Index].output_tensors_name[0], add3Index + 1,
                        spec->num_operator_specs);
                    if (tmpVec.size() != 1) {
                        continue;
                    }
                    if (spec->ops[mulIndex].type == OT_Eltwise &&
                        spec->ops[mulIndex].ps.eltwise_spec.elt_mode == ELTWISE_PROD &&
                        spec->ops[mulIndex].num_inputs == 2 &&
                        ((std::string(spec->ops[add3Index].input_tensors_name[0]) ==
                                 std::string(spec->ops[mulIndex].input_tensors_name[0]) &&
                             std::string(spec->ops[relu6Index].output_tensors_name[0]) ==
                                 std::string(spec->ops[mulIndex].input_tensors_name[1])) ||
                            (std::string(spec->ops[add3Index].input_tensors_name[0]) ==
                                    std::string(spec->ops[mulIndex].input_tensors_name[1]) &&
                                std::string(spec->ops[relu6Index].output_tensors_name[0]) ==
                                    std::string(spec->ops[mulIndex].input_tensors_name[0])))) {
                        auto tmpVec = searchOperatorIndexByInput(spec,
                            spec->ops[relu6Index].output_tensors_name[0], relu6Index + 1,
                            spec->num_operator_specs);
                        if (tmpVec.size() != 1) {
                            continue;
                        }
                        tmpVec = searchOperatorIndexByInput(spec,
                            spec->ops[mulIndex].output_tensors_name[0], mulIndex + 1,
                            spec->num_operator_specs);
                        if (tmpVec.size() == 1 && spec->ops[tmpVec[0].first].type == OT_Power &&
                            UNI_ABS(spec->ops[tmpVec[0].first].ps.power_spec.scale - 1 / 6.0) <
                                0.0001 &&
                            spec->ops[tmpVec[0].first].ps.power_spec.shift == 0 &&
                            spec->ops[tmpVec[0].first].ps.power_spec.power == 1) {
                            int div6Index = tmpVec[0].first;
                            memcpy(spec->ops[div6Index].input_tensors_name[0],
                                spec->ops[add3Index].input_tensors_name[0], NAME_LEN);
                            spec->ops[div6Index].type = OT_HSwish;
                            setOperatorInvalid(spec, mulIndex);
                        } else {
                            spec->ops[mulIndex].num_inputs = 1;
                            delete spec->ops[mulIndex].input_tensors_name[1];
                            memcpy(spec->ops[mulIndex].input_tensors_name[0],
                                spec->ops[add3Index].input_tensors_name[0], NAME_LEN);
                            spec->ops[mulIndex].type = OT_HSwishNoDiv;
                        }
                        setOperatorInvalid(spec, add3Index);
                        setOperatorInvalid(spec, relu6Index);
                        hasOptimized = true;
                    }
                }
            }
        }
        return hasOptimized;
    }
};
#endif
