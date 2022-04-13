// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_HSigmoidOPTIMIZER
#define _H_HSigmoidOPTIMIZER

#include "OPOptimizer.hpp"

class HSigmoidOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Relu6) {
                int relu6Index = i;
                int add3Index = relu6Index - 1;
                int div6Index = relu6Index + 1;
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
                    if (spec->ops[div6Index].type == OT_Power &&
                        UNI_ABS(spec->ops[div6Index].ps.power_spec.scale - 1 / 6.0) < 0.0001 &&
                        spec->ops[div6Index].ps.power_spec.shift == 0 &&
                        spec->ops[div6Index].ps.power_spec.power == 1 &&
                        std::string(spec->ops[relu6Index].output_tensors_name[0]) ==
                            std::string(spec->ops[div6Index].input_tensors_name[0])) {
                        auto tmpVec = searchOperatorIndexByInput(spec,
                            spec->ops[relu6Index].output_tensors_name[0], relu6Index + 1,
                            spec->num_operator_specs);
                        if (tmpVec.size() != 1) {
                            continue;
                        }
                        UNI_MEMCPY(spec->ops[div6Index].input_tensors_name[0],
                            spec->ops[add3Index].input_tensors_name[0], NAME_LEN);
                        spec->ops[div6Index].type = OT_HSigmoid;
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
