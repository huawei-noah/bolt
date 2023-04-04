// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ConstantFuseOPTIMIZER
#define _H_ConstantFuseOPTIMIZER

#include "OPOptimizer.hpp"

class ConstantFuseOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 2; i++) {
            if (spec->ops[i].type == OT_SharedWeight) {
                auto next =
                    searchOperatorIndexByInput(spec, spec->ops[i].output_tensors_name[0], i + 1);
                if (next.size() != 1) {
                    continue;
                }
                int ii = next[0].first;
                if (spec->ops[ii].type == OT_ConstantOfShape && spec->ops[ii].num_inputs == 1) {
                    auto next = searchOperatorIndexByInput(
                        spec, spec->ops[ii].output_tensors_name[0], ii + 1);
                    if (next.size() != 1) {
                        continue;
                    }
                    int j = next[0].first;
                    int wid = searchWeightIndex(spec, spec->ops[i].name);
                    if (!(spec->ws[wid].mdt == DT_I32 || spec->ws[wid].mdt == DT_U32)) {
                        continue;
                    }
                    I32 num_shape = *((I32 *)spec->ws[wid].weight);
                    I32 value = spec->ops[ii].ps.constant_of_shape_spec.value;
                    if (spec->ops[j].type == OT_Expand && next[0].second == 1) {
                        spec->ops[j].ps.expand_spec.num_shape = num_shape;
                        for (int k = 0; k < num_shape; k++) {
                            spec->ops[j].ps.expand_spec.shape[k] = value;
                        }
                        mt_free(spec->ops[j].input_tensors_name[1]);
                        spec->ops[j].num_inputs = 1;
                        setOperatorInvalid(spec, i);
                        setOperatorInvalid(spec, ii);
                        hasOptimized = true;
                    }
                }
            }
        }
        return hasOptimized;
    }
};
#endif
