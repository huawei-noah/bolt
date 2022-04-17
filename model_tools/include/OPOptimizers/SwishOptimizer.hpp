// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_SwishOPTIMIZER
#define _H_SwishOPTIMIZER

#include "OPOptimizer.hpp"

class SwishOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Sigmoid) {
                auto next = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (next.size() != 1) {
                    continue;
                }
                int eltId = next[0].first;
                int inputId = next[0].second;
                if (spec->ops[eltId].type == OT_Eltwise && spec->ops[eltId].num_inputs == 2 &&
                    spec->ops[eltId].ps.eltwise_spec.mode == ELTWISE_PROD &&
                    std::string(spec->ops[i].input_tensors_name[0]) ==
                        std::string(spec->ops[eltId].input_tensors_name[1 - inputId])) {
                    spec->ops[eltId].type = OT_Swish;

                    setOperatorInvalid(spec, i);
                    mt_free(spec->ops[eltId].input_tensors_name[inputId]);
                    spec->ops[eltId].num_inputs = 1;
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
