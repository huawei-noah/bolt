// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_SPLICEFCOPTIMIZER
#define _H_SPLICEFCOPTIMIZER

#include "OPOptimizer.hpp"

class SpliceFCOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (OT_Splice == spec->ops[i].type) {
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 || OT_FC != spec->ops[nextOpIndexes[0].first].type) {
                    continue;
                }
                int fcOpIndex = nextOpIndexes[0].first;
                if (spec->ops[fcOpIndex].ps.fc_spec.num_slices > 1) {
                    UNI_WARNING_LOG("encounter unoptimize Splice+FC layer(multi-outputs): %s\n",
                        spec->ops[i].name);
                    continue;
                }

                // delete splice weight(forward indexes)
                int spliceWeightIndex = searchWeightIndex(spec, spec->ops[i].name);
                CHECK_REQUIREMENT(spliceWeightIndex >= 0);
                setWeightOperatorInvalid(spec, spliceWeightIndex);

                TdnnParamSpec p;
                p.num_context = spec->ops[i].ps.splice_spec.num_context;
                for (int j = 0; j < p.num_context; j++) {
                    p.context[j] = spec->ops[i].ps.splice_spec.context[j];
                }
                p.num_outputs = spec->ops[fcOpIndex].ps.fc_spec.num_outputs;
                p.activation_type = ACTIVATION_NULL;

                spec->ops[fcOpIndex].ps.tdnn_spec = p;
                spec->ops[fcOpIndex].type = OT_Tdnn;
                setOperatorInvalid(spec, i, true);
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }
};
#endif
