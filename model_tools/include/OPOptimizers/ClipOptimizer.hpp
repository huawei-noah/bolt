// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CLIPOPTIMIZER
#define _H_CLIPOPTIMIZER

#include "OPOptimizer.hpp"

class ClipOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        hasOptimized |= clipclip(spec);
        hasOptimized |= clip2relu6(spec);
        return hasOptimized;
    }

    bool clip2relu6(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Clip && spec->ops[i].ps.clip_spec.min == 0 &&
                spec->ops[i].ps.clip_spec.max == 6) {
                spec->ops[i].type = OT_Relu6;
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }

    bool clipclip(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Clip) {
                int opIndex0 = i;
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[opIndex0].output_tensors_name[0], opIndex0 + 1,
                    spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 || OT_Clip != spec->ops[nextOpIndexes[0].first].type) {
                    UNI_WARNING_LOG(
                        "encounter unoptimize Clip layer(no Clip): %s\n", spec->ops[i].name);
                    continue;
                }
                int opIndex1 = nextOpIndexes[0].first;

                spec->ops[opIndex0].ps.clip_spec.min = UNI_MAX(
                    spec->ops[opIndex0].ps.clip_spec.min, spec->ops[opIndex1].ps.clip_spec.min);
                spec->ops[opIndex0].ps.clip_spec.max = UNI_MIN(
                    spec->ops[opIndex0].ps.clip_spec.max, spec->ops[opIndex1].ps.clip_spec.max);
                setOperatorInvalid(spec, opIndex1, true);
                hasOptimized = true;
                i = opIndex1;
            }
        }
        return hasOptimized;
    }
};
#endif
