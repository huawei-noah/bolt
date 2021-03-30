// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_PADOPTIMIZER
#define _H_PADOPTIMIZER

#include "OPOptimizer.hpp"

class PadOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Pad && spec->ops[i].ps.pad_spec.pad_mode == Pad_Constant &&
                spec->ops[i].ps.pad_spec.constant_value == 0) {
                int padOpIndex = i;
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[padOpIndex].output_tensors_name[0], padOpIndex + 1,
                    spec->num_operator_specs);
                if (!(nextOpIndexes.size() == 1 &&
                        (OT_Pooling == spec->ops[nextOpIndexes[0].first].type ||
                            OT_Conv == spec->ops[nextOpIndexes[0].first].type))) {
                    continue;
                }
                int nextOpIndex = nextOpIndexes[0].first;
                if (spec->ops[nextOpIndex].type == OT_Pooling) {
                    spec->ops[nextOpIndex].ps.pooling_spec.padding_before +=
                        spec->ops[padOpIndex].ps.pad_spec.before;
                    spec->ops[nextOpIndex].ps.pooling_spec.padding_after +=
                        spec->ops[padOpIndex].ps.pad_spec.after;
                    spec->ops[nextOpIndex].ps.pooling_spec.padding_top +=
                        spec->ops[padOpIndex].ps.pad_spec.top;
                    spec->ops[nextOpIndex].ps.pooling_spec.padding_bottom +=
                        spec->ops[padOpIndex].ps.pad_spec.bottom;
                    spec->ops[nextOpIndex].ps.pooling_spec.padding_left +=
                        spec->ops[padOpIndex].ps.pad_spec.left;
                    spec->ops[nextOpIndex].ps.pooling_spec.padding_right +=
                        spec->ops[padOpIndex].ps.pad_spec.right;
                }
                if (spec->ops[nextOpIndex].type == OT_Conv) {
                    spec->ops[nextOpIndex].ps.conv_spec.padding_before +=
                        spec->ops[padOpIndex].ps.pad_spec.before;
                    spec->ops[nextOpIndex].ps.conv_spec.padding_after +=
                        spec->ops[padOpIndex].ps.pad_spec.after;
                    spec->ops[nextOpIndex].ps.conv_spec.padding_top +=
                        spec->ops[padOpIndex].ps.pad_spec.top;
                    spec->ops[nextOpIndex].ps.conv_spec.padding_bottom +=
                        spec->ops[padOpIndex].ps.pad_spec.bottom;
                    spec->ops[nextOpIndex].ps.conv_spec.padding_left +=
                        spec->ops[padOpIndex].ps.pad_spec.left;
                    spec->ops[nextOpIndex].ps.conv_spec.padding_right +=
                        spec->ops[padOpIndex].ps.pad_spec.right;
                }
                setOperatorInvalid(spec, padOpIndex, true);
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }
};
#endif
