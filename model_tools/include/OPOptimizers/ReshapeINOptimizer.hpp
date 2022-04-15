// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_RESHAPEINOPTIMIZER
#define _H_RESHAPEINOPTIMIZER

#include "OPOptimizer.hpp"

class ReshapeINOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_InstanceNorm) {
                int inOpIndex = i;

                std::vector<std::pair<int, int>> prevOpIndexes = searchOperatorIndexByOutput(
                    spec, spec->ops[i].input_tensors_name[0], 0, inOpIndex);
                if (prevOpIndexes.size() != 1 ||
                    OT_Reshape != spec->ops[prevOpIndexes[0].first].type) {
                    continue;
                }

                int axis = spec->ops[inOpIndex].ps.in_spec.axis;
                if (axis < 0) {
                    continue;
                }

                int reshapeIndex = prevOpIndexes[0].first;
                int *reshape0Shape = spec->ops[reshapeIndex].ps.reshape_spec.shape;

                bool thisOptimized = true;
                for (int j = 0; j < axis; ++j) {
                    if (reshape0Shape[j] != 0 && reshape0Shape[j] != 1) {
                        thisOptimized = false;
                    }
                }

                if (thisOptimized && (reshape0Shape[axis] > 0)) {
                    spec->ops[inOpIndex].ps.in_spec.axis_dim = reshape0Shape[axis];
                    setOperatorInvalid(spec, reshapeIndex, true);
                    hasOptimized = true;
                }
            }
        }

        return hasOptimized;
    }
};
#endif
