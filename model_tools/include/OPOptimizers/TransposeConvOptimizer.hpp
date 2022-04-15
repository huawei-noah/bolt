// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TRANSPOSE_CONV_OPTIMIZER
#define _H_TRANSPOSE_CONV_OPTIMIZER

#include "OPOptimizer.hpp"

class TransposeConvOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Transpose &&
                ((spec->ops[i].ps.transpose_spec.num_axes == 4 &&
                     spec->ops[i].ps.transpose_spec.axes[0] == 0 &&
                     spec->ops[i].ps.transpose_spec.axes[1] == 3 &&
                     spec->ops[i].ps.transpose_spec.axes[2] == 1 &&
                     spec->ops[i].ps.transpose_spec.axes[3] == 2) ||
                    (spec->ops[i].ps.transpose_spec.num_axes == 3 &&
                        spec->ops[i].ps.transpose_spec.axes[0] == 0 &&
                        spec->ops[i].ps.transpose_spec.axes[1] == 2 &&
                        spec->ops[i].ps.transpose_spec.axes[2] == 1))) {
                auto nextop = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (nextop.size() != 1) {
                    continue;
                }
                int next = nextop[0].first;
                if (spec->ops[next].type == OT_Where) {
                    nextop = searchOperatorIndexByInput(spec,
                        spec->ops[next].output_tensors_name[0], next + 1, spec->num_operator_specs);
                    if (nextop.size() != 1) {
                        continue;
                    }
                    next = nextop[0].first;
                }
                if (spec->ops[next].type == OT_Conv) {
                    spec->ops[i].ps.transpose_spec.df = DF_NCHWC8;
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
