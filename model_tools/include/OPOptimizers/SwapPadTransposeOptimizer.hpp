// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_SwapPadTransposeOPTIMIZER
#define _H_SwapPadTransposeOPTIMIZER

#include "SwapOPOptimizer.hpp"

class SwapPadTransposeOptimizer : public SwapOPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Pad && spec->ops[i].num_outputs == 1) {
                auto tmpVec = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (tmpVec.size() != 1) {
                    continue;
                }
                int next = tmpVec[0].first;
                if (spec->ops[next].type == OT_Transpose) {
                    auto padPs = spec->ops[i].ps.pad_spec;
                    auto transPs = spec->ops[next].ps.transpose_spec;
                    if (padPs.front != 0 || padPs.back != 0) {
                        if (transPs.num_axes == 4 && transPs.axes[0] == 0
                            && transPs.axes[1] == 3
                            && transPs.axes[2] == 1
                            && transPs.axes[3] == 2
                            ) {
                                spec->ops[i].ps.pad_spec.front = padPs.left;
                                spec->ops[i].ps.pad_spec.back = padPs.right;
                                spec->ops[i].ps.pad_spec.top = padPs.front;
                                spec->ops[i].ps.pad_spec.bottom = padPs.back;
                                spec->ops[i].ps.pad_spec.left = padPs.top;
                                spec->ops[i].ps.pad_spec.right = padPs.bottom;

                                shift_left(spec, {i, next});
                                hasOptimized = true;
                                i++;
                        }
                    }
                }
            }
        }
        return hasOptimized;
    }
};
#endif
