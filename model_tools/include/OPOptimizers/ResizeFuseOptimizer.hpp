// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ResizeFuseOPTIMIZER
#define _H_ResizeFuseOPTIMIZER

#include "OPOptimizer.hpp"

class ResizeFuseOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 5; i++) {
            if (spec->ops[i + 1].type == OT_Shape) {
                if (spec->ops[i + 2].type == OT_TfSlice) {
                    if (spec->ops[i + 3].type == OT_Concat) {
                        std::vector<std::pair<int, int>> prevOpIndexes = searchOperatorIndexByOutput(
                            spec, spec->ops[i + 3].input_tensors_name[1], 0, i);
                        CHECK_REQUIREMENT(prevOpIndexes.size() == 1);
                        int weightIndex =
                            searchWeightIndex(spec, spec->ops[prevOpIndexes[0].first].name);
                        if (weightIndex == -1) {
                            continue;
                        }
                        if (spec->ops[i + 4].type == OT_Resize) {
                            if (spec->ws[weightIndex].mdt != DT_I32 &&
                                spec->ws[weightIndex].mdt != DT_U32) {
                                continue;
                            }
                            if (spec->ws[weightIndex].bytes_of_weight != bytesOf(DT_I32) * 2) {
                                continue;
                            }
                            int *ptr = (int *)(spec->ws[weightIndex].weight);
                            spec->ops[i + 4].ps.resize_spec.sizes[0] = ptr[0];
                            spec->ops[i + 4].ps.resize_spec.sizes[1] = ptr[1];
                            spec->ops[i + 4].ps.resize_spec.num_sizes = 2;
                            spec->ops[i + 4].ps.resize_spec.num_scales = 0;
                            setOperatorInvalid(spec, i + 1, true);
                            setOperatorInvalid(spec, i + 2, true);
                            setOperatorInvalid(spec, i + 3, true);
                            hasOptimized = true;
                        }
                    }
                }
            }
        }
        return hasOptimized;
    }
};
#endif
