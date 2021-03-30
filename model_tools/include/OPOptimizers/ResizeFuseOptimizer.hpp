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
            if (spec->ops[i].type == OT_Relu) {
                auto cur_relu_spec = spec->ops[i].ps.relu_spec;
                std::string tmp_str = spec->ops[i].name;
                if (cur_relu_spec.neg_slope == 0) {
                    continue;
                }
                if (spec->ops[i + 1].type == OT_Shape) {
                    if (spec->ops[i + 2].type == OT_TfSlice) {
                        if (spec->ops[i + 3].type == OT_Concat) {
                            int weightIndex =
                                searchWeightIndex(spec, spec->ops[i + 3].input_tensors_name[1]);
                            if (weightIndex == -1) {
                                continue;
                            }
                            if (spec->ops[i + 4].type == OT_Resize) {
                                float *weight_ptr = (float *)(spec->ws[weightIndex].weight);
                                spec->ops[i + 4].ps.resize_spec.sizes[0] = weight_ptr[0];
                                spec->ops[i + 4].ps.resize_spec.sizes[1] = weight_ptr[1];
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
        }
        return hasOptimized;
    }
};
#endif
