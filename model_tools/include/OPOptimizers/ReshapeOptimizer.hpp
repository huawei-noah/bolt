// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_RESHAPEOPTIMIZER
#define _H_RESHAPEOPTIMIZER

#include "OPOptimizer.hpp"

class ReshapeOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Squeeze || spec->ops[i].type == OT_Unsqueeze) {
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 ||
                    OT_Reshape != spec->ops[nextOpIndexes[0].first].type) {
                    continue;
                }
                setOperatorInvalid(spec, i, true);
                hasOptimized = true;
            }
            if (spec->ops[i].type == OT_Reshape && spec->ops[i].ps.reshape_spec.axis == 0 &&
                spec->ops[i].ps.reshape_spec.num_axes == -1) {
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (nextOpIndexes.size() != 1) {
                    continue;
                }
                int nextOpIndex = nextOpIndexes[0].first;
                if (spec->ops[nextOpIndex].type == OT_Squeeze) {
                    int index = 0;
                    for (int j = 0; j < spec->ops[i].ps.reshape_spec.shape_size; j++) {
                        bool flag = false;
                        for (int k = 0; k < spec->ops[nextOpIndex].ps.squeeze_spec.axes_num; k++) {
                            if (j == spec->ops[nextOpIndex].ps.squeeze_spec.axes[k]) {
                                flag = true;
                                break;
                            }
                        }
                        if (UNI_ABS(spec->ops[i].ps.reshape_spec.shape_dims[j]) > 1 || !flag) {
                            if (flag) {
                                UNI_WARNING_LOG("try to squeeze an non-1 axis.\n");
                            }
                            spec->ops[i].ps.reshape_spec.shape_dims[index] =
                                spec->ops[i].ps.reshape_spec.shape_dims[j];
                            index++;
                        }
                    }
                    spec->ops[i].ps.reshape_spec.shape_size = index;
                    setOperatorInvalid(spec, nextOpIndex, true);
                    hasOptimized = true;
                }
                if (spec->ops[nextOpIndex].type == OT_Unsqueeze) {
                    const int dim_max = 8;
                    int shapes[dim_max];
                    memset(shapes, 0, sizeof(int) * dim_max);
                    for (int k = 0; k < spec->ops[nextOpIndex].ps.unsqueeze_spec.axes_num; k++) {
                        shapes[spec->ops[nextOpIndex].ps.unsqueeze_spec.axes[k]] = 1;
                    }
                    int index = 0;
                    int j;
                    for (j = 0; j < dim_max; j++) {
                        if (shapes[j] == 0) {
                            if (index < spec->ops[i].ps.reshape_spec.shape_size) {
                                shapes[j] = spec->ops[i].ps.reshape_spec.shape_dims[index];
                                index++;
                            } else {
                                break;
                            }
                        }
                    }
                    spec->ops[i].ps.reshape_spec.shape_size = j;
                    for (int j = 0; j < spec->ops[i].ps.reshape_spec.shape_size; j++) {
                        spec->ops[i].ps.reshape_spec.shape_dims[j] = shapes[j];
                    }
                    setOperatorInvalid(spec, nextOpIndex, true);
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
