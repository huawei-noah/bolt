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
#include "shape_infer.h"

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
                spec->ops[i].ps.reshape_spec.num_axes == -1 && spec->ops[i].num_inputs == 1) {
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (nextOpIndexes.size() != 1) {
                    continue;
                }
                int nextOpIndex = nextOpIndexes[0].first;
                if (spec->ops[nextOpIndex].type == OT_Squeeze) {
                    int index = 0;
                    for (int j = 0; j < spec->ops[i].ps.reshape_spec.num_shape; j++) {
                        bool flag = false;
                        for (int k = 0; k < spec->ops[nextOpIndex].ps.squeeze_spec.num_axes; k++) {
                            if (j == spec->ops[nextOpIndex].ps.squeeze_spec.axes[k]) {
                                flag = true;
                                break;
                            }
                        }
                        if (UNI_ABS(spec->ops[i].ps.reshape_spec.shape[j]) > 1 || !flag) {
                            if (flag) {
                                UNI_WARNING_LOG("try to squeeze an non-1 axis.\n");
                            }
                            spec->ops[i].ps.reshape_spec.shape[index] =
                                spec->ops[i].ps.reshape_spec.shape[j];
                            index++;
                        }
                    }
                    spec->ops[i].ps.reshape_spec.num_shape = index;
                    setOperatorInvalid(spec, nextOpIndex, true);
                    hasOptimized = true;
                }
                if (spec->ops[nextOpIndex].type == OT_Unsqueeze ||
                    spec->ops[nextOpIndex].type == OT_Squeeze) {
                    TensorDesc desc0, desc1;
                    desc0.nDims = spec->ops[i].ps.reshape_spec.num_shape;
                    for (U32 n = 0; n < desc0.nDims; n++) {
                        desc0.dims[desc0.nDims - 1 - n] = spec->ops[i].ps.reshape_spec.shape[n];
                    }
                    if (spec->ops[nextOpIndex].type == OT_Unsqueeze) {
                        CHECK_STATUS(unsqueeze_infer_output_size_cpu(
                            desc0, spec->ops[nextOpIndex].ps.unsqueeze_spec, &desc1));
                    } else {
                        CHECK_STATUS(squeeze_infer_output_size_cpu(
                            desc0, spec->ops[nextOpIndex].ps.squeeze_spec, &desc1));
                    }
                    spec->ops[i].ps.reshape_spec.num_shape = desc1.nDims;
                    for (U32 n = 0; n < desc1.nDims; n++) {
                        spec->ops[i].ps.reshape_spec.shape[n] = desc1.dims[desc1.nDims - 1 - n];
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
