// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ACTIVATIONOPTIMIZER
#define _H_ACTIVATIONOPTIMIZER

#include "OPOptimizer.hpp"

class ActivationOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (OT_Conv == spec->ops[i].type || OT_Eltwise == spec->ops[i].type ||
                OT_Tdnn == spec->ops[i].type) {
                int prevOpIndex = i;
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[prevOpIndex].output_tensors_name[0], prevOpIndex + 1,
                    spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 || OT_Relu != spec->ops[nextOpIndexes[0].first].type ||
                    spec->ops[nextOpIndexes[0].first].ps.relu_spec.neg_slope != 0) {
                    continue;
                }
                int atOpIndex = nextOpIndexes[0].first;

                // tensor relationship rewrite
                if (spec->ops[prevOpIndex].type == OT_Conv) {
                    switch (spec->ops[prevOpIndex].ps.conv_spec.convolution_type) {
                        case Convolution_Pointwise: {
                            spec->ops[prevOpIndex].ps.conv_spec.pw_activation_type = ACTIVATION_RELU;
                            break;
                        }
                        case Convolution_Deconvolution: {
                            spec->ops[prevOpIndex].ps.conv_spec.pw_activation_type = ACTIVATION_RELU;
                            break;
                        }
                        case Convolution_Depthwise: {
                            spec->ops[prevOpIndex].ps.conv_spec.dw_activation_type = ACTIVATION_RELU;
                            break;
                        }
                        case Convolution_Dilation: {
                            spec->ops[prevOpIndex].ps.conv_spec.pw_activation_type = ACTIVATION_RELU;
                            break;
                        }
                        default: {
                            CHECK_REQUIREMENT(0);
                            break;
                        }
                    }
                    spec->ops[prevOpIndex].ps.conv_spec.activation_spec.relu_spec =
                        spec->ops[atOpIndex].ps.relu_spec;
                }
                if (spec->ops[prevOpIndex].type == OT_Eltwise) {
                    spec->ops[prevOpIndex].ps.eltwise_spec.activation_type = ACTIVATION_RELU;
                    spec->ops[prevOpIndex].ps.eltwise_spec.activation_spec.relu_spec =
                        spec->ops[atOpIndex].ps.relu_spec;
                }
                if (spec->ops[prevOpIndex].type == OT_Tdnn) {
                    spec->ops[prevOpIndex].ps.tdnn_spec.activation_type = ACTIVATION_RELU;
                    spec->ops[prevOpIndex].ps.tdnn_spec.activation_spec.relu_spec =
                        spec->ops[atOpIndex].ps.relu_spec;
                }
                setOperatorInvalid(spec, atOpIndex, true);
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }
};
#endif
