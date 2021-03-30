// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_DEPTHWISEPOINTWISEOPTIMIZER
#define _H_DEPTHWISEPOINTWISEOPTIMIZER

#include "OPOptimizer.hpp"

class DepthwisePointwiseOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            // process depthwise convolution
            if (spec->ops[i].type == OT_Conv &&
                spec->ops[i].ps.conv_spec.convolution_type == Convolution_Depthwise) {
                int dwConvOpIndex = i;
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[dwConvOpIndex].output_tensors_name[0], dwConvOpIndex + 1,
                    spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 || OT_Conv != spec->ops[nextOpIndexes[0].first].type ||
                    spec->ops[nextOpIndexes[0].first].ps.conv_spec.convolution_type !=
                        Convolution_Pointwise ||
                    spec->ops[nextOpIndexes[0].first].ps.conv_spec.group != 1) {
                    UNI_WARNING_LOG("encounter unoptimize DepthwiseConv layer(no PointwiseConv): "
                                    "%s\n",
                        spec->ops[i].name);
                    continue;
                }
                int convOpIndex = nextOpIndexes[0].first;

                // reallocate weights and bias
                int dwConvWeightIndex = searchWeightIndex(spec, spec->ops[dwConvOpIndex].name);
                int convWeightIndex = searchWeightIndex(spec, spec->ops[convOpIndex].name);
                CHECK_REQUIREMENT(dwConvWeightIndex != -1);
                CHECK_REQUIREMENT(convWeightIndex != -1);
                CHECK_REQUIREMENT(spec->ws[dwConvWeightIndex].mdt == DT_F32);
                CHECK_REQUIREMENT(spec->ws[convWeightIndex].mdt == DT_F32);

                U32 weightSize = spec->ws[dwConvWeightIndex].bytes_of_weight +
                    spec->ws[convWeightIndex].bytes_of_weight;
                U8 *weight = (U8 *)mt_new_storage(weightSize);
                memcpy(weight, spec->ws[dwConvWeightIndex].weight,
                    spec->ws[dwConvWeightIndex].bytes_of_weight);
                memcpy(weight + spec->ws[dwConvWeightIndex].bytes_of_weight,
                    spec->ws[convWeightIndex].weight, spec->ws[convWeightIndex].bytes_of_weight);

                U32 vecSize = sizeof(F32) *
                    (spec->ops[dwConvOpIndex].ps.conv_spec.num_outputs +
                        spec->ops[convOpIndex].ps.conv_spec.num_outputs);
                U8 *vec = (U8 *)mt_new_storage(vecSize);
                U8 *ptr = vec;
                if (spec->ws[dwConvWeightIndex].bytes_of_vec == 0) {
                    memset(
                        ptr, 0, sizeof(F32) * (spec->ops[dwConvOpIndex].ps.conv_spec.num_outputs));
                } else {
                    CHECK_REQUIREMENT(
                        sizeof(F32) * (spec->ops[dwConvOpIndex].ps.conv_spec.num_outputs) ==
                        spec->ws[dwConvWeightIndex].bytes_of_vec);
                    memcpy(ptr, spec->ws[dwConvWeightIndex].vec,
                        spec->ws[dwConvWeightIndex].bytes_of_vec);
                }
                ptr = vec + sizeof(F32) * (spec->ops[dwConvOpIndex].ps.conv_spec.num_outputs);
                if (spec->ws[convWeightIndex].bytes_of_vec == 0) {
                    memset(ptr, 0, sizeof(F32) * (spec->ops[convOpIndex].ps.conv_spec.num_outputs));
                } else {
                    CHECK_REQUIREMENT(
                        sizeof(F32) * (spec->ops[convOpIndex].ps.conv_spec.num_outputs) ==
                        spec->ws[convWeightIndex].bytes_of_vec);
                    memcpy(
                        ptr, spec->ws[convWeightIndex].vec, spec->ws[convWeightIndex].bytes_of_vec);
                }

                // free and reallocate
                if (spec->ws[dwConvWeightIndex].weight != nullptr) {
                    spec->ws[dwConvWeightIndex].bytes_of_weight = 0;
                    if (outOfFileMapRange(spec->ws[dwConvWeightIndex].weight, spec->mfd)) {
                        delete spec->ws[dwConvWeightIndex].weight;
                    }
                    spec->ws[dwConvWeightIndex].weight = nullptr;
                }
                if (spec->ws[dwConvWeightIndex].vec != nullptr) {
                    spec->ws[dwConvWeightIndex].bytes_of_vec = 0;
                    if (outOfFileMapRange(spec->ws[dwConvWeightIndex].vec, spec->mfd)) {
                        delete spec->ws[dwConvWeightIndex].vec;
                    }
                    spec->ws[dwConvWeightIndex].vec = nullptr;
                }
                if (spec->ws[convWeightIndex].weight != nullptr) {
                    spec->ws[convWeightIndex].bytes_of_weight = 0;
                    if (outOfFileMapRange(spec->ws[convWeightIndex].weight, spec->mfd)) {
                        delete spec->ws[convWeightIndex].weight;
                    }
                    spec->ws[convWeightIndex].weight = nullptr;
                }
                if (spec->ws[convWeightIndex].vec != nullptr) {
                    spec->ws[convWeightIndex].bytes_of_vec = 0;
                    if (outOfFileMapRange(spec->ws[convWeightIndex].vec, spec->mfd)) {
                        delete spec->ws[convWeightIndex].vec;
                    }
                    spec->ws[convWeightIndex].vec = nullptr;
                }

                // retain depthwise convolution operator
                spec->ops[dwConvOpIndex].ps.conv_spec.num_outputs =
                    spec->ops[convOpIndex].ps.conv_spec.num_outputs;
                spec->ops[dwConvOpIndex].ps.conv_spec.convolution_type =
                    Convolution_Depthwise_Pointwise;
                spec->ops[dwConvOpIndex].ps.conv_spec.pw_activation_type =
                    spec->ops[convOpIndex].ps.conv_spec.pw_activation_type;
                spec->ws[dwConvWeightIndex].bytes_of_weight = weightSize;
                spec->ws[dwConvWeightIndex].weight = weight;
                spec->ws[dwConvWeightIndex].bytes_of_vec = vecSize;
                spec->ws[dwConvWeightIndex].vec = vec;
                setOperatorInvalid(spec, convOpIndex, true);
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }
};
#endif
