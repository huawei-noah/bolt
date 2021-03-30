// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_DilationConvolutionOPTIMIZER
#define _H_DilationConvolutionOPTIMIZER

#include "OPOptimizer.hpp"

class DilationConvolutionOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 2; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Conv && spec->ops[i - 1].type == OT_SpaceToBatchNd &&
                spec->ops[i + 1].type == OT_BatchToSpaceNd) {
                int spaceToBatchIndex = searchWeightIndex(spec, spec->ops[i - 1].name);
                int batchToSpaceIndex = searchWeightIndex(spec, spec->ops[i + 1].name);
                float *vecPtr = (float *)(spec->ws[spaceToBatchIndex].vec);
                float *vecPtr2 = (float *)(spec->ws[batchToSpaceIndex].vec);
                if (spec->ws[spaceToBatchIndex].bytes_of_vec != 16 &&
                    spec->ws[batchToSpaceIndex].bytes_of_vec != 16) {
                    UNI_ERROR_LOG("not support");
                }

                spec->ops[i].ps.conv_spec.convolution_type = Convolution_Dilation;
                spec->ops[i].ps.conv_spec.padding_top = vecPtr[0] - vecPtr2[0];
                spec->ops[i].ps.conv_spec.padding_bottom = vecPtr[1] - vecPtr2[1];
                spec->ops[i].ps.conv_spec.padding_left = vecPtr[2] - vecPtr2[2];
                spec->ops[i].ps.conv_spec.padding_right = vecPtr[3] - vecPtr2[3];
                spec->ops[i].ps.conv_spec.dilatedRate_h = 2;
                spec->ops[i].ps.conv_spec.dilatedRate_w = 2;

                setOperatorInvalid(spec, i - 1, true);
                setOperatorInvalid(spec, i + 1, true);

                if (spec->ws[spaceToBatchIndex].weight != nullptr) {
                    spec->ws[spaceToBatchIndex].bytes_of_weight = 0;
                    if (outOfFileMapRange(spec->ws[spaceToBatchIndex].weight, spec->mfd)) {
                        delete spec->ws[spaceToBatchIndex].weight;
                    }
                    spec->ws[spaceToBatchIndex].weight = nullptr;
                }
                if (spec->ws[spaceToBatchIndex].vec != nullptr) {
                    spec->ws[spaceToBatchIndex].bytes_of_vec = 0;
                    if (outOfFileMapRange(spec->ws[spaceToBatchIndex].vec, spec->mfd)) {
                        delete spec->ws[spaceToBatchIndex].vec;
                    }
                    spec->ws[spaceToBatchIndex].vec = nullptr;
                }
                if (spec->ws[batchToSpaceIndex].weight != nullptr) {
                    spec->ws[batchToSpaceIndex].bytes_of_weight = 0;
                    if (outOfFileMapRange(spec->ws[batchToSpaceIndex].weight, spec->mfd)) {
                        delete spec->ws[batchToSpaceIndex].weight;
                    }
                    spec->ws[batchToSpaceIndex].weight = nullptr;
                }
                if (spec->ws[batchToSpaceIndex].vec != nullptr) {
                    spec->ws[batchToSpaceIndex].bytes_of_vec = 0;
                    if (outOfFileMapRange(spec->ws[batchToSpaceIndex].vec, spec->mfd)) {
                        delete spec->ws[batchToSpaceIndex].vec;
                    }
                    spec->ws[batchToSpaceIndex].vec = nullptr;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
