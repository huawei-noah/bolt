// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_SCALEWEIGHTOPTIMIZER
#define _H_SCALEWEIGHTOPTIMIZER

#include "OPOptimizer.hpp"

// scale(ax + b) + conv --> scale(x + b / a) + conv
class ScaleWeightOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (OT_Scale == spec->ops[i].type && spec->ops[i].num_inputs == 1) {
                int scaleOpIndex = i;
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[scaleOpIndex].output_tensors_name[0], scaleOpIndex + 1,
                    spec->num_operator_specs);
                if (nextOpIndexes.size() != 1) {
                    continue;
                }
                int convOpIndex = nextOpIndexes[0].first;
                if (!(OT_Conv == spec->ops[convOpIndex].type &&
                        (spec->ops[convOpIndex].ps.conv_spec.convolution_type ==
                                CONVOLUTION_POINTWISE ||
                            spec->ops[convOpIndex].ps.conv_spec.convolution_type ==
                                CONVOLUTION_DILATION))) {
                    continue;
                }

                // scale
                int scaleWeightIndex = searchWeightIndex(spec, spec->ops[scaleOpIndex].name);
                CHECK_REQUIREMENT(scaleWeightIndex >= 0);
                CHECK_REQUIREMENT(spec->ws[scaleWeightIndex].mdt == DT_F32);
                U32 channelAlpha = spec->ws[scaleWeightIndex].bytes_of_weight /
                    bytesOf(spec->ws[scaleWeightIndex].mdt);
                U32 channelBeta = spec->ws[scaleWeightIndex].bytes_of_vec /
                    bytesOf(spec->ws[scaleWeightIndex].mdt);
                U32 channelCur = UNI_MAX(channelAlpha, channelBeta);
                F32 *alphaPtr = (F32 *)spec->ws[scaleWeightIndex].weight;
                if (alphaPtr == nullptr) {
                    continue;
                }
                F32 *betaPtr = (F32 *)spec->ws[scaleWeightIndex].vec;

                int convWeightIndex = searchWeightIndex(spec, spec->ops[convOpIndex].name);
                CHECK_REQUIREMENT(convWeightIndex >= 0);
                if (spec->ws[convWeightIndex].mdt == DT_BIN01 ||
                    spec->ws[convWeightIndex].mdt == DT_BIN11) {
                    continue;
                }
                CHECK_REQUIREMENT(spec->ws[convWeightIndex].mdt == DT_F32);

                if (betaPtr == nullptr) {
                    setOperatorInvalid(spec, scaleOpIndex, true);
                } else {
                    F32 *vecTemp = (F32 *)mt_malloc(spec->ws[scaleWeightIndex].bytes_of_vec);
                    for (U32 m = 0; m < channelCur; m++) {
                        vecTemp[m] = betaPtr[m] / alphaPtr[m];
                    }
                    mt_free(spec->ws[scaleWeightIndex].vec, spec);
                    spec->ws[scaleWeightIndex].vec = (U8 *)vecTemp;
                }
                F32 *oldWeight = (F32 *)spec->ws[convWeightIndex].weight;
                F32 *weightTemp = (F32 *)mt_malloc(spec->ws[convWeightIndex].bytes_of_weight);
                int weightPerChannel = spec->ops[convOpIndex].ps.conv_spec.kernel_t *
                    spec->ops[convOpIndex].ps.conv_spec.kernel_h *
                    spec->ops[convOpIndex].ps.conv_spec.kernel_w;
                for (U32 n = 0, id = 0; n < spec->ops[convOpIndex].ps.conv_spec.num_outputs; n++) {
                    for (U32 m = 0; m < channelCur; m++) {
                        for (int j = 0; j < weightPerChannel; j++, id++) {
                            weightTemp[id] = oldWeight[id] * alphaPtr[m];
                        }
                    }
                }
                mt_free(spec->ws[convWeightIndex].weight, spec);
                spec->ws[convWeightIndex].weight = (U8 *)weightTemp;

                spec->ws[scaleWeightIndex].bytes_of_weight = 0;
                mt_free(spec->ws[scaleWeightIndex].weight, spec);
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }
};
#endif
