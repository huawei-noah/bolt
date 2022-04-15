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
        for (int i = 0; i < spec->num_operator_specs; i++) {
            bool mergeDilation = false;
            int spaceToBatchOpIndex = 0;
            int batchToSpaceOpIndex = 0;
            int convOpIndex = 0;
            if (spec->ops[i].type == OT_SpaceToBatchNd) {
                spaceToBatchOpIndex = i;
                std::string outName = spec->ops[i].output_tensors_name[0];
                auto indexPair =
                    searchOperatorIndexByInput(spec, outName, 0, spec->num_operator_specs);
                int index = indexPair[0].first;
                if (spec->ops[index].type == OT_Conv) {
                    convOpIndex = index;
                    outName = spec->ops[index].output_tensors_name[0];
                    indexPair =
                        searchOperatorIndexByInput(spec, outName, 0, spec->num_operator_specs);
                    index = indexPair[0].first;
                    if (spec->ops[index].type == OT_BatchToSpaceNd) {
                        batchToSpaceOpIndex = index;
                        mergeDilation = true;
                    }
                }
            }
            if (mergeDilation) {
                int spaceToBatchIndex = searchWeightIndex(spec, spec->ops[spaceToBatchOpIndex].name);
                int batchToSpaceIndex = searchWeightIndex(spec, spec->ops[batchToSpaceOpIndex].name);
                float *dilations = (float *)(spec->ws[spaceToBatchIndex].weight);
                float *pad1 = (float *)(spec->ws[spaceToBatchIndex].vec);
                float *pad2 = (float *)(spec->ws[batchToSpaceIndex].vec);
                int dim = spec->ws[spaceToBatchIndex].bytes_of_weight /
                    bytesOf(spec->ws[spaceToBatchIndex].mdt);

                spec->ops[convOpIndex].ps.conv_spec.dilatedRate_h = dilations[0];
                spec->ops[convOpIndex].ps.conv_spec.pad_top = pad1[0] - pad2[0];
                spec->ops[convOpIndex].ps.conv_spec.pad_bottom = pad1[1] - pad2[1];
                if (dim > 1) {
                    spec->ops[convOpIndex].ps.conv_spec.dilatedRate_w = dilations[1];
                    spec->ops[convOpIndex].ps.conv_spec.pad_left = pad1[2] - pad2[2];
                    spec->ops[convOpIndex].ps.conv_spec.pad_right = pad1[3] - pad2[3];
                }
                if (dim == 3) {
                    spec->ops[convOpIndex].ps.conv_spec.dilatedRate_t = dilations[0];
                    spec->ops[convOpIndex].ps.conv_spec.pad_before = pad1[0] - pad2[0];
                    spec->ops[convOpIndex].ps.conv_spec.pad_after = pad1[1] - pad2[1];
                    spec->ops[convOpIndex].ps.conv_spec.dilatedRate_h = dilations[1];
                    spec->ops[convOpIndex].ps.conv_spec.pad_top = pad1[2] - pad2[2];
                    spec->ops[convOpIndex].ps.conv_spec.pad_bottom = pad1[3] - pad2[3];
                    spec->ops[convOpIndex].ps.conv_spec.dilatedRate_w = dilations[2];
                    spec->ops[convOpIndex].ps.conv_spec.pad_left = pad1[4] - pad2[4];
                    spec->ops[convOpIndex].ps.conv_spec.pad_right = pad1[5] - pad2[5];
                }

                setOperatorInvalid(spec, spaceToBatchOpIndex, true);
                setOperatorInvalid(spec, batchToSpaceOpIndex, true);
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }
};
#endif
