// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CONV_FC_OPTIMIZER
#define _H_CONV_FC_OPTIMIZER

#include "OPOptimizer.hpp"

class ConvFCOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < (int)spec->num_operator_specs - 3; i++) {
            if (spec->ops[i].type == OT_Conv) {
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                int transId = -1;
                for (U32 j = 0; j < nextOpIndexes.size(); j++) {
                    if (OT_Transpose == spec->ops[nextOpIndexes[j].first].type &&
                        spec->ops[nextOpIndexes[j].first].ps.transpose_spec.num_axes == 4 &&
                        spec->ops[nextOpIndexes[j].first].ps.transpose_spec.axes[0] == 0 &&
                        spec->ops[nextOpIndexes[j].first].ps.transpose_spec.axes[1] == 2 &&
                        spec->ops[nextOpIndexes[j].first].ps.transpose_spec.axes[2] == 1 &&
                        spec->ops[nextOpIndexes[j].first].ps.transpose_spec.axes[3] == 3) {
                        transId = nextOpIndexes[j].first;
                        break;
                    }
                }
                if (transId == -1) {
                    continue;
                }
                nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[transId].output_tensors_name[0], transId + 1,
                    spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 ||
                    OT_Reshape != spec->ops[nextOpIndexes[0].first].type ||
                    spec->ops[nextOpIndexes[0].first].ps.reshape_spec.num_shape != 3) {
                    continue;
                }
                int reshapeId = nextOpIndexes[0].first;

                nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[reshapeId].output_tensors_name[0], reshapeId + 1,
                    spec->num_operator_specs);
                if (nextOpIndexes.size() != 1 || OT_FC != spec->ops[nextOpIndexes[0].first].type) {
                    continue;
                }

                int fcId = nextOpIndexes[0].first;
                int wid = searchWeightIndex(spec, spec->ops[fcId].name);
                CHECK_REQUIREMENT(wid >= 0);
                CHECK_REQUIREMENT(spec->ws[wid].mdt == DT_F32);
                F32 *p0 = (F32 *)spec->ws[wid].weight;
                F32 *p1 = (F32 *)mt_malloc(spec->ws[wid].bytes_of_weight);
                U32 column = spec->ops[i].ps.conv_spec.num_outputs;
                U32 row = spec->ws[wid].bytes_of_weight / bytesOf(spec->ws[wid].mdt) /
                    spec->ops[fcId].ps.fc_spec.num_outputs / column;
                for (U32 a = 0, offset = 0; a < spec->ops[fcId].ps.fc_spec.num_outputs;
                     a++, offset += row * column) {
                    for (U32 r = 0; r < row; r++) {
                        for (U32 c = 0; c < column; c++) {
                            p1[offset + r * column + c] = p0[offset + c * row + r];
                        }
                    }
                }
                spec->ws[wid].weight = (U8 *)p1;
                mt_free(p0, spec);
                spec->ops[transId].ps.transpose_spec.axes[2] = 3;
                spec->ops[transId].ps.transpose_spec.axes[3] = 1;
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }
};
#endif
