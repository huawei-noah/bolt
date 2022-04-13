// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TRANSPOSEMULTOSCALEOPTIMIZER
#define _H_TRANSPOSEMULTOSCALEOPTIMIZER

#include "OPOptimizer.hpp"

class TransposeMulToScaleOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Eltwise && spec->ops[i].num_inputs == 2 &&
                spec->ops[i].ps.eltwise_spec.mode == ELTWISE_PROD) {
                int mulOpIndex = i;
                std::vector<std::pair<int, int>> prevOpIndexes = searchOperatorIndexByOutput(
                    spec, spec->ops[mulOpIndex].input_tensors_name[0], 0, mulOpIndex);
                if (prevOpIndexes.size() != 1 ||
                    (OT_Transpose != spec->ops[prevOpIndexes[0].first].type &&
                        OT_Reshape != spec->ops[prevOpIndexes[0].first].type)) {
                    continue;
                }
                int transposeOpIndex00 = prevOpIndexes[0].first;
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[transposeOpIndex00].output_tensors_name[0], transposeOpIndex00 + 1);
                if (nextOpIndexes.size() != 1) {
                    continue;
                }

                prevOpIndexes = searchOperatorIndexByOutput(
                    spec, spec->ops[mulOpIndex].input_tensors_name[1], 0, mulOpIndex);
                if (prevOpIndexes.size() != 1 ||
                    (OT_Transpose != spec->ops[prevOpIndexes[0].first].type &&
                        OT_Reshape != spec->ops[prevOpIndexes[0].first].type)) {
                    continue;
                }
                int transposeOpIndex01 = prevOpIndexes[0].first;
                nextOpIndexes = searchOperatorIndexByInput(spec,
                    spec->ops[transposeOpIndex01].output_tensors_name[0], transposeOpIndex01 + 1);
                if (nextOpIndexes.size() != 1) {
                    continue;
                }

                nextOpIndexes = searchOperatorIndexByInput(
                    spec, spec->ops[mulOpIndex].output_tensors_name[0], mulOpIndex + 1);
                if (nextOpIndexes.size() != 1 ||
                    (OT_Transpose != spec->ops[nextOpIndexes[0].first].type &&
                        OT_Reshape != spec->ops[nextOpIndexes[0].first].type)) {
                    continue;
                }
                int transposeOpIndex10 = nextOpIndexes[0].first;

                if (transposeOpIndex10 == mulOpIndex + 1 ||
                    (transposeOpIndex10 == mulOpIndex + 2 &&
                        spec->ops[mulOpIndex + 1].type == OT_Relu)) {
                    spec->ops[mulOpIndex].type = OT_Scale;
                    spec->ops[mulOpIndex].ps.scale_spec.axis = 1;
                    setOperatorInvalid(spec, transposeOpIndex00, true);
                    setOperatorInvalid(spec, transposeOpIndex01, true);
                    setOperatorInvalid(spec, transposeOpIndex10, true);
                    hasOptimized = true;
                    i = transposeOpIndex10;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
