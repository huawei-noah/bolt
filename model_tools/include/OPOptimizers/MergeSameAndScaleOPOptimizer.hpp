// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MERGESAMEANDSCALEOPOPTIMIZER
#define _H_MERGESAMEANDSCALEOPOPTIMIZER

#include <unordered_map>
#include "OPOptimizer.hpp"

/* merge same type op
            op                                      op
          /    \                                    |
        relu  relu             --->               relu
        /        \                                /  \
      op1        op2                            op1  op2
*/

/*  merge scale + power
              |                                    |
             scale                                 |
             |             --->     scale(alpha*p.scale, beta+p.shift)
       power(p.power=1)                            |
            |                                      |
*/

class MergeSameAndScaleOPOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        const int queryNum = 1;
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (!isValidOperator(spec, i)) {
                continue;
            }

            // merge same type op -> only relu now
            std::string curOut = spec->ops[i].output_tensors_name[0];
            auto nextOpIndexes =
                searchOperatorIndexByInput(spec, curOut, i + 1, spec->num_operator_specs, false);
            if (nextOpIndexes.size() > 1) {
                int nodeNum = nextOpIndexes.size();
                std::unordered_map<int, int> m;
                for (int j = 0; j < nodeNum; ++j) {
                    int idx = nextOpIndexes[j].first;
                    if (m.count((int)spec->ops[idx].type) && spec->ops[idx].type == OT_Relu) {
                        int destIdx = m[(int)spec->ops[idx].type];
                        std::string curIdxOut = spec->ops[idx].output_tensors_name[0];
                        auto nextIdxOpIndexes = searchOperatorIndexByInput(
                            spec, curIdxOut, idx + 1, spec->num_operator_specs, false);
                        for (auto nextIdxOpIndex : nextIdxOpIndexes) {
                            str_copy(spec->ops[nextIdxOpIndex.first]
                                         .input_tensors_name[nextIdxOpIndex.second],
                                spec->ops[destIdx].output_tensors_name[0],
                                strlen(spec->ops[destIdx].output_tensors_name[0]));
                        }
                        setOperatorInvalid(spec, idx);
                        hasOptimized = true;
                    } else {
                        m[(int)spec->ops[idx].type] = idx;
                    }
                }
            }

            // merge scale + power
            if ((spec->ops[i].type == OT_Scale) && (nextOpIndexes.size() <= 2) &&
                (nextOpIndexes.size() > 0) && (spec->ops[nextOpIndexes[0].first].type == OT_Power) &&
                (spec->ops[nextOpIndexes[0].first].ps.power_spec.power == 1)) {
                int curIdx = nextOpIndexes[0].first;
                if (nextOpIndexes.size() == 2 &&
                    strcmp(spec->ops[curIdx].input_tensors_name[0],
                        spec->ops[curIdx].output_tensors_name[0])) {
                    continue;
                }

                F32 powerAlpha = spec->ops[nextOpIndexes[0].first].ps.power_spec.scale;
                F32 powerBeta = spec->ops[nextOpIndexes[0].first].ps.power_spec.shift;

                int scaleWeightIndex = searchWeightIndex(spec, spec->ops[i].name);
                CHECK_REQUIREMENT(scaleWeightIndex >= 0);
                CHECK_REQUIREMENT(spec->ws[scaleWeightIndex].mdt == DT_F32);
                U32 channelAlpha = spec->ws[scaleWeightIndex].bytes_of_weight /
                    bytesOf(spec->ws[scaleWeightIndex].mdt);

                if (spec->ws[scaleWeightIndex].vec == nullptr) {
                    spec->ws[scaleWeightIndex].bytes_of_vec = channelAlpha * sizeof(F32);
                    spec->ws[scaleWeightIndex].vec =
                        (U8 *)mt_new_storage(spec->ws[scaleWeightIndex].bytes_of_vec);
                    memset(
                        spec->ws[scaleWeightIndex].vec, 0, spec->ws[scaleWeightIndex].bytes_of_vec);
                }

                F32 *scaleAlphaPtr = (F32 *)spec->ws[scaleWeightIndex].weight;
                F32 *scaleBetaPtr = (F32 *)spec->ws[scaleWeightIndex].vec;
                for (U32 m = 0; m < channelAlpha; ++m) {
                    scaleAlphaPtr[m] *= powerAlpha;
                    scaleBetaPtr[m] += powerBeta;
                }

                setOperatorInvalid(spec, curIdx);
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }
};
#endif
