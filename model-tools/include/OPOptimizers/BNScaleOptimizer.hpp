// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_BNSCALEOPTIMIZER
#define _H_BNSCALEOPTIMIZER

#include <vector>
#include <string>
#include <math.h>
#include "model_tools.h"
#include "op_type.h"
#include "OPOptimizer.hpp"

// Fold remaining BN into the following scale
class BNScaleOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_Scale};

        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_BatchNorm) {
                int bnOpIndex = i;
                int scaleOpIndex = searchOperatorIndexForward(spec, i+1, queryOps, queryNum);
                if (scaleOpIndex == -1) {
                    std::cout << "[WARNING] encounter unoptimized BatchNorm layer (no Scale following): " << spec->ops[i].name << std::endl;
                    continue;
                }

                // tensor relationship rewrite
                str_copy(spec->ops[scaleOpIndex].input_tensors_name[0], spec->ops[bnOpIndex].input_tensors_name[0], NAME_LEN);
                hasOptimized = true;

                // bn
                int bnWeightIndex = searchWeightIndex(spec, spec->ops[bnOpIndex].name);
                CHECK_REQUIREMENT(bnWeightIndex >= 0);
                CHECK_REQUIREMENT(spec->ws[bnWeightIndex].mdt == DT_F32);
                F32 epsCur = spec->ops[bnOpIndex].ps.bn_spec.eps;
                U32 channelCur = spec->ws[bnWeightIndex].bytes_of_weight / bytesOf(spec->ws[bnWeightIndex].mdt);
                F32* meanPtr = (F32 *)spec->ws[bnWeightIndex].weight;
                F32* varPtr = (F32 *)spec->ws[bnWeightIndex].vec;

                std::vector<float> stdValue(channelCur);
                for (U32 j=0; j < channelCur; j++) {
                    stdValue[j] = sqrt(varPtr[j] + epsCur);
                }

                // scale
                int scaleWeightIndex = searchWeightIndex(spec, spec->ops[scaleOpIndex].name);
                CHECK_REQUIREMENT(scaleWeightIndex >= 0);
                CHECK_REQUIREMENT(spec->ws[scaleWeightIndex].mdt == DT_F32);
                U32 channelAlpha = spec->ws[scaleWeightIndex].bytes_of_weight / bytesOf(spec->ws[scaleWeightIndex].mdt);
                CHECK_REQUIREMENT(channelAlpha == channelCur);

                if (spec->ws[scaleWeightIndex].vec == nullptr) {
                    spec->ws[scaleWeightIndex].bytes_of_vec = channelCur * sizeof(F32);
                    spec->ws[scaleWeightIndex].vec = (U8 *)mt_new_storage(spec->ws[scaleWeightIndex].bytes_of_vec);
                    memset(spec->ws[scaleWeightIndex].vec, 0, spec->ws[scaleWeightIndex].bytes_of_vec);
                }

                F32* alphaPtr = (F32*)spec->ws[scaleWeightIndex].weight;
                F32* betaPtr = (F32*)spec->ws[scaleWeightIndex].vec;

                for (U32 m = 0; m < channelCur; m++) {
                    alphaPtr[m] /= stdValue[m];
                    betaPtr[m] = betaPtr[m] - alphaPtr[m] * meanPtr[m] / stdValue[m];
                }
                // free BN memory
                if(spec->ws[bnWeightIndex].weight != nullptr) {
                    spec->ws[bnWeightIndex].bytes_of_weight = 0;
                    delete [] spec->ws[bnWeightIndex].weight;
                    spec->ws[bnWeightIndex].weight = nullptr;
                }
                if(spec->ws[bnWeightIndex].vec != nullptr) {
                    spec->ws[bnWeightIndex].bytes_of_vec = 0;
                    delete [] spec->ws[bnWeightIndex].vec;
                    spec->ws[bnWeightIndex].vec = nullptr;
                }
                setOperatorInvalid(spec, bnOpIndex);

                // If the previous OP is Concat, we need to take care of the possible padded channels before Concat.
                const int queryNum = 1;
                OperatorType queryOps[queryNum] = {OT_Concat};

                int concatOpIndex = searchOperatorIndexBackward(spec, i-1, queryOps, queryNum);
                if (concatOpIndex != -1) {
                    spec->ops[scaleOpIndex].ps.scale_spec.num_concat = spec->ops[concatOpIndex].num_inputs;
                    // Rename concat output and scale input to avoid desc differences for inplace tensor
                    std::string oldName = spec->ops[concatOpIndex].output_tensors_name[0];
                    std::string breakName = "break_" + oldName;
                    str_copy(spec->ops[concatOpIndex].output_tensors_name[0], breakName.c_str(), NAME_LEN);
                    str_copy(spec->ops[scaleOpIndex].input_tensors_name[0], breakName.c_str(), NAME_LEN);
                }
            }
        }
        return hasOptimized;
    }
};
#endif
