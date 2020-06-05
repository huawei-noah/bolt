// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_FCFCOPTIMIZER
#define _H_FCFCOPTIMIZER

#include <vector>
#include <string>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class FCFCOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override
    {
        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_FC};
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_FC) {
                int curOpIndex = i;
                int prevOpIndex = searchOperatorIndexBackward(spec, curOpIndex - 1, queryOps, queryNum);
                if (prevOpIndex == -1) {
                    continue;
                }
                if (strncmp(spec->ops[curOpIndex].input_tensors_name[0], spec->ops[prevOpIndex].input_tensors_name[0], NAME_LEN)) {
                    continue;
                }
 
                int prevWeightIndex = searchWeightIndex(spec, spec->ops[prevOpIndex].name);
                int curWeightIndex = searchWeightIndex(spec, spec->ops[curOpIndex].name);
                CHECK_REQUIREMENT(prevWeightIndex != -1);
                CHECK_REQUIREMENT(curWeightIndex != -1);
                CHECK_REQUIREMENT(spec->ws[prevWeightIndex].mdt == DT_F32);
                CHECK_REQUIREMENT(spec->ws[curWeightIndex].mdt == DT_F32);

                U32 weightSize = spec->ws[prevWeightIndex].bytes_of_weight + spec->ws[curWeightIndex].bytes_of_weight;
                U8* weight = (U8 *)mt_new_storage(weightSize);
                memcpy(weight, spec->ws[prevWeightIndex].weight, spec->ws[prevWeightIndex].bytes_of_weight);
                memcpy(weight + spec->ws[prevWeightIndex].bytes_of_weight,
                       spec->ws[curWeightIndex].weight,
                       spec->ws[curWeightIndex].bytes_of_weight);

                U32 vecSize = sizeof(F32) * (spec->ops[prevOpIndex].ps.fc_spec.num_outputs \
                                           + spec->ops[curOpIndex].ps.fc_spec.num_outputs);
                U8* vec = (U8 *)mt_new_storage(vecSize);
                U8* ptr = vec;
                if (spec->ws[prevWeightIndex].bytes_of_vec == 0) {
                    memset(ptr, 0, sizeof(F32)*(spec->ops[prevOpIndex].ps.fc_spec.num_outputs));
                } else {
                    CHECK_REQUIREMENT(sizeof(F32)*(spec->ops[prevOpIndex].ps.fc_spec.num_outputs) == spec->ws[prevWeightIndex].bytes_of_vec);
                    memcpy(ptr, spec->ws[prevWeightIndex].vec, spec->ws[prevWeightIndex].bytes_of_vec);
                }
                ptr = vec + sizeof(F32)*(spec->ops[prevOpIndex].ps.fc_spec.num_outputs);
                if (spec->ws[curWeightIndex].bytes_of_vec == 0) {
                    memset(ptr, 0, sizeof(F32)*(spec->ops[curOpIndex].ps.fc_spec.num_outputs));
                } else {
                    CHECK_REQUIREMENT(sizeof(F32)*(spec->ops[curOpIndex].ps.fc_spec.num_outputs) == spec->ws[curWeightIndex].bytes_of_vec);
                    memcpy(ptr, spec->ws[curWeightIndex].vec, spec->ws[curWeightIndex].bytes_of_vec);
                }

                if (spec->ws[prevWeightIndex].weight != nullptr) {
                    spec->ws[prevWeightIndex].bytes_of_weight = 0;
                    delete [] spec->ws[prevWeightIndex].weight;
                    spec->ws[prevWeightIndex].weight = nullptr;
                }
                if (spec->ws[prevWeightIndex].vec != nullptr) {
                    spec->ws[prevWeightIndex].bytes_of_vec = 0;
                    delete [] spec->ws[prevWeightIndex].vec;
                    spec->ws[prevWeightIndex].vec = nullptr;
                }
                if (spec->ws[curWeightIndex].weight != nullptr) {
                    spec->ws[curWeightIndex].bytes_of_weight = 0;
                    delete [] spec->ws[curWeightIndex].weight;
                    spec->ws[curWeightIndex].weight = nullptr;
                }
                if (spec->ws[curWeightIndex].vec != nullptr) {
                    spec->ws[curWeightIndex].bytes_of_vec = 0;
                    delete [] spec->ws[curWeightIndex].vec;
                    spec->ws[curWeightIndex].vec = nullptr;
                }

                // FC params
                spec->ops[prevOpIndex].ps.fc_spec.num_slices++;
                U32 slices = spec->ops[prevOpIndex].ps.fc_spec.num_slices;
                CHECK_REQUIREMENT(slices <= sizeof(spec->ops[prevOpIndex].ps.fc_spec.slice_point) / sizeof(int));
                spec->ops[prevOpIndex].ps.fc_spec.slice_point[slices - 1] = spec->ops[curOpIndex].ps.fc_spec.num_outputs;
                spec->ops[prevOpIndex].ps.fc_spec.num_outputs += spec->ops[curOpIndex].ps.fc_spec.num_outputs;

                // operator spec
                spec->ops[prevOpIndex].num_outputs = slices;
                I8 **names = (I8**)mt_new_storage(slices * sizeof(I8 *));

                for (U32 j = 0; j < slices - 1; j++) {
                    names[j] = spec->ops[prevOpIndex].output_tensors_name[j];
                }
                names[slices - 1] = spec->ops[curOpIndex].output_tensors_name[0];
                delete [] spec->ops[prevOpIndex].output_tensors_name;
                delete [] spec->ops[curOpIndex].output_tensors_name;
                spec->ops[curOpIndex].output_tensors_name = nullptr;
                spec->ops[curOpIndex].num_outputs = 0;
                spec->ops[prevOpIndex].output_tensors_name = names;

                // weight spec
                spec->ws[prevWeightIndex].bytes_of_weight = weightSize;
                spec->ws[prevWeightIndex].weight = weight;
                spec->ws[prevWeightIndex].bytes_of_vec = vecSize;
                spec->ws[prevWeightIndex].vec = vec;
                hasOptimized = true;

                setOperatorInvalid(spec, curOpIndex);
                i = curOpIndex;
            }
        }
        return hasOptimized;
    }
};
#endif
