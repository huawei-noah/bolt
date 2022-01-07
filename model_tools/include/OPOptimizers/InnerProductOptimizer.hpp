// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_InnerProductOPTIMIZER
#define _H_InnerProductOPTIMIZER

#include "OPOptimizer.hpp"

class InnerProductOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_FC && spec->ops[i + 1].type == OT_Scale) {
                int firScaleIndex = i;
                int secScaleIndex = i + 1;
                std::string firScaleOutput = spec->ops[firScaleIndex].output_tensors_name[0];
                std::string secScaleInput = spec->ops[secScaleIndex].input_tensors_name[0];
                if (spec->ops[firScaleIndex].num_inputs == 1 &&
                    spec->ops[firScaleIndex].num_outputs == 1) {
                    if (spec->ops[secScaleIndex].num_inputs == 1 &&
                        spec->ops[secScaleIndex].num_outputs == 1) {
                        if (firScaleOutput == secScaleInput) {
                            int firScaleWeightIndex =
                                searchWeightIndex(spec, spec->ops[firScaleIndex].name);
                            int secScaleWeightIndex =
                                searchWeightIndex(spec, spec->ops[secScaleIndex].name);
                            if (spec->ws[firScaleWeightIndex].bytes_of_weight != 0 &&
                                spec->ws[firScaleWeightIndex].bytes_of_vec == 0 &&
                                spec->ws[secScaleWeightIndex].bytes_of_weight == 0 &&
                                spec->ws[secScaleWeightIndex].bytes_of_vec != 0) {
                                if (spec->ops[firScaleIndex].ps.fc_spec.num_outputs * sizeof(float) !=
                                    spec->ws[secScaleWeightIndex].bytes_of_vec) {
                                    continue;
                                }
                                spec->ws[firScaleWeightIndex].bytes_of_vec =
                                    spec->ws[secScaleWeightIndex].bytes_of_vec;
                                U8 *ln_vec = (U8 *)mt_new_storage(
                                    spec->ws[secScaleWeightIndex].bytes_of_vec);
                                memcpy(ln_vec, spec->ws[secScaleWeightIndex].vec,
                                    spec->ws[secScaleWeightIndex].bytes_of_vec);
                                spec->ws[firScaleWeightIndex].vec = ln_vec;

                                spec->ws[secScaleWeightIndex].bytes_of_vec = 0;
                                if (outOfFileMapRange(spec->ws[secScaleWeightIndex].vec, spec->mfd)) {
                                    delete spec->ws[secScaleWeightIndex].vec;
                                }
                                spec->ws[secScaleWeightIndex].vec = nullptr;
                                memcpy(spec->ops[firScaleIndex].output_tensors_name[0],
                                    spec->ops[secScaleIndex].output_tensors_name[0], NAME_LEN);
                                setOperatorInvalid(spec, secScaleIndex);
                                hasOptimized = true;
                            }
                        }
                    }
                }
            }
        }
        return hasOptimized;
    }
};
#endif
