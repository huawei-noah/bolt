// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TRANSPOSEOPTIMIZER
#define _H_TRANSPOSEOPTIMIZER

#include "OPOptimizer.hpp"

class TransposeOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Transpose && spec->ops[i].num_inputs == 1 &&
                spec->ops[i].num_outputs == 1) {
                auto tmpVec = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (tmpVec.size() != 1) {
                    continue;
                }
                int next = tmpVec[0].first;
                if (spec->ops[next].type == OT_Transpose && spec->ops[next].num_inputs == 1) {
                    auto ps1 = spec->ops[i].ps.transpose_spec;
                    auto ps2 = spec->ops[next].ps.transpose_spec;
                    if (ps1.trans_size != ps2.trans_size) {
                        UNI_ERROR_LOG("neighbor two transpose operators(%s, %s) dimensions not "
                                      "equal.\n",
                            spec->ops[i].name, spec->ops[next].name);
                        continue;
                    }
                    bool invalid = true;
                    for (U32 j = 0; j < ps2.trans_size; j++) {
                        ps2.trans_dims[j] = ps1.trans_dims[ps2.trans_dims[j]];
                        if (ps2.trans_dims[j] != j) {
                            invalid = false;
                        }
                    }
                    if (invalid) {
                        setOperatorInvalid(spec, next, true);
                    } else {
                        spec->ops[next].ps.transpose_spec = ps2;
                    }
                    setOperatorInvalid(spec, i, true);
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
