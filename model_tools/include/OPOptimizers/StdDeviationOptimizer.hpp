// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_STDDEVIATIONOPTIMIZER
#define _H_STDDEVIATIONOPTIMIZER

#include "OPOptimizer.hpp"

class StdDeviationOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 2; i++) {
            if (OT_SqDiff == spec->ops[i].type) {
                if (OT_Reduction == spec->ops[i + 1].type && OT_Power == spec->ops[i + 2].type) {
                    CHECK_REQUIREMENT(
                        REDUCTION_MEAN == spec->ops[i + 1].ps.reduction_spec.reduction_mode);
                    spec->ops[i + 1].ps.reduction_spec.reduction_mode = REDUCTION_STD_DEVIATION;

                    str_copy(spec->ops[i + 1].input_tensors_name[0],
                        spec->ops[i].input_tensors_name[0], NAME_LEN);
                    str_copy(spec->ops[i + 1].output_tensors_name[0],
                        spec->ops[i + 2].output_tensors_name[0], NAME_LEN);
                    hasOptimized = true;
                    setOperatorInvalid(spec, i);
                    setOperatorInvalid(spec, i + 2);
                    i += 2;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
