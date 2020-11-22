// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_SQUEEZERESHAPEOPTIMIZER
#define _H_SQUEEZERESHAPEOPTIMIZER

#include "model_tools.h"
#include "OPOptimizer.hpp"

class SqueezeReshapeOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_Reshape};
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Squeeze) {
                int squeezeIndex = i;
                int reshapeIndex =
                    searchOperatorIndexForward(spec, squeezeIndex + 1, queryOps, queryNum);
                if (reshapeIndex == -1) {
                    continue;
                }
                if (strncmp(spec->ops[squeezeIndex].output_tensors_name[0],
                        spec->ops[reshapeIndex].input_tensors_name[0], NAME_LEN)) {
                    continue;
                }

                str_copy(spec->ops[reshapeIndex].input_tensors_name[0],
                    spec->ops[squeezeIndex].input_tensors_name[0], NAME_LEN);
                hasOptimized = true;
                setOperatorInvalid(spec, squeezeIndex);
                i = reshapeIndex;
            }
        }
        return hasOptimized;
    }
};
#endif
