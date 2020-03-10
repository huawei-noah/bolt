// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CLIPCLIPOPTIMIZER
#define _H_CLIPCLIPOPTIMIZER

#include <vector>
#include <string>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class ClipClipOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_Clip};
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Clip) {
                int opIndex0 = i;
                int opIndex1 = searchOperatorIndexForward(spec, opIndex0+1, queryOps, queryNum);
                if (opIndex1 == -1)
                    continue;

                str_copy(spec->ops[opIndex0].output_tensors_name[0], spec->ops[opIndex1].output_tensors_name[0], NAME_LEN);
                hasOptimized = true;
                spec->ops[opIndex0].ps.clip_spec.min = UNI_MAX(spec->ops[opIndex0].ps.clip_spec.min, 
                                                                  spec->ops[opIndex1].ps.clip_spec.min);
                spec->ops[opIndex0].ps.clip_spec.max = UNI_MIN(spec->ops[opIndex0].ps.clip_spec.max, 
                                                                  spec->ops[opIndex1].ps.clip_spec.max);
                setOperatorInvalid(spec, opIndex1);
                i = opIndex1;
            }
        }
        return hasOptimized;
    }
};
#endif
