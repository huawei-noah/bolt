// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CONSTUPSAMPLEOPTIMIZER
#define _H_CONSTUPSAMPLEOPTIMIZER

#include <vector>
#include <string>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class ConstUpsampleOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_Constant};
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Upsample) {
                int upsampleOpIndex = i;
                int constOpIndex = searchOperatorIndexBackward(spec, i-1, queryOps, queryNum);
                if (constOpIndex == -1) {
                    std::cout << "[WARNING] encounter unoptimized Upsample (no Constant before): " << spec->ops[upsampleOpIndex].name << std::endl;
                    continue;
                }

                CHECK_REQUIREMENT(spec->ops[constOpIndex].ps.const_f32_spec.size == 4);
                memcpy(spec->ops[upsampleOpIndex].ps.upsample_spec.scale, spec->ops[constOpIndex].ps.const_f32_spec.values, 4*bytesOf(DT_F32));
                delete [] spec->ops[constOpIndex].ps.const_f32_spec.values;
                setOperatorInvalid(spec, constOpIndex);
            }
        }
        return hasOptimized;
    }
};
#endif
