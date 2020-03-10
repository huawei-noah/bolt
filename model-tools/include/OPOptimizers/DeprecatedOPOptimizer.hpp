// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_DEPRECATEDOPOPTIMIZER
#define _H_DEPRECATEDOPOPTIMIZER

#include <vector>
#include <string>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class DeprecatedOPOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        bool hasOptimized = false;

        for (int i = 0; i< spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_None && i > 0) {
                str_copy(spec->ops[i - 1].output_tensors_name[0], spec->ops[i].output_tensors_name[0], NAME_LEN);
                hasOptimized = true;
                continue;
            }

            if (spec->ops[i].type == OT_Pad) {
                if(spec->ops[i].ps.pad_spec.top == 0 && spec->ops[i].ps.pad_spec.bottom == 0 && 
                    spec->ops[i].ps.pad_spec.left == 0 && spec->ops[i].ps.pad_spec.right == 0){
                    str_copy(spec->ops[i + 1].input_tensors_name[0], spec->ops[i].input_tensors_name[0], NAME_LEN);
                    hasOptimized = true;
                    spec->ops[i].type = OT_None;  // trick
                    continue;
                }
            }

            if(spec->ops[i].type == OT_Input) {
                spec->ops[i].type = OT_None;  // trick
                continue;
            }
        }
        return hasOptimized;
    }
    public:
        static bool isDeprecatedOp(OperatorType opType){
            if (opType == OT_None) {
                return true;
            } else {
                return false;
            }
        }
        static bool isDeprecatedOpWeight(const ModelSpec* spec, int index){
            if (index >= spec->num_weight_specs) {
                return true;
            } else {
                if (spec->ws[index].bytes_of_weight == 0 && spec->ws[index].bytes_of_vec == 0) {
                    return true;
                } else {
                    return false;
                }
            }
        }
};
#endif
