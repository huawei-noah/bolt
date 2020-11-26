// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ShGaUnCoReOPTIMIZER
#define _H_ShGaUnCoReOPTIMIZER

#include "OPOptimizer.hpp"

class ShGaUnCoReOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Shape) {
                int shapeOpIndex = i;
                if (spec->ops[i + 1].type == OT_Gather && spec->ops[i + 2].type == OT_Unsqueeze &&
                    spec->ops[i + 3].type == OT_Unsqueeze && spec->ops[i + 4].type == OT_Concat &&
                    spec->ops[i + 5].type == OT_Reshape) {
                    for (int k = 1; k < (int)(spec->ops[shapeOpIndex - 1].num_outputs); k++) {
                        delete spec->ops[shapeOpIndex - 1].output_tensors_name[k];
                        spec->ops[shapeOpIndex - 1].output_tensors_name[k] = nullptr;
                    }
                    spec->ops[shapeOpIndex - 1].num_outputs = 1;

                    for (int k = 1; k < (int)(spec->ops[i + 5].num_outputs); k++) {
                        delete spec->ops[i + 5].input_tensors_name[k];
                        spec->ops[i + 5].input_tensors_name[k] = nullptr;
                    }
                    spec->ops[i + 5].num_inputs = 1;

                    // make the reshape proper
                    spec->ops[i + 5].ps.reshape_spec.shape_dims[0] = 1;
                    spec->ops[i + 5].ps.reshape_spec.shape_dims[1] = -1;
                    spec->ops[i + 5].ps.reshape_spec.shape_size = 2;

                    // drop the redundant op
                    setOperatorInvalid(spec, i);
                    setOperatorInvalid(spec, i + 1);
                    setOperatorInvalid(spec, i + 2);
                    setOperatorInvalid(spec, i + 3);
                    setOperatorInvalid(spec, i + 4);
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
