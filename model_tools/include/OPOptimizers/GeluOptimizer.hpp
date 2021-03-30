// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_GeluOPTIMIZER
#define _H_GeluOPTIMIZER

#include "OPOptimizer.hpp"

class GeluOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Erf) {
                int erfIndex = i;
                int firMulIndex = erfIndex - 2;
                int divIndex = erfIndex - 1;
                int AddIndex = erfIndex + 1;
                int secMulIndex = erfIndex + 2;

                if (spec->ops[firMulIndex].type == OT_Power) {
                    if (spec->ops[divIndex].type == OT_Power) {
                        if (spec->ops[AddIndex].type == OT_Power) {
                            if (spec->ops[secMulIndex].type == OT_Eltwise) {
                                spec->ops[secMulIndex].num_inputs = 1;
                                free(spec->ops[secMulIndex].input_tensors_name[1]);
                                memcpy(spec->ops[secMulIndex].input_tensors_name[0],
                                    spec->ops[divIndex].input_tensors_name[0], NAME_LEN);
                                spec->ops[secMulIndex].type = OT_Gelu;
                                setOperatorInvalid(spec, firMulIndex);
                                setOperatorInvalid(spec, divIndex);
                                setOperatorInvalid(spec, erfIndex);
                                setOperatorInvalid(spec, AddIndex);

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
