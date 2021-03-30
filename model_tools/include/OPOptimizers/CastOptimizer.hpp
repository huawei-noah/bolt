// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CastOPTIMIZER
#define _H_CastOPTIMIZER

#include "OPOptimizer.hpp"

class CastOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Cast) {
                if (spec->ops[i].ps.cast_spec.targetDt == DT_I32) {
                    continue;
                }
                std::string castInput = std::string(spec->ops[i].input_tensors_name[0]);
                std::string castOutput = std::string(spec->ops[i].output_tensors_name[0]);

                for (int k = 0; k < (int)(spec->ops[i + 1].num_inputs); k++) {
                    if (spec->ops[i + 1].input_tensors_name[k] == castOutput) {
                        str_copy(
                            spec->ops[i + 1].input_tensors_name[k], castInput.c_str(), NAME_LEN);
                        break;
                    }
                }

                hasOptimized = true;
                // cut off the op[i] input and output information
                for (U32 j = 0; j < spec->ops[i].num_inputs; j++) {
                    free(spec->ops[i].input_tensors_name[j]);
                }
                spec->ops[i].num_inputs = 0;
                for (U32 j = 0; j < spec->ops[i].num_outputs; j++) {
                    free(spec->ops[i].output_tensors_name[j]);
                }
                spec->ops[i].num_outputs = 0;

                setOperatorInvalid(spec, i, true);
            }
        }
        return hasOptimized;
    }
};
#endif
