// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_SwapTransposeEltOPTIMIZER
#define _H_SwapTransposeEltOPTIMIZER

#include "OPOptimizer.hpp"

class SwapTransposeEltOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Transpose && spec->ops[i].num_outputs == 1) {
                auto tmpVec = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (tmpVec.size() != 1) {
                    continue;
                }
                int next = tmpVec[0].first;
                if (spec->ops[next].type == OT_Power || spec->ops[next].type == OT_Relu ||
                    spec->ops[next].type == OT_HSwish || spec->ops[next].type == OT_HSwishNoDiv) {
                    auto ps1 = spec->ops[i].ps;
                    auto ps2 = spec->ops[next].ps;

                    auto opType1 = spec->ops[i].type;
                    auto opType2 = spec->ops[next].type;

                    std::string opName1 = spec->ops[i].name;
                    std::string opName2 = spec->ops[next].name;
                    str_copy(spec->ops[i].name, opName2.c_str(), NAME_LEN);
                    str_copy(spec->ops[next].name, opName1.c_str(), NAME_LEN);
                    spec->ops[i].type = opType2;
                    spec->ops[next].type = opType1;
                    spec->ops[i].ps = ps2;
                    spec->ops[next].ps = ps1;
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
