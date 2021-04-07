// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_SwapPadTransposeOPTIMIZER
#define _H_SwapPadTransposeOPTIMIZER

#include "OPOptimizer.hpp"

class SwapPadTransposeOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Pad && spec->ops[i].num_outputs == 1) {
                auto tmpVec = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (tmpVec.size() != 1) {
                    continue;
                }
                int next = tmpVec[0].first;
                if (spec->ops[next].type == OT_Transpose) {
                    auto padPs = spec->ops[i].ps.pad_spec;
                    auto transPs = spec->ops[next].ps.transpose_spec;
                    if (padPs.front != 0 || padPs.back != 0) {
                        if (transPs.trans_size == 4) {
                            if (transPs.trans_dims[0] == 0) {
                                int oriC1 = padPs.front;
                                int oriC2 = padPs.back;
                                int oriH1 = padPs.top;
                                int oriH2 = padPs.bottom;
                                int oriW1 = padPs.left;
                                int oriW2 = padPs.right;

                                padPs.front = oriW1;
                                padPs.back = oriW2;
                                padPs.top = oriC1;
                                padPs.bottom = oriC2;
                                padPs.left = oriH1;
                                padPs.right = oriH2;

                                std::string padName = spec->ops[i].name;
                                std::string transName = spec->ops[next].name;
                                str_copy(spec->ops[i].name, transName.c_str(), NAME_LEN);
                                str_copy(spec->ops[next].name, padName.c_str(), NAME_LEN);
                                spec->ops[i].type = OT_Transpose;
                                spec->ops[next].type = OT_Pad;
                                spec->ops[i].ps.transpose_spec = transPs;
                                spec->ops[next].ps.pad_spec = padPs;
                                hasOptimized = true;
                                i++;
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
