// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_InvariantSliceOPTIMIZER
#define _H_InvariantSliceOPTIMIZER

#include "OPOptimizer.hpp"

class InvariantSliceOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Slice) {
                int sliceOpIndex = i;
                int curSliceSize = spec->ops[sliceOpIndex].ps.slice_spec.num_slice;
                if (curSliceSize == 0) {
                    UNI_MEMCPY(spec->ops[sliceOpIndex - 1].output_tensors_name[0],
                        spec->ops[sliceOpIndex].output_tensors_name[0], NAME_LEN);
                    setOperatorInvalid(spec, sliceOpIndex);
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
