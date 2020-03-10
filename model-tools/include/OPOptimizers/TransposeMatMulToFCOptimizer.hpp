// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_TRANSPOSEMATMULTOFCOPTIMIZER
#define _H_TRANSPOSEMATMULTOFCOPTIMIZER

#include <vector>
#include <string>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class TransposeMatMulToFCOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_MatMul};
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Transpose) {
                int transposeOpIndex = i;
                int transposeWeightIndex = searchWeightIndex(spec, spec->ops[transposeOpIndex].name);
                if (transposeWeightIndex < 0) {
                    // This transpose does not have weight
                    continue;
                }

                int matmulOpIndex = searchOperatorIndexForward(spec, transposeOpIndex + 1, queryOps, queryNum);
                if (-1 == matmulOpIndex) {
                    std::cerr << "[ERROR] encountered transpose with weight but cannot be optimized\n";
                    CHECK_STATUS(NOT_SUPPORTED);
                }

                hasOptimized = true;

                // Update matmul to fc
                spec->ops[matmulOpIndex].type = OT_FC;
                spec->ops[matmulOpIndex].num_inputs = 1;
                free(spec->ops[matmulOpIndex].input_tensors_name[1]);
                spec->ops[matmulOpIndex].ps.fc_spec.num_outputs = spec->ws[transposeWeightIndex].bytes_of_vec;

                // Adjust the owner of the weight
                str_copy(spec->ws[transposeWeightIndex].op_name, spec->ops[matmulOpIndex].name, NAME_LEN);
                spec->ws[transposeWeightIndex].bytes_of_vec = 0;

                setOperatorInvalid(spec, transposeOpIndex);
                i = matmulOpIndex;
            }
        }
        return hasOptimized;
    }
};
#endif
