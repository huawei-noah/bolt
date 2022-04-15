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

#include "OPOptimizer.hpp"

class TransposeMatMulToFCOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_MatMul};
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Transpose) {
                int transposeOpIndex = i;
                int transposeWeightIndex = searchWeightIndex(spec, spec->ops[transposeOpIndex].name);
                if (transposeWeightIndex < 0) {
                    // This transpose does not have weight

                    // fuse Transpose and Matmul
                    std::string curOut = spec->ops[i].output_tensors_name[0];
                    auto nextOpIndexes = searchOperatorIndexByInput(
                        spec, curOut, i + 1, spec->num_operator_specs, false);
                    if (nextOpIndexes.size() != 1 ||
                        spec->ops[nextOpIndexes[0].first].type != OT_MatMul) {
                        continue;
                    }

                    int matmulIdx = nextOpIndexes[0].first;
                    int matmulSubIdx = nextOpIndexes[0].second;
                    U32 paramSize = spec->ops[i].ps.transpose_spec.trans_size;
                    U32 *transDims = spec->ops[i].ps.transpose_spec.trans_dims;
                    bool fuseTranspose = true;
                    if (paramSize < 2) {
                        continue;
                    }
                    for (U32 paramIdx = 0; paramIdx < paramSize - 2; ++paramIdx) {
                        if (paramIdx != transDims[paramIdx]) {
                            fuseTranspose = false;
                        }
                    }

                    if (fuseTranspose) {
                        // Transpose(0, 1, 3, 2) + Matmul ===> Matmul_Transposed

                        if (matmulSubIdx == 0) {
                            spec->ops[matmulIdx].ps.matmul_spec.transpose_a =
                                !spec->ops[matmulIdx].ps.matmul_spec.transpose_a;
                            str_copy(spec->ops[matmulIdx].input_tensors_name[0],
                                spec->ops[i].input_tensors_name[0],
                                strlen(spec->ops[i].input_tensors_name[0]));
                        } else if (matmulSubIdx == 1) {
                            spec->ops[matmulIdx].ps.matmul_spec.transpose_b =
                                !spec->ops[matmulIdx].ps.matmul_spec.transpose_b;
                            str_copy(spec->ops[matmulIdx].input_tensors_name[1],
                                spec->ops[i].input_tensors_name[0],
                                strlen(spec->ops[i].input_tensors_name[0]));
                        }
                        setOperatorInvalid(spec, i);
                    } else if (transDims[paramSize - 2] == (paramSize - 1)) {
                        // Transpose(x, x, 3, x) + Matmul ===> Transpose(x, x, x, 3) + Matmul_Transposed
                        transDims[paramSize - 2] = transDims[paramSize - 1];
                        transDims[paramSize - 1] = paramSize - 1;
                        if (matmulSubIdx == 0) {
                            spec->ops[matmulIdx].ps.matmul_spec.transpose_a =
                                !spec->ops[matmulIdx].ps.matmul_spec.transpose_a;
                        } else if (matmulSubIdx == 1) {
                            spec->ops[matmulIdx].ps.matmul_spec.transpose_b =
                                !spec->ops[matmulIdx].ps.matmul_spec.transpose_b;
                        }
                    }

                    continue;
                }

                int matmulOpIndex =
                    searchOperatorIndexForward(spec, transposeOpIndex + 1, queryOps, queryNum);
                if (-1 == matmulOpIndex) {
                    UNI_ERROR_LOG("encountered transpose with weight but cannot be optimized\n");
                    CHECK_STATUS(NOT_SUPPORTED);
                }

                hasOptimized = true;

                // Update matmul to fc
                spec->ops[matmulOpIndex].type = OT_FC;
                spec->ops[matmulOpIndex].num_inputs = 1;
                delete spec->ops[matmulOpIndex].input_tensors_name[1];
                spec->ops[matmulOpIndex].ps.fc_spec.num_outputs =
                    spec->ws[transposeWeightIndex].bytes_of_vec;
                spec->ops[matmulOpIndex].ps.fc_spec.num_slices = 1;
                spec->ops[matmulOpIndex].ps.fc_spec.slice_point[0] =
                    spec->ops[matmulOpIndex].ps.fc_spec.num_outputs;

                // Adjust the owner of the weight
                str_copy(spec->ws[transposeWeightIndex].op_name, spec->ops[matmulOpIndex].name,
                    NAME_LEN);
                spec->ws[transposeWeightIndex].bytes_of_vec = 0;

                setOperatorInvalid(spec, transposeOpIndex);
                i = matmulOpIndex;
            }
        }
        return hasOptimized;
    }
};
#endif
