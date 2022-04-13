// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MERGESHAREDWEIGHTOPTIMIZER
#define _H_MERGESHAREDWEIGHTOPTIMIZER

#include "OPOptimizer.hpp"
#include "model_print.h"

class MergeSharedWeightOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_MatMul) {
                // Merge SharedWeight
                std::string curIn = spec->ops[i].input_tensors_name[1];
                auto prevOpIndexes = searchOperatorIndexByOutput(spec, curIn, 0, i);
                if (prevOpIndexes.size() != 1 ||
                    OT_SharedWeight != spec->ops[prevOpIndexes[0].first].type) {
                    continue;
                }
                int sharedWeightIdx = prevOpIndexes[0].first;

                if (spec->ops[sharedWeightIdx].ps.shared_weight_spec.desc.nDims > 2) {
                    continue;  // it's not clear how to support
                }

                std::string sharedOut = spec->ops[sharedWeightIdx].output_tensors_name[0];
                auto nextOpIndexes = searchOperatorIndexByInput(
                    spec, sharedOut, sharedWeightIdx + 1, spec->num_operator_specs, false);
                if (nextOpIndexes.size() > 1) {
                    continue;
                }

                hasOptimized = true;

                int matmulOpIndex = i;
                bool transpose_b = spec->ops[i].ps.matmul_spec.transpose_b;
                // Update matmul to fc
                spec->ops[matmulOpIndex].type = OT_FC;
                spec->ops[matmulOpIndex].num_inputs = 1;
                mt_free(spec->ops[matmulOpIndex].input_tensors_name[1]);
                spec->ops[matmulOpIndex].ps.fc_spec.num_outputs =
                    spec->ops[sharedWeightIdx].ps.shared_weight_spec.desc.dims[1];
                spec->ops[matmulOpIndex].ps.fc_spec.num_slices = 1;
                spec->ops[matmulOpIndex].ps.fc_spec.slice_point[0] =
                    spec->ops[matmulOpIndex].ps.fc_spec.num_outputs;

                int matmulWsIndex = searchWeightIndex(spec, spec->ops[matmulOpIndex].name);
                int sharedWsIndex = searchWeightIndex(spec, spec->ops[sharedWeightIdx].name);
                U32 weightSize = spec->ws[sharedWsIndex].bytes_of_weight;
                U8 *weight = (U8 *)mt_malloc(weightSize);

                if (transpose_b) {
                    UNI_MEMCPY(weight, spec->ws[sharedWsIndex].weight, weightSize);
                } else {
                    // transpose
                    U32 row = spec->ops[sharedWeightIdx].ps.shared_weight_spec.desc.dims[1];
                    U32 column = spec->ops[sharedWeightIdx].ps.shared_weight_spec.desc.dims[0];
                    U32 unit = bytesOf(spec->ws[sharedWsIndex].mdt);
                    for (U32 r = 0; r < row; ++r) {
                        for (U32 c = 0; c < column; ++c) {
                            UNI_MEMCPY(weight + (c * row + r) * unit,
                                spec->ws[sharedWsIndex].weight + (r * column + c) * unit, unit);
                        }
                    }
                }
                spec->ws[matmulWsIndex].weight = weight;
                spec->ws[matmulWsIndex].bytes_of_weight = weightSize;
                spec->ws[matmulWsIndex].bytes_of_vec = 0;

                sharedOut = spec->ops[sharedWeightIdx].output_tensors_name[0];
                nextOpIndexes = searchOperatorIndexByInput(
                    spec, sharedOut, sharedWeightIdx + 1, spec->num_operator_specs, false);
                if (nextOpIndexes.size() == 0) {
                    setOperatorInvalid(spec, sharedWeightIdx, true);
                }

                // Merge Eltwise + SharedWeight to FC bias
                std::string curOut = spec->ops[i].output_tensors_name[0];
                nextOpIndexes = searchOperatorIndexByInput(
                    spec, curOut, i + 1, spec->num_operator_specs, false);
                if (nextOpIndexes.size() != 1 ||
                    OT_Eltwise != spec->ops[nextOpIndexes[0].first].type) {
                    continue;
                }
                int eltwiseIdx = nextOpIndexes[0].first;
                int order = nextOpIndexes[0].second;
                std::string sharedweightName;
                if (order == 0) {
                    sharedweightName = spec->ops[eltwiseIdx].input_tensors_name[1];
                } else if (order == 1) {
                    sharedweightName = spec->ops[eltwiseIdx].input_tensors_name[0];
                }
                prevOpIndexes = searchOperatorIndexByOutput(spec, sharedweightName, 0, eltwiseIdx);
                if (prevOpIndexes.size() != 1 ||
                    OT_SharedWeight != spec->ops[prevOpIndexes[0].first].type) {
                    continue;
                }
                sharedWeightIdx = prevOpIndexes[0].first;
                sharedWsIndex = searchWeightIndex(spec, spec->ops[sharedWeightIdx].name);
                if (spec->ws[sharedWsIndex].bytes_of_weight / bytesOf(spec->ws[sharedWsIndex].mdt) ==
                    spec->ops[matmulOpIndex].ps.fc_spec.num_outputs) {
                    U32 biasSize = spec->ws[sharedWsIndex].bytes_of_weight;
                    U8 *bias = (U8 *)mt_malloc(biasSize);
                    UNI_MEMCPY(bias, spec->ws[sharedWsIndex].weight, biasSize);
                    spec->ws[matmulWsIndex].vec = bias;
                    spec->ws[matmulWsIndex].bytes_of_vec = biasSize;

                    sharedOut = spec->ops[sharedWeightIdx].output_tensors_name[0];
                    nextOpIndexes = searchOperatorIndexByInput(
                        spec, sharedOut, sharedWeightIdx + 1, spec->num_operator_specs, false);
                    if (nextOpIndexes.size() == 1) {
                        setOperatorInvalid(spec, sharedWeightIdx, true);
                    }
                    setOperatorInvalid(spec, eltwiseIdx, true);
                }
            }
        }
        return hasOptimized;
    }
};
#endif
