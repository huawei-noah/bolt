// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CONVOLUTIONELTWISEOPTIMIZER
#define _H_CONVOLUTIONELTWISEOPTIMIZER

#include <unordered_map>
#include "OPOptimizer.hpp"

/* fuse convolution and eltwise

         \     /                              \      / 
        conv  op             --->              \    op
          \  /                                  \  / 
        eltwise                             conv_eltwise
*/

class ConvolutionEltwiseOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        const int queryNum = 1;
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Eltwise && spec->ops[i].num_inputs == 2 &&
                spec->ops[i].ps.eltwise_spec.elt_mode == ELTWISE_SUM) {
                bool thisOptimized = false;
                int fuseConvIdx = -1;
                std::string curIn = spec->ops[i].input_tensors_name[0];
                auto prevOpIndexes0 = searchOperatorIndexByOutput(spec, curIn, 0, i);
                int curInIdx = prevOpIndexes0.size() > 0 ? prevOpIndexes0[0].first : -1;
                auto nextOpIndexes = searchOperatorIndexByInput(
                    spec, curIn, curInIdx + 1, spec->num_operator_specs, false);

                int nextNodeNum = 1;
                for (auto nextOpIndex : nextOpIndexes) {
                    if (isValidOperator(spec, nextOpIndex.first) &&
                        strcmp(spec->ops[nextOpIndex.first].name, spec->ops[i].name)) {
                        ++nextNodeNum;
                    }
                }
                if (nextNodeNum == 1 && isValidOperator(spec, curInIdx) &&
                    spec->ops[curInIdx].type == OT_Conv &&
                    spec->ops[curInIdx].ps.conv_spec.convolution_type == Convolution_Pointwise &&
                    spec->ops[curInIdx].ps.conv_spec.pw_activation_type == ACTIVATION_NULL) {
                    thisOptimized = true;
                    fuseConvIdx = curInIdx;
                    curInIdx = 0;
                }

                if (!thisOptimized) {
                    nextNodeNum = 1;
                    curIn = spec->ops[i].input_tensors_name[1];
                    prevOpIndexes0 = searchOperatorIndexByOutput(spec, curIn, 0, i);
                    curInIdx = prevOpIndexes0.size() > 0 ? prevOpIndexes0[0].first : -1;
                    nextOpIndexes = searchOperatorIndexByInput(
                        spec, curIn, curInIdx + 1, spec->num_operator_specs, false);
                    for (auto nextOpIndex : nextOpIndexes) {
                        if (isValidOperator(spec, nextOpIndex.first) &&
                            strcmp(spec->ops[nextOpIndex.first].name, spec->ops[i].name)) {
                            ++nextNodeNum;
                        }
                    }
                    if (nextNodeNum == 1 && isValidOperator(spec, curInIdx) &&
                        spec->ops[curInIdx].type == OT_Conv &&
                        spec->ops[curInIdx].ps.conv_spec.convolution_type == Convolution_Pointwise &&
                        spec->ops[curInIdx].ps.conv_spec.pw_activation_type == ACTIVATION_NULL) {
                        thisOptimized = true;
                        fuseConvIdx = curInIdx;
                        curInIdx = 1;
                    }
                }

                if (thisOptimized && isValidOperator(spec, fuseConvIdx)) {
                    hasOptimized = true;

                    std::string nodeName = "fuse_add_" + std::string(spec->ops[i].name);
                    if (curInIdx == 1) {
                        str_copy(spec->ops[i].input_tensors_name[1],
                            spec->ops[i].input_tensors_name[0],
                            strlen(spec->ops[i].input_tensors_name[0]));
                    }
                    str_copy(spec->ops[i].input_tensors_name[0],
                        spec->ops[fuseConvIdx].input_tensors_name[0],
                        strlen(spec->ops[fuseConvIdx].input_tensors_name[0]));
                    str_copy(spec->ops[i].name, nodeName.data(), strlen(nodeName.data()));

                    spec->ops[fuseConvIdx].ps.conv_spec.pw_activation_type =
                        spec->ops[i].ps.eltwise_spec.activation_type;
                    spec->ops[fuseConvIdx].ps.conv_spec.activation_spec.relu_spec =
                        spec->ops[i].ps.eltwise_spec.activation_spec.relu_spec;

                    spec->ops[i].type = OT_Conv;
                    spec->ops[i].ps.conv_spec = spec->ops[fuseConvIdx].ps.conv_spec;
                    int convWeightIndex = searchWeightIndex(spec, spec->ops[fuseConvIdx].name);
                    str_copy(spec->ws[convWeightIndex].op_name, nodeName.data(),
                        strlen(nodeName.data()));

                    setOperatorInvalid(spec, fuseConvIdx, true);
                }
            }
        }
        return hasOptimized;
    }
};
#endif
