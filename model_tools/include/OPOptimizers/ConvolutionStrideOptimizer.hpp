// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CONVOLUTIONSTRIDEOPTIMIZER
#define _H_CONVOLUTIONSTRIDEOPTIMIZER

#include "OPOptimizer.hpp"

/* for resnet50

   conv(56w3s1)       /                 conv(28w3s2)       /
         \          /                         \          /   
   conv(56w1s1)   /           ==>       conv(28w1s1)  pool(28w1s2)
           \    /                               \    /
            elt                                  elt
          /     \                             /      \ 
  conv(28w1s2) conv(28w1s2)           conv(28w1s1)  conv(28w1s1)
*/
class ConvolutionStrideOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Eltwise && spec->ops[i].num_inputs >= 2) {
                std::vector<int> conv1x1Vector, convVector;
                std::string curOut = spec->ops[i].output_tensors_name[0];
                auto nextOpIndexes = searchOperatorIndexByInput(
                    spec, curOut, i + 1, spec->num_operator_specs, false);
                if (nextOpIndexes.size() == 0) {
                    continue;
                }

                std::string curIn0 = spec->ops[i].input_tensors_name[0];
                auto prevOpIndexes0 = searchOperatorIndexByOutput(spec, curIn0, 0, i);
                std::string curIn1 = spec->ops[i].input_tensors_name[1];
                auto prevOpIndexes1 = searchOperatorIndexByOutput(spec, curIn1, 0, i);

                if (prevOpIndexes0.size() != 1 || prevOpIndexes1.size() != 1) {
                    continue;
                }

                bool thisOptimized = true;

                // search forward
                for (auto nextOpIndex : nextOpIndexes) {
                    int opIdx = nextOpIndex.first;
                    if (spec->ops[opIdx].type == OT_Conv &&
                        spec->ops[opIdx].ps.conv_spec.kernel_h == 1 &&
                        spec->ops[opIdx].ps.conv_spec.kernel_w == 1 &&
                        spec->ops[opIdx].ps.conv_spec.stride_h > 1 &&
                        spec->ops[opIdx].ps.conv_spec.stride_w > 1) {
                        conv1x1Vector.push_back(opIdx);
                        if (spec->ops[opIdx].ps.conv_spec.stride_h !=
                                spec->ops[conv1x1Vector[0]].ps.conv_spec.stride_h ||
                            spec->ops[opIdx].ps.conv_spec.stride_w !=
                                spec->ops[conv1x1Vector[0]].ps.conv_spec.stride_w) {
                            thisOptimized = false;
                            break;
                        }
                    } else {
                        thisOptimized = false;
                        break;
                    }
                }
                if (!thisOptimized) {
                    continue;
                }

                bool inputNullBranch = false;
                int eltwiseInputIdx = 0, branchStartIdx = 0;
                std::vector<std::pair<int, int>> prevOpIndexes = {
                    prevOpIndexes0[0], prevOpIndexes1[0]};

                // search backward
                for (U32 j = 0; j < prevOpIndexes.size(); j++) {
                    auto prevOpIndex = prevOpIndexes[j];
                    int opIdx = prevOpIndex.first;

                    // null branch, should insert pooling
                    std::string nextOut = spec->ops[opIdx].output_tensors_name[0];
                    auto nextOutOpIndexes =
                        searchOperatorIndexByInput(spec, nextOut, opIdx, spec->num_operator_specs);
                    if (nextOutOpIndexes.size() > 1) {
                        branchStartIdx = opIdx;
                        eltwiseInputIdx = j;
                        inputNullBranch = true;
                        continue;
                    }

                    if (spec->ops[opIdx].type == OT_Conv &&
                        spec->ops[opIdx].ps.conv_spec.kernel_h == 1 &&
                        spec->ops[opIdx].ps.conv_spec.kernel_w == 1) {
                        int searchIdx = opIdx;
                        while (isValidOperator(spec, searchIdx)) {
                            if (spec->ops[searchIdx].type == OT_Conv &&
                                spec->ops[searchIdx].ps.conv_spec.kernel_h == 1 &&
                                spec->ops[searchIdx].ps.conv_spec.kernel_w == 1 &&
                                spec->ops[searchIdx].ps.conv_spec.stride_h > 1 &&
                                spec->ops[searchIdx].ps.conv_spec.stride_w > 1) {
                                conv1x1Vector.push_back(searchIdx);
                            } else if (spec->ops[searchIdx].type == OT_Conv &&
                                spec->ops[searchIdx].ps.conv_spec.stride_h == 1 &&
                                spec->ops[searchIdx].ps.conv_spec.stride_w == 1) {
                                if (spec->ops[searchIdx].ps.conv_spec.kernel_h > 1 &&
                                    spec->ops[searchIdx].ps.conv_spec.kernel_w > 1) {
                                    convVector.push_back(searchIdx);
                                    break;
                                }
                            } else {
                                thisOptimized = false;
                                break;
                            }
                            std::string curX = spec->ops[searchIdx].input_tensors_name[0];
                            auto prevXIndexes =
                                searchOperatorIndexByOutput(spec, curX, 0, searchIdx);
                            if (prevXIndexes.size() != 1) {
                                thisOptimized = false;
                                break;
                            }
                            searchIdx = prevXIndexes[0].first;
                        }
                    } else if (spec->ops[opIdx].type == OT_Conv &&
                        spec->ops[opIdx].ps.conv_spec.stride_h == 1 &&
                        spec->ops[opIdx].ps.conv_spec.stride_w == 1) {
                        convVector.push_back(opIdx);
                    } else {
                        thisOptimized = false;
                    }

                    if (!thisOptimized) {
                        break;
                    }
                }

                // set the stride and insert pooling op
                int thisStideH = 0, thisStideW = 0;
                if (thisOptimized) {
                    for (int strideIdx : convVector) {
                        thisStideH = spec->ops[conv1x1Vector[0]].ps.conv_spec.stride_h;
                        thisStideW = spec->ops[conv1x1Vector[0]].ps.conv_spec.stride_w;
                        spec->ops[strideIdx].ps.conv_spec.stride_h = thisStideH;
                        spec->ops[strideIdx].ps.conv_spec.stride_w = thisStideW;
                    }
                    for (int strideIdx : conv1x1Vector) {
                        spec->ops[strideIdx].ps.conv_spec.stride_h = 1;
                        spec->ops[strideIdx].ps.conv_spec.stride_w = 1;
                    }
                    if (inputNullBranch) {
                        std::string poolingName = "Pooling_" + std::to_string(branchStartIdx) +
                            std::to_string(eltwiseInputIdx);
                        OperatorSpec poolOperator =
                            mt_create_operator(poolingName.c_str(), OT_Pooling, 1, 1);
                        poolOperator.ps.pooling_spec.kernel_h = 1;
                        poolOperator.ps.pooling_spec.kernel_w = 1;
                        poolOperator.ps.pooling_spec.kernel_t = 1;
                        poolOperator.ps.pooling_spec.stride_h = thisStideH;
                        poolOperator.ps.pooling_spec.stride_w = thisStideW;
                        poolOperator.ps.pooling_spec.stride_t = 1;
                        poolOperator.ps.pooling_spec.padding_before = 0;
                        poolOperator.ps.pooling_spec.padding_after = 0;
                        poolOperator.ps.pooling_spec.padding_top = 0;
                        poolOperator.ps.pooling_spec.padding_bottom = 0;
                        poolOperator.ps.pooling_spec.padding_left = 0;
                        poolOperator.ps.pooling_spec.padding_right = 0;
                        poolOperator.ps.pooling_spec.rm = FLOOR;
                        poolOperator.ps.pooling_spec.mode = POOLING_MAX;

                        str_copy(poolOperator.output_tensors_name[0], poolingName.data(),
                            strlen(poolingName.data()));
                        str_copy(poolOperator.input_tensors_name[0],
                            spec->ops[branchStartIdx].output_tensors_name[0],
                            strlen(spec->ops[branchStartIdx].output_tensors_name[0]));
                        str_copy(spec->ops[i].input_tensors_name[eltwiseInputIdx],
                            poolingName.data(), strlen(poolingName.data()));
                        mt_insert_operator(spec, branchStartIdx + 1, poolOperator);
                        i += 1;
                    }
                    hasOptimized = true;
                }
            }
        }

        return hasOptimized;
    }
};
#endif
