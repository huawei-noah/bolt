// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TransConcatTransOPTIMIZER
#define _H_TransConcatTransOPTIMIZER

#include "OPOptimizer.hpp"

class TransConcatTransOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        std::set<std::string> modelOutput;
        for (int i = 0; i < spec->num_outputs; i++) {
            modelOutput.insert(spec->output_names[i]);
        }
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Concat) {
                if (!(spec->ops[i].ps.concat_spec.axis == -1 ||
                        spec->ops[i].ps.concat_spec.axis == 3)) {
                    continue;
                }
                std::vector<int> inIndex, outIndex;
                std::vector<std::pair<int, int>> insertIndex;
                bool tag1 = true;
                bool tag2 = true;
                for (U32 j = 0; j < spec->ops[i].num_inputs; j++) {
                    auto tmpVec =
                        searchOperatorIndexByOutput(spec, spec->ops[i].input_tensors_name[j], 0, i);
                    if (tmpVec.size() != 1) {
                        tag1 = false;
                        break;
                    }
                    int lastIndex = tmpVec[0].first;
                    auto transPs = spec->ops[lastIndex].ps.transpose_spec;
                    if (spec->ops[lastIndex].type != OT_Transpose || transPs.trans_size != 4 ||
                        transPs.trans_dims[0] != 0 || transPs.trans_dims[1] != 2 ||
                        transPs.trans_dims[2] != 3 || transPs.trans_dims[3] != 1) {
                        tag1 = false;
                        break;
                    }
                    tmpVec = searchOperatorIndexByInput(spec, spec->ops[i].input_tensors_name[j],
                        lastIndex + 1, spec->num_operator_specs);
                    if (tmpVec.size() != 1) {
                        tag1 = false;
                        break;
                    }
                    inIndex.push_back(lastIndex);
                }
                if (!tag1) {
                    continue;
                }

                auto tmpVec = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (tmpVec.size() < 1) {
                    tag2 = false;
                    continue;
                }
                for (U32 k = 0; k < tmpVec.size(); k++) {
                    int nextIndex = tmpVec[k].first;
                    auto transPs = spec->ops[nextIndex].ps.transpose_spec;
                    if (spec->ops[nextIndex].type != OT_Transpose || transPs.trans_size != 4 ||
                        transPs.trans_dims[0] != 0 || transPs.trans_dims[1] != 3 ||
                        transPs.trans_dims[2] != 1 || transPs.trans_dims[3] != 2) {
                        tag2 = false;
                        insertIndex.push_back(tmpVec[k]);
                    } else {
                        outIndex.push_back(nextIndex);
                    }
                }

                if (modelOutput.find(spec->ops[i].output_tensors_name[0]) != modelOutput.end()) {
                    continue;
                }

                for (auto item : inIndex) {
                    setOperatorInvalid(spec, item, true);
                }
                for (auto item : outIndex) {
                    setOperatorInvalid(spec, item, true);
                }
                spec->ops[i].ps.concat_spec.axis = 1;
                if (!tag2) {
                    TransposeParamSpec transPs;
                    transPs.trans_size = 4;
                    transPs.trans_dims[0] = 0;
                    transPs.trans_dims[1] = 2;
                    transPs.trans_dims[2] = 3;
                    transPs.trans_dims[3] = 1;
                    std::string name = "concat_transpose_" + std::to_string(i);
                    OperatorSpec transposeOperator =
                        mt_create_operator(name.c_str(), OT_Transpose, 1, 1);
                    transposeOperator.ps.transpose_spec = transPs;
                    str_copy(transposeOperator.input_tensors_name[0],
                        spec->ops[i].output_tensors_name[0], NAME_LEN);
                    str_copy(transposeOperator.output_tensors_name[0], name.c_str(), name.size());
                    for (U32 k = 0; k < insertIndex.size(); k++) {
                        str_copy(
                            spec->ops[insertIndex[k].first].input_tensors_name[insertIndex[k].second],
                            name.c_str(), name.size());
                    }
                    mt_insert_operator(spec, i + 1, transposeOperator);
                }
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }
};
#endif
