// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_EltwiseConstantOPTIMIZER
#define _H_EltwiseConstantOPTIMIZER

#include "OPOptimizer.hpp"

class EltwiseConstantOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Eltwise && spec->ops[i].num_inputs == 2) {
                std::vector<std::vector<std::pair<int, int>>> elts;
                for (U32 j = 0; j < spec->ops[i].num_inputs; j++) {
                    auto prev =
                        searchOperatorIndexByOutput(spec, spec->ops[i].input_tensors_name[j], 0, i);
                    elts.push_back(prev);
                }
                int shape = -1, constant = -1, exp = -1, branch = -1;
                float value = -1;
                for (U32 j = 0; j < elts.size(); j++) {
                    if (elts[j].size() != 1) {
                        continue;
                    }
                    if (spec->ops[elts[j][0].first].type == OT_ConstantOfShape) {
                        branch = j;
                        constant = elts[j][0].first;
                        value = spec->ops[elts[j][0].first].ps.constant_of_shape_spec.value;
                    }
                    if (spec->ops[elts[j][0].first].type == OT_Exp) {
                        exp = elts[j][0].first;
                        auto next = searchOperatorIndexByInput(spec,
                            spec->ops[exp].output_tensors_name[0], exp + 1,
                            spec->num_operator_specs);
                        if (searchOperatorIndexByInput(spec, spec->ops[exp].output_tensors_name[0],
                                exp + 1, spec->num_operator_specs)
                                .size() != 1) {
                            exp = -1;
                            continue;
                        }
                        auto prev = searchOperatorIndexByOutput(
                            spec, spec->ops[exp].input_tensors_name[0], 0, exp);
                        if (prev.size() != 1) {
                            exp = -1;
                            continue;
                        }
                        if (spec->ops[prev[0].first].type == OT_ConstantOfShape) {
                            branch = j;
                            constant = prev[0].first;
                            value = spec->ops[elts[j][0].first].ps.constant_of_shape_spec.value;
                        } else {
                            exp = -1;
                        }
                    }
                }
                if (constant == -1 ||
                    searchOperatorIndexByInput(spec, spec->ops[constant].output_tensors_name[0],
                        constant + 1, spec->num_operator_specs)
                            .size() != 1) {
                    continue;
                }
                auto prev = searchOperatorIndexByOutput(
                    spec, spec->ops[constant].input_tensors_name[0], 0, constant);
                if (prev.size() != 1) {
                    continue;
                }
                shape = prev[0].first;
                if (spec->ops[shape].type != OT_Shape ||
                    searchOperatorIndexByInput(spec, spec->ops[shape].output_tensors_name[0],
                        shape + 1, spec->num_operator_specs)
                            .size() != 1) {
                    continue;
                }
                if (spec->ops[i].ps.eltwise_spec.mode == ELTWISE_SUM) {
                    if (exp == -1 || value == 0) {
                        setOperatorInvalid(spec, shape, true);
                        setOperatorInvalid(spec, constant, true);
                        UNI_STRCPY(spec->ops[i].input_tensors_name[branch],
                            spec->ops[i].input_tensors_name[1 - branch]);
                        setOperatorInvalid(spec, i, true);
                    }
                }
                if (spec->ops[i].ps.eltwise_spec.mode == ELTWISE_PROD) {
                    if (exp != -1 || value == 1) {
                        setOperatorInvalid(spec, shape, true);
                        setOperatorInvalid(spec, constant, true);
                        if (exp != -1) {
                            setOperatorInvalid(spec, exp, true);
                        }
                        UNI_STRCPY(spec->ops[i].input_tensors_name[branch],
                            spec->ops[i].input_tensors_name[1 - branch]);
                        setOperatorInvalid(spec, i, true);
                        hasOptimized = true;
                    }
                }
            }
        }
        return hasOptimized;
    }
};
#endif
