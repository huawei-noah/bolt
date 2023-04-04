// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TRANSPOSE_CONV_OPTIMIZER
#define _H_TRANSPOSE_CONV_OPTIMIZER

#include "OPOptimizer.hpp"

class TransposeConvOptimizer : public OPOptimizer {
    bool recursive(ModelSpec *spec, int i, int j)
    {
        if (nchwc8.find(spec->ops[i].type) != nchwc8.end()) {
            return true;
        }
        if ((spec->ops[i].type == OT_Eltwise || spec->ops[i].type == OT_Concat) &&
            spec->ops[i].num_inputs == 2) {
            auto prevOpIndexes =
                searchOperatorIndexByOutput(spec, spec->ops[i].input_tensors_name[1 - j], 0, i);
            if (prevOpIndexes.size() == 0) {
                int id = -1;
                for (int k = 0; k < spec->num_inputs; k++) {
                    if (spec->input_names[k] == std::string(spec->ops[i].input_tensors_name[1 - j])) {
                        id = k;
                        break;
                    }
                }
                if (id == -1) {
                    UNI_ERROR_LOG(
                        "encounter unknown tensor %s.\n", spec->ops[i].input_tensors_name[1 - j]);
                    return false;
                }
                auto nextop = searchOperatorIndexByInput(
                    spec, spec->input_names[id], 0, spec->num_operator_specs);
                if (nextop.size() != 1) {
                    return false;
                }
                nextop = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                U32 count = 0;
                for (U32 k = 0; k < nextop.size(); k++) {
                    count += this->recursive(spec, nextop[k].first, nextop[k].second);
                }
                if (count == nextop.size()) {
                    TensorDesc iDesc = spec->input_dims[id];
                    spec->input_dims[id].df = DF_NCHWC8;
                    UNI_WARNING_LOG("change input %s dimension from %s to %s.\n",
                        spec->input_names[id], tensorDesc2Str(iDesc).c_str(),
                        tensorDesc2Str(spec->input_dims[id]).c_str());
                    return true;
                } else {
                    return false;
                }
            }
            //if ((prevOpIndexes.size() != 1) || (-1 == prevOpIndexes[0].first)) {
            //    return false;
            //}
            if (nchwc8.find(spec->ops[prevOpIndexes[0].first].type) == nchwc8.end() &&
                bypass.find(spec->ops[prevOpIndexes[0].first].type) == bypass.end()) {
                return false;
            }
            return true;
        }
        if (bypass.find(spec->ops[i].type) != bypass.end()) {
            auto nextop = searchOperatorIndexByInput(
                spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
            U32 count = 0;
            for (U32 k = 0; k < nextop.size(); k++) {
                count += this->recursive(spec, nextop[k].first, nextop[k].second);
            }
            return (count == nextop.size());
        }
        return false;
    }

    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Transpose &&
                ((spec->ops[i].ps.transpose_spec.num_axes == 4 &&
                     spec->ops[i].ps.transpose_spec.axes[0] == 0 &&
                     spec->ops[i].ps.transpose_spec.axes[1] == 3 &&
                     spec->ops[i].ps.transpose_spec.axes[2] == 1 &&
                     spec->ops[i].ps.transpose_spec.axes[3] == 2) ||
                    (spec->ops[i].ps.transpose_spec.num_axes == 3 &&
                        spec->ops[i].ps.transpose_spec.axes[0] == 0 &&
                        spec->ops[i].ps.transpose_spec.axes[1] == 2 &&
                        spec->ops[i].ps.transpose_spec.axes[2] == 1))) {
                auto nextop = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (nextop.size() == 0) {
                    continue;
                }
                U32 count = 0;
                for (U32 j = 0; j < nextop.size(); j++) {
                    count += this->recursive(spec, nextop[j].first, nextop[j].second);
                }
                if (count == nextop.size()) {
                    spec->ops[i].ps.transpose_spec.df = DF_NCHWC8;
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }

private:
    std::set<OperatorType> nchwc8 = {OT_Conv, OT_Pooling};
    std::set<OperatorType> bypass = {OT_Where, OT_Reshape};
};
#endif
