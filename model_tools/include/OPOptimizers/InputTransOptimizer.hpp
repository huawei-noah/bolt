// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_InputTransOPTIMIZER
#define _H_InputTransOPTIMIZER

#include "OPOptimizer.hpp"
#include "shape_infer.h"

class InputTransOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        std::set<std::string> skipLists = {};
        bool hasOptimized = false;
        std::map<std::string, int> modelInput;
        for (int i = 0; i < spec->num_inputs; i++) {
            std::string name = spec->input_names[i];
            if (skipLists.find(name) == skipLists.end()) {
                modelInput[name] = i;
            }
        }
        for (int i = 0; i < spec->num_operator_specs; i++) {
            U32 j = 0;
            if (spec->ops[i].type == OT_Gather) {
                j = 1;
            }
            std::string name;
            if (spec->ops[i].num_inputs > j) {
                name = spec->ops[i].input_tensors_name[j];
                if (modelInput.find(name) == modelInput.end()) {
                    continue;
                }
                auto next = searchOperatorIndexByInput(spec, name, 0, spec->num_operator_specs);
                if (next.size() != 1) {
                    continue;
                }
            } else {
                continue;
            }
            int id = modelInput[name];
            TensorDesc iDesc = spec->input_dims[id];
            TensorDesc oDesc = spec->input_dims[id];
            bool opt = false;
            switch (spec->ops[i].type) {
                case OT_Transpose: {
                    auto p = spec->ops[i].ps.transpose_spec;
                    if ((p.num_axes == 4 && p.axes[0] == 0 && p.axes[1] == 3 && p.axes[2] == 1 &&
                            p.axes[3] == 2) ||
                        (p.num_axes == 3 && p.axes[0] == 0 && p.axes[1] == 2 && p.axes[2] == 1)) {
                        CHECK_STATUS(transpose_infer_output_size_cpu(iDesc, p, &oDesc));
                        setOperatorInvalid(spec, i, true);
                        opt = true;
                    }
                    break;
                }
#if 0
                case OT_Reshape:
                    CHECK_STATUS(
                        reshape_infer_output_size_cpu(iDesc, spec->ops[i].ps.reshape_spec, &oDesc));
                    setOperatorInvalid(spec, i, true);
                    opt = true;
                    break;
                case OT_Squeeze:
                    CHECK_STATUS(
                        squeeze_infer_output_size_cpu(iDesc, spec->ops[i].ps.squeeze_spec, &oDesc));
                    setOperatorInvalid(spec, i, true);
                    opt = true;
                    break;
                case OT_Unsqueeze:
                    CHECK_STATUS(unsqueeze_infer_output_size_cpu(
                        iDesc, spec->ops[i].ps.unsqueeze_spec, &oDesc));
                    setOperatorInvalid(spec, i, true);
                    opt = true;
                    break;
#endif
                case OT_Embedding: {
                    int weightIdx = searchWeightIndex(spec, spec->ops[i].name);
                    if (spec->ws[weightIdx].bytes_of_weight > 0) {
                        oDesc.dt = DT_U32;
                        opt = true;
                    }
                    break;
                }
                case OT_Cast: {
                    if (iDesc.dt == spec->ops[i].ps.cast_spec.dt) {
                        setOperatorInvalid(spec, i, true, 0);
                        opt = true;
                    }
                    break;
                }
                case OT_Gather: {
                    oDesc.dt = DT_U32;
                    opt = true;
                    break;
                }
                default:
                    break;
            }
            if (opt) {
                UNI_INFO_LOG("change model input(%s) from (%s) to (%s)\n", name.c_str(),
                    tensorDesc2Str(iDesc).c_str(), tensorDesc2Str(oDesc).c_str());
                spec->input_dims[id] = oDesc;
                hasOptimized = true;
            }
        }
        if (hasOptimized) {
            UNI_WARNING_LOG("If you don't want to change input dimension, you can add your input "
                            "name into 'skipLists' in this function.\n");
        }
        return hasOptimized;
    }
};
#endif
