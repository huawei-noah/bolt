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

class InputTransOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        std::map<std::string, int> modelInput;
        for (int i = 0; i < spec->num_inputs; i++) {
            modelInput[spec->input_names[i]] = i;
        }
        for (int i = 0; i < spec->num_operator_specs; i++) {
            std::string name;
            if (spec->ops[i].num_inputs > 0) {
                name = spec->ops[i].input_tensors_name[0];
                if (modelInput.find(name) == modelInput.end()) {
                    continue;
                }
                auto tmpVec = searchOperatorIndexByInput(spec, name, 0, spec->num_operator_specs);
                if (tmpVec.size() != 1) {
                    continue;
                }
            } else {
                continue;
            }
            auto transPs = spec->ops[i].ps.transpose_spec;
            int id = modelInput[name];
            if (spec->ops[i].type == OT_Transpose && transPs.num_axes == 4 && transPs.axes[0] == 0 &&
                transPs.axes[1] == 3 && transPs.axes[2] == 1 && transPs.axes[3] == 2) {
                setOperatorInvalid(spec, i, true);
                int c = spec->input_dims[id].dims[0];
                int w = spec->input_dims[id].dims[1];
                int h = spec->input_dims[id].dims[2];
                int n = spec->input_dims[id].dims[3];
                spec->input_dims[id].dims[0] = w;
                spec->input_dims[id].dims[1] = h;
                spec->input_dims[id].dims[2] = c;
                spec->input_dims[id].dims[3] = n;
                hasOptimized = true;
            }
            if (spec->ops[i].type == OT_Transpose && transPs.num_axes == 3 &&
                transPs.axes[0] == 0 && transPs.axes[1] == 2 && transPs.axes[2] == 1) {
                setOperatorInvalid(spec, i, true);
                int c = spec->input_dims[id].dims[0];
                int h = spec->input_dims[id].dims[1];
                int n = spec->input_dims[id].dims[2];
                spec->input_dims[id].dims[0] = h;
                spec->input_dims[id].dims[1] = c;
                spec->input_dims[id].dims[2] = n;
                hasOptimized = true;
            }
            if (spec->ops[i].type == OT_Embedding || spec->ops[i].type == OT_Gather) {
                if (spec->input_dims[modelInput[name]].dt == DT_F32 ||
                    spec->input_dims[modelInput[name]].dt == DT_F16) {
                    spec->input_dims[modelInput[name]].dt = DT_U32;
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
