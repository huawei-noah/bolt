// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ModifyDtOfInputOPTIMIZER
#define _H_ModifyDtOfInputOPTIMIZER

#include <map>
#include <string>
#include "OPOptimizer.hpp"

class ModifyDtOfInputOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        std::map<std::string, int> inputPositions;
        for (int i = 0; i < (int)spec->num_inputs; i++) {
            std::string inputName = spec->input_names[i];
            inputPositions[inputName] = i;
        }

        for (int i = 0; i < (int)spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Embedding) {
                for (int j = 0; j < (int)spec->ops[i].num_inputs; j++) {
                    std::string inputName = spec->ops[i].input_tensors_name[j];
                    if (inputPositions.find(inputName) != inputPositions.end()) {
                        spec->input_dims[inputPositions[inputName]].dt = DT_U32;
                        hasOptimized = true;
                    }
                }
            }
        }
        return hasOptimized;
    }
};
#endif
