// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_INPLACEOPTIMIZER
#define _H_INPLACEOPTIMIZER

#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class InPlaceOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        bool hasOptimized = false;
        std::vector<std::string> unrepeatedInputNames;
        std::vector<std::string> repeated;

        // Insert pass
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (isInPlaceOp(spec->ops[i].type)) {
                CHECK_REQUIREMENT(1 == spec->ops[i].num_inputs);
                std::string inputName = spec->ops[i].input_tensors_name[0];
                if (find(unrepeatedInputNames.begin(), unrepeatedInputNames.end(), inputName) == unrepeatedInputNames.end()) {
                    unrepeatedInputNames.push_back(inputName);
                } else {
                    repeated.push_back(inputName);
                }
            }
        }

        for (std::string name : repeated) {
            std::vector<std::string>::iterator it = find(unrepeatedInputNames.begin(), unrepeatedInputNames.end(), name);
            if (it != unrepeatedInputNames.end()) {
                unrepeatedInputNames.erase(it);
            }
        }

        // Erase pass
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (OT_None == spec->ops[i].type || isInPlaceOp(spec->ops[i].type)) {
                continue;
            }

            for (U32 j = 0; j < spec->ops[i].num_inputs; j++) {
                std::string inputName = spec->ops[i].input_tensors_name[j];
                std::vector<std::string>::iterator it = find(unrepeatedInputNames.begin(), unrepeatedInputNames.end(), inputName);
                if (it != unrepeatedInputNames.end()) {
                    unrepeatedInputNames.erase(it);
                }
            }
        }

        for (int i = spec->num_operator_specs - 1; i >= 0; i--) {
            if (isInPlaceOp(spec->ops[i].type)) {
                CHECK_REQUIREMENT(spec->ops[i].num_inputs == 1);
                std::string inputName = spec->ops[i].input_tensors_name[0];
                if (find(unrepeatedInputNames.begin(), unrepeatedInputNames.end(), inputName) == unrepeatedInputNames.end()) {
                    // Input is used multiple times, so should not be in-place
                    continue;
                }

                CHECK_REQUIREMENT(spec->ops[i].num_outputs == 1);
                str_copy(spec->ops[i].input_tensors_name[0], spec->ops[i].output_tensors_name[0], NAME_LEN);
                hasOptimized = true;

                I32 found = 0;
                for (int j = i - 1; j >= 0; j--) {
                    if (spec->ops[j].type != OT_None) {
                        for (U32 k = 0; k < spec->ops[j].num_outputs; k++) {
                            std::string prevOutputName = spec->ops[j].output_tensors_name[k];
                            if (prevOutputName == inputName) {
                                str_copy(spec->ops[j].output_tensors_name[k], spec->ops[i].input_tensors_name[0], NAME_LEN);
                                found = 1;
                                break;
                            }
                        }
                    }
                    if (1 == found) {
                        break;
                    }
                }

                if (0 == found) {
                    std::string newName = spec->ops[i].input_tensors_name[0];
                    std::cout << "[Warning] in-place tensor seems to be model input: " << inputName << ". New name: " << newName <<std::endl;
                }
            }
        }
        return hasOptimized;
    }

public:
    static bool isInPlaceOp(OperatorType opType)
    {
        switch (opType) {
            case OT_Relu: {
                return true;
            }
            case OT_Relu6: {
                return true;
            }
            case OT_HSwish: {
                return true;
            }
            case OT_HSigmoid: {
                return true;
            }
            case OT_Sigmoid: {
                return true;
            }
            case OT_Gelu: {
                return true;
            }
            case OT_TanH: {
                return true;
            }
            case OT_Logistic: {
                return true;
            }
            default: {
                return false;
            }
        }
    }
};
#endif
