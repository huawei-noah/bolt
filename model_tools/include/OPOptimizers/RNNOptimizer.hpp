// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_RNNOPTIMIZER
#define _H_RNNOPTIMIZER

#include <string>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class RNNOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        OperatorType queryShape[1] = {OT_Shape};

        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_RNN && spec->ops[i].num_inputs >= 1) {
                int rnnOpIndex = i;
                std::string rnn_input = spec->ops[rnnOpIndex].input_tensors_name[0];
                int shapeOpIndex = searchOperatorIndexBackward(spec, i - 1, queryShape, 1, false);
                if (shapeOpIndex < 1) {
                    UNI_WARNING_LOG("rnn %s can not be optimized\n", spec->ops[i].name);
                    continue;
                }

                std::string last_output = spec->ops[shapeOpIndex - 1].output_tensors_name[0];
                std::string shape_input = spec->ops[shapeOpIndex].input_tensors_name[0];
                if ((last_output == rnn_input) && (last_output == shape_input)) {
                    if (spec->ops[shapeOpIndex + 1].type == OT_Slice) {
                        if (spec->ops[shapeOpIndex + 2].type == OT_Unsqueeze) {
                            if (spec->ops[shapeOpIndex + 3].type == OT_Concat) {
                                if (spec->ops[shapeOpIndex + 4].type == OT_ConstantOfShape) {
                                    for (int j = 0; j < 5; j++) {
                                        setOperatorInvalid(spec, shapeOpIndex + j);
                                    }

                                    int rnn_original_input_size = spec->ops[rnnOpIndex].num_inputs;
                                    for (int k = 1; k < rnn_original_input_size; k++) {
                                        delete spec->ops[rnnOpIndex].input_tensors_name[k];
                                        spec->ops[rnnOpIndex].input_tensors_name[k] = nullptr;
                                    }
                                    spec->ops[rnnOpIndex].num_inputs = 1;
                                }
                            }
                        }
                    }
                }
                // bi-RNN
                if (rnnOpIndex + 2 < spec->num_operator_specs &&
                    spec->ops[rnnOpIndex + 1].type == OT_Transpose &&
                    spec->ops[rnnOpIndex + 2].type == OT_Reshape &&
                    spec->ops[rnnOpIndex].ps.rnn_spec.steps == -2) {
                    memcpy(spec->ops[rnnOpIndex].output_tensors_name[0],
                        spec->ops[rnnOpIndex + 2].output_tensors_name[0], NAME_LEN);
                    setOperatorInvalid(spec, rnnOpIndex + 1);
                    setOperatorInvalid(spec, rnnOpIndex + 2);
                }
            }
        }
        return hasOptimized;
    }
};
#endif
