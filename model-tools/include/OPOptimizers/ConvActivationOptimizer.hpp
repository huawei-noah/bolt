// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CONVACTIVATIONOPTIMIZER
#define _H_CONVACTIVATIONOPTIMIZER

#include <vector>
#include <string>
#include "model_tools.h"
#include "OPOptimizer.hpp"

// Optimize both Convolution and Deconvolution
class ConvActivationOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 2;
        OperatorType queryOps[queryNum] = {OT_Conv, OT_Deconvolution};
        bool hasOptimized = false;
        for (int i = 1; i< spec->num_operator_specs; i++) {
            OperatorType curOT = spec->ops[i].type;
            if (curOT == OT_Relu) {
                if (spec->ops[i].num_inputs > 0 && spec->ops[i].num_outputs > 0) {
                    std::string inputName = spec->ops[i].input_tensors_name[0];
                    std::string outputName = spec->ops[i].output_tensors_name[0];
                    if (inputName != outputName) {
                        std::cout << "[WARNING] encounter unoptimized Relu layer (not inPlace): " << spec->ops[i].name << std::endl;
                        continue;
                    }
                }
                int atOpIndex   = i;
                int convOpIndex = searchOperatorIndexBackward(spec, atOpIndex - 1, queryOps, queryNum);

                if (convOpIndex == -1) {
                    std::cout << "[WARNING] encounter unoptimized Relu layer (no Conv or Deconv before): " << spec->ops[atOpIndex].name << std::endl;
                    continue;
                }

                // tensor relationship rewrite
                str_copy(spec->ops[convOpIndex].output_tensors_name[0], spec->ops[atOpIndex].output_tensors_name[0], NAME_LEN);
                hasOptimized = true;

                switch (spec->ops[convOpIndex].ps.conv_spec.convolution_type) {
                    case Convolution_Pointwise: {
                        spec->ops[convOpIndex].ps.conv_spec.pw_activation_type = ACTIVATION_RELU;
                        break;
                    }
                    case Convolution_Deconvolution: {
                        spec->ops[convOpIndex].ps.conv_spec.pw_activation_type = ACTIVATION_RELU;
                        break;
                    }
                    case Convolution_Depthwise: {
                        spec->ops[convOpIndex].ps.conv_spec.dw_activation_type = ACTIVATION_RELU;
                        break;
                    }
                    default: {
                        CHECK_REQUIREMENT(0);
                        break;
                    }
                }
                spec->ops[convOpIndex].ps.conv_spec.activation_spec.relu_spec = spec->ops[atOpIndex].ps.relu_spec;
                setOperatorInvalid(spec, atOpIndex);
            }
            if (spec->ops[i].type == OT_Clip || spec->ops[i].type == OT_Sigmoid) {
                // tensor_computing does not support fusion
                std::cout << "[Info] encounter unoptimized " << OperatorTypeName()[spec->ops[i].type] << " layer" <<std::endl;
                continue;
            }
        }
        return hasOptimized;
    }
};
#endif
