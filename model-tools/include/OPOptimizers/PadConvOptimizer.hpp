// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_PADCONVOPTIMIZER
#define _H_PADCONVOPTIMIZER

#include <vector>
#include <string>
#include "model_tools.h"
#include "OPOptimizer.hpp"

// Optimize both Convolution and Deconvolution
class PadConvOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 2;
        OperatorType queryOps[queryNum] = {OT_Conv, OT_Deconvolution};
        bool hasOptimized = false;
        for (int i = 0; i< spec->num_operator_specs; i++) {
            OperatorType curOT = spec->ops[i].type;
            if (curOT == OT_Pad && spec->ops[i].ps.pad_spec.pad_mode == Pad_Constant && spec->ops[i].ps.pad_spec.constant_value == 0) {
                int padOpIndex  = i;
                if (spec->ops[padOpIndex].ps.pad_spec.constant_value != 0) {
                    std::cout << "[WARNING] encounter unoptimized Pad layer (value not 0): " << spec->ops[i].name << std::endl;
                    continue;
                }                
                
                int convOpIndex = searchOperatorIndexForward(spec, padOpIndex + 1, queryOps, queryNum);
                
                if (convOpIndex == -1) {
                    std::cout << "[WARNING] encounter unoptimized Pad layer (no Conv or Deconv after): " << spec->ops[padOpIndex].name << std::endl;
                    continue;
                }

                // tensor relationship rewrite
                str_copy(spec->ops[convOpIndex].input_tensors_name[0], spec->ops[padOpIndex].input_tensors_name[0], NAME_LEN);
                hasOptimized = true;              
                spec->ops[convOpIndex].ps.conv_spec.padding_bottom += spec->ops[padOpIndex].ps.pad_spec.bottom;
                spec->ops[convOpIndex].ps.conv_spec.padding_left += spec->ops[padOpIndex].ps.pad_spec.left;
                spec->ops[convOpIndex].ps.conv_spec.padding_right += spec->ops[padOpIndex].ps.pad_spec.right;
                spec->ops[convOpIndex].ps.conv_spec.padding_top += spec->ops[padOpIndex].ps.pad_spec.top;                    
                setOperatorInvalid(spec, padOpIndex);     
            }          
        }
        return hasOptimized;
    } 
};
#endif
