// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_Dynamic1ReshapeOPTIMIZER
#define _H_Dynamic1ReshapeOPTIMIZER

#include "OPOptimizer.hpp"

class Dynamic1ReshapeOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs - 7; i++) {
            if (spec->ops[i].type == OT_Shape && spec->ops[i+1].type == OT_Cast) {
		        if (spec->ops[i+2].type == OT_Slice && spec->ops[i+3].type == OT_Concat) {
		            if (spec->ops[i+4].type == OT_Cast && spec->ops[i+5].type == OT_Reshape) {
			            if (spec->ops[i+6].type == OT_FC && spec->ops[i+7].type == OT_Reshape) {
			                std::string concatFirstInput = "weight_" + std::string(spec->ops[i+3].input_tensors_name[0]);
			                std::string concatThirdInput = "weight_" + std::string(spec->ops[i+3].input_tensors_name[2]);
			                int concatFirstInputWeightIndex = searchWeightIndex(spec, concatFirstInput);
			                int concatThirdInputWeightIndex = searchWeightIndex(spec, concatThirdInput);
                            if (concatFirstInputWeightIndex == -1 || concatThirdInputWeightIndex == -1) {
			                    continue;
			                }
			                U8* firstPtr = spec->ws[concatFirstInputWeightIndex].weight;
			                I32* firstPtrI32 = (I32*)firstPtr;
			                U8* thridPtr = spec->ws[concatThirdInputWeightIndex].weight; 
			                I32* thridPtrI32 = (I32*)thridPtr;

			                // reconstruct the topology
			                std::string blockInput = std::string(spec->ops[i].input_tensors_name[0]);
			                str_copy(spec->ops[i+5].input_tensors_name[0], blockInput.c_str(), NAME_LEN);
			                // modified the reshape operators
			                spec->ops[i+5].num_inputs = 1;
			                spec->ops[i+7].num_inputs = 1;
			                // redefine the reshape parameters
			                spec->ops[i+7].ps.reshape_spec.num_shape = 3;
			                spec->ops[i+7].ps.reshape_spec.shape[0] = (int)(firstPtrI32[0]);
			                spec->ops[i+7].ps.reshape_spec.shape[1] = -1;
			                spec->ops[i+7].ps.reshape_spec.shape[2] = (int)(thridPtrI32[0]);

			                // drop the redundant ops
			                setOperatorInvalid(spec, i);
			                setOperatorInvalid(spec, i+1);
			                setOperatorInvalid(spec, i+2);
			                setOperatorInvalid(spec, i+3);
			                setOperatorInvalid(spec, i+4);

			                hasOptimized = true;
			            }
		            }    
		        }
            }
        }
        return hasOptimized;
    }
};
#endif
