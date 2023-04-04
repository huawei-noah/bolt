// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_Dynamic2ReshapeOPTIMIZER
#define _H_Dynamic2ReshapeOPTIMIZER

#include "OPOptimizer.hpp"

class Dynamic2ReshapeOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs - 12; i++) {
            if (spec->ops[i].type == OT_Shape && spec->ops[i+1].type == OT_Cast) {
		        if (spec->ops[i+2].type == OT_Gather && spec->ops[i+3].type == OT_Reduction) {
		            if (spec->ops[i+4].type == OT_Gather && spec->ops[i+5].type == OT_Concat) {
			            if (spec->ops[i+6].type == OT_Cast && spec->ops[i+7].type == OT_Reduction) {
			                if (spec->ops[i+8].type == OT_Concat && spec->ops[i+9].type == OT_Cast) {
			                    if (spec->ops[i+10].type == OT_Reshape && spec->ops[i+11].type == OT_FC) {
				                    if (spec->ops[i+12].type == OT_Reshape) {
			                            std::string concatSecInput = "weight_" + std::string(spec->ops[i+5].input_tensors_name[1]);
			                            int concatSecInputWeightIndex = searchWeightIndex(spec, concatSecInput);
                                        if (concatSecInputWeightIndex == -1) {
			                                continue;
			                            }
					                    U8* secPtr = spec->ws[concatSecInputWeightIndex].weight;
			                            I32* secPtrI32 = (I32*)secPtr;

					                    // fetch the FC
					                    std::string fcName = std::string(spec->ops[i+11].name);
					                    int fcWeightIndex = searchWeightIndex(spec, fcName);
                                        U32 matrixM = spec->ws[fcWeightIndex].bytes_of_weight / sizeof(float) / secPtrI32[0];

			                            // reconstruct the topology
			                            std::string blockInput = std::string(spec->ops[i].input_tensors_name[0]);
			                            str_copy(spec->ops[i+10].input_tensors_name[0], blockInput.c_str(), NAME_LEN);
			                            // modified the reshape operators
			                            spec->ops[i+10].num_inputs = 1;
			                            spec->ops[i+12].num_inputs = 1;
			                            // redefine the reshape parameters
					                    spec->ops[i+10].ps.reshape_spec.num_shape = 2;
				       	                spec->ops[i+10].ps.reshape_spec.shape[0] = -1;
					                    spec->ops[i+10].ps.reshape_spec.shape[1] = matrixM;

			                            spec->ops[i+12].ps.reshape_spec.num_shape = 3;
			                            spec->ops[i+12].ps.reshape_spec.shape[0] = 1;
			                            spec->ops[i+12].ps.reshape_spec.shape[1] = -1;
			                            spec->ops[i+12].ps.reshape_spec.shape[2] = (int)(secPtrI32[0]);

			                            // drop the redundant ops
			                            setOperatorInvalid(spec, i);
			                            setOperatorInvalid(spec, i+1);
			                            setOperatorInvalid(spec, i+2);
			                            setOperatorInvalid(spec, i+3);
			                            setOperatorInvalid(spec, i+4);
                                        setOperatorInvalid(spec, i+5);
                                        setOperatorInvalid(spec, i+6);
                                        setOperatorInvalid(spec, i+7);
                                        setOperatorInvalid(spec, i+8);
                                        setOperatorInvalid(spec, i+9);					

			                            hasOptimized = true;
			                        }
		                        }    
			                }
			            }
		            }		
	  	        }  
            }
        }
        return hasOptimized;
    }
};
#endif
