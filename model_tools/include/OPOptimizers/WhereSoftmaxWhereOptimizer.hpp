// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_WhereSoftmaxWhereOPTIMIZER
#define _H_WhereSoftmaxWhereOPTIMIZER

#include "OPOptimizer.hpp"
#include <iostream>

class WhereSoftmaxWhereOptimizer : public OPOptimizer {
    int searchSharedWeight(ModelSpec* spec, std::string weight_name)
    {
        for (int i = 0; i < spec->num_weight_specs; i++) {
	        if (std::string(spec->ws[i].op_name) == weight_name) {
	            return i;
	        }
	    }
	    return -1;
    }

    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Where && spec->ops[i + 1].type == OT_Softmax && spec->ops[i + 2].type == OT_Where && spec->ops[i+3].type == OT_MatMul) {
                int firstWhereXIndex = searchSharedWeight(spec, "weight_" + std::string(spec->ops[i].input_tensors_name[1]));
		        if (firstWhereXIndex == -1) {
		            std::cout << "can not find the x tensor of Where op, failed.\n";
		            exit(-1);  
		        }
		        U32 firstWhereXSize = spec->ws[firstWhereXIndex].bytes_of_weight;
		        float* firstWhereXPtr = (float*)(spec->ws[firstWhereXIndex].weight);
		        int secondWhereXIndex = searchSharedWeight(spec, "weight_" + std::string(spec->ops[i + 2].input_tensors_name[1]));
                if (secondWhereXIndex == -1) {
                    std::cout << "can not find the x tensor of Where op, failed.\n";
                    exit(-1);
                }
		        U32 secondWhereXSize = spec->ws[secondWhereXIndex].bytes_of_weight;
		        float* secondWhereXPtr = (float*)(spec->ws[secondWhereXIndex].weight);
		        SoftmaxParamSpec smps = spec->ops[i + 1].ps.softmax_spec;
		
                if (4 == firstWhereXSize && 4 == secondWhereXSize) {
		            if ( (-1.0/0.0) == firstWhereXPtr[0] && 0 == secondWhereXPtr[0]) {
                        if (std::string(spec->ops[i].input_tensors_name[0]) == std::string(spec->ops[i+2].input_tensors_name[0])) {
		                    if (smps.axis == -1) {
			                    UNI_MEMCPY(spec->ops[i+3].input_tensors_name[0], spec->ops[i+1].output_tensors_name[0], NAME_LEN);
			                    setOperatorInvalid(spec, i + 2);
			                    hasOptimized = true;
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
