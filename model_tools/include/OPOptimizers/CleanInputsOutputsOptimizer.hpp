// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CLEANINPUTSOUTPUTSOPTIMIZER
#define _H_CLEANINPUTSOUTPUTSOPTIMIZER

#include "OPOptimizer.hpp"
#include <map>

class CleanInputsOutputsOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
   	    bool hasOptimized = false;
        std::map<std::string, int> inputsMentionedCount;
	    std::map<std::string, int> outputsMentionedCount;
	    for (int i = 0; i < (int)(spec->num_inputs); i++) {
	        inputsMentionedCount[std::string(spec->input_names[i])] = 0;
	    }
	    for (int i = 0; i < (int)(spec->num_outputs); i++) {
	        outputsMentionedCount[std::string(spec->output_names[i])] = 0;
	    }
        for (int i = 0; i < (int)(spec->num_operator_specs); i++) {
            OperatorSpec opSpec = spec->ops[i];
            for (int j = 0; j < (int)(opSpec.num_inputs); j++) {
	            std::string tmpInput = std::string(opSpec.input_tensors_name[j]);
		        if (inputsMentionedCount.find(tmpInput) != inputsMentionedCount.end()) {
		            inputsMentionedCount[tmpInput] += 1;
		        }
	        }
            for (int j = 0; j < (int)(opSpec.num_outputs); j++) {
                std::string tmpOutput = std::string(opSpec.output_tensors_name[j]);
                if (outputsMentionedCount.find(tmpOutput) != outputsMentionedCount.end()) {
                    outputsMentionedCount[tmpOutput] += 1;
                }
            }
	    }

	    std::vector<int> removeInputIndex;
	    std::vector<int> removeOutputIndex;
        for (int i = 0; i < (int)(spec->num_inputs); i++) {
	        if (inputsMentionedCount[std::string(spec->input_names[i])] == 0) {
	            removeInputIndex.push_back(i); 
	        }
        }
        for (int i = 0; i < (int)(spec->num_outputs); i++) {
            if (outputsMentionedCount[std::string(spec->output_names[i])] == 0) {
                removeOutputIndex.push_back(i);
            }
        } 

	    int swapInputIndex = spec->num_inputs - 1;
	    for (int i = 0; i < (int)(removeInputIndex.size()); i++) {
	        int preIndex = removeInputIndex[i];
	        if (preIndex != swapInputIndex) {
	            I8* tmpPtr = spec->input_names[preIndex];
		        spec->input_names[preIndex] = spec->input_names[swapInputIndex];
		        spec->input_names[swapInputIndex] = tmpPtr;

		        TensorDesc tmpTD = spec->input_dims[preIndex];
		        spec->input_dims[preIndex] = spec->input_dims[swapInputIndex];
		        spec->input_dims[swapInputIndex] = tmpTD;
	        } else {
	            break;
	        }
	        swapInputIndex--;
	    }
	    int originalInputNum = spec->num_inputs;
	    for (int i = 0; i < (int)(removeInputIndex.size()); i++) {
	        mt_free(spec->input_names[originalInputNum - 1 - i]);
	        spec->num_inputs--;
	        hasOptimized = true;
	    }

        int swapOutputIndex = spec->num_outputs - 1;
        for (int i = 0; i < (int)(removeOutputIndex.size()); i++) {
            int preIndex = removeOutputIndex[i];
            if (preIndex != swapOutputIndex) {
                I8* tmpPtr = spec->output_names[preIndex];
                spec->output_names[preIndex] = spec->output_names[swapOutputIndex];
                spec->output_names[swapOutputIndex] = tmpPtr;
            } else {
                break;
            }
            swapOutputIndex--;
        }
        int originalOutputNum = spec->num_outputs;
        for (int i = 0; i < (int)(removeOutputIndex.size()); i++) {
            mt_free(spec->output_names[originalOutputNum - 1 - i]);
            spec->num_outputs--;
	        hasOptimized = true;
        }
        return hasOptimized;
    }
};
#endif
