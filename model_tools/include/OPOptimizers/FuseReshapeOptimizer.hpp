// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_FuseReshapeOPTIMIZER
#define _H_FuseReshapeOPTIMIZER
#include "OPOptimizer.hpp"

/*
       reshape                       |
          |                          | 
       ___|_____                  ___|_____
      |         |                |         |
   reshape   reshape   --->      |         | 
      |         |                |         | 
   reshape   reshape           reshape   reshape

*/

class FuseReshapeOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].type == OT_Reshape) {  // bottom-up
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                bool tag = true;
                if (nextOpIndexes.size() == 0) {
                    tag = false;
                }
                for (unsigned int k = 0; k < nextOpIndexes.size(); k++) {
                    if (spec->ops[nextOpIndexes[k].first].type != OT_Reshape) {
                        tag = false;
                        break;
                    }
                }
                if (tag) {
                    setOperatorInvalid(spec, i, true);
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
