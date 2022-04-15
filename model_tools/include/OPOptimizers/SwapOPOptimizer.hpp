// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_SwapOPOPTIMIZER
#define _H_SwapOPOPTIMIZER

#include "OPOptimizer.hpp"

class SwapOPOptimizer : public OPOptimizer {
protected:
    // original [0, 1, 2] -> [1, 2, 0]
    void shift_left(ModelSpec *spec, std::vector<int> order) {
        int num = order.size();
        if (num <= 1) {
            return;
        }
        std::string name0 = spec->ops[order[0]].name;
        auto type0 = spec->ops[order[0]].type;
        auto ps0 = spec->ops[order[0]].ps;
        for (int i = 1; i < num; i++) {
            UNI_STRCPY(spec->ops[order[i-1]].name, spec->ops[order[i]].name);
            spec->ops[order[i-1]].type = spec->ops[order[i]].type;
            spec->ops[order[i-1]].ps = spec->ops[order[i]].ps;
        }
        UNI_STRCPY(spec->ops[order[num-1]].name, name0.c_str());
        spec->ops[order[num-1]].type = type0;
        spec->ops[order[num-1]].ps = ps0;
    }
};
#endif
