// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_OPOPTIMIZER
#define _H_OPOPTIMIZER

#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <map>
#include "model_tools.h"
#include "op_type.h"

class OPOptimizer {
    public:
        /**
         * @param spec
         */
        virtual ~OPOptimizer() {}
        virtual bool optimize(ModelSpec* spec) = 0;
        int searchWeightIndex(ModelSpec* spec, char* op_name) {
            if (spec->num_weight_specs <= 0) {
                return -1;
            }

            std::map<std::string, int> weightsIndex;
            for (int i=0; i < spec->num_weight_specs; i++) {
                std::string key = spec->ws[i].op_name;    // attention, this is static structure attribute
                weightsIndex[key] = i;
            }
            
            std::string opNameStr = op_name;
            std::map<std::string, int>::iterator iter = weightsIndex.find(opNameStr);
            if(iter == weightsIndex.end())
                return -1;
            else
                return weightsIndex[opNameStr];
        }

        bool isValidOperator(ModelSpec* spec, int index){
            if (index >= spec->num_operator_specs) {
                return false;
            }
                
            if (spec->ops[index].type != OT_None) {
                return true;
            } else {
                return false;
            }
        }

        void setOperatorInvalid(ModelSpec* spec, int index) {
            if (index >= spec->num_operator_specs || index < 0) {
                return;
            }
            spec->ops[index].type = OT_None;
        }

        int searchOperatorIndexBackward(ModelSpec* spec, int end, OperatorType *queryOps, int queryNum, bool unskip=true) {
            for (int i = end; i >= 0; i--) {
                if (isValidOperator(spec, i)) {
                    for (int j=0; j<queryNum; j++) {
                        OperatorType opType = queryOps[j];
                        if (spec->ops[i].type == opType) {
                            return i;
                        }
                    }
                    if (unskip) {
                        return -1;
                    }
                }
            }
            return -1;
        }

        int searchOperatorIndexForward(ModelSpec* spec, int start, OperatorType *queryOps, int queryNum, bool unskip=true){
            for (int i = start; i < spec->num_operator_specs; i++) {
                if (isValidOperator(spec, i)) {
                    for (int j=0; j<queryNum; j++) {
                        OperatorType opType = queryOps[j];
                        if(spec->ops[i].type == opType) {
                            return i;
                        }
                    }
                    if (unskip) {
                        return -1;
                    }
                }
            }
            return -1;
        }
};
#endif
