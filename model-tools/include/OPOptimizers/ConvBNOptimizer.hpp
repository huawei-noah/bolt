// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CONVBNOPTIMIZER
#define _H_CONVBNOPTIMIZER

#include <vector>
#include <string>
#include <math.h>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class ConvBNOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 3;
        OperatorType queryOps[queryNum] = {OT_Conv, OT_FC, OT_Deconvolution};

        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_BatchNorm) {
                int bnOpIndex = i;
                int convOpIndex = searchOperatorIndexBackward(spec, i-1, queryOps, queryNum);
                if (convOpIndex == -1) {
                    std::cout << "[Info] encounter unoptimize BatchNorm layer(no Conv before): " << spec->ops[i].name << std::endl;
                    continue;
                }

                // tensor relationship rewrite
                str_copy(spec->ops[convOpIndex].output_tensors_name[0], spec->ops[bnOpIndex].output_tensors_name[0], NAME_LEN);
                hasOptimized = true;

                // bn
                int bnWeightIndex = searchWeightIndex(spec, spec->ops[bnOpIndex].name);
                CHECK_REQUIREMENT(bnWeightIndex >= 0);
                CHECK_REQUIREMENT(spec->ws[bnWeightIndex].mdt == DT_F32);
                F32 epsCur = spec->ops[bnOpIndex].ps.bn_spec.eps;
                U32 channelCur = spec->ws[bnWeightIndex].bytes_of_weight / bytesOf(spec->ws[bnWeightIndex].mdt);
                F32* meanPtr = (F32 *)spec->ws[bnWeightIndex].weight;
                F32* varPtr = (F32 *)spec->ws[bnWeightIndex].vec;

                std::vector<float> stdValue(channelCur);
                for (U32 j=0; j < channelCur; j++) {
                    stdValue[j] = sqrt(varPtr[j] + epsCur);
                }

                // conv
                int convWeightIndex = searchWeightIndex(spec, spec->ops[convOpIndex].name);
                CHECK_REQUIREMENT(convWeightIndex >= 0);
                // Now weight mdt can be DT_BIN01 or DT_BIN11
                U32 isBNN = 0;
                if (spec->ws[convWeightIndex].mdt == DT_BIN01 || spec->ws[convWeightIndex].mdt == DT_BIN11) {
                    isBNN = 1;
                }
                F32* weightTemp = (F32*)spec->ws[convWeightIndex].weight;
                if(spec->ws[convWeightIndex].vec == nullptr){
                    spec->ws[convWeightIndex].bytes_of_vec = channelCur * sizeof(F32);
                    if (isBNN == 1) {
                        spec->ws[convWeightIndex].bytes_of_vec *= 2;
                    }
                    spec->ws[convWeightIndex].vec = (U8 *)mt_new_storage(spec->ws[convWeightIndex].bytes_of_vec);
                    memset(spec->ws[convWeightIndex].vec, 0, spec->ws[convWeightIndex].bytes_of_vec);
                }
                F32* vecTemp    = (F32*)spec->ws[convWeightIndex].vec;
                if (isBNN == 1) { // Do not modify weights for BNN
                    F32* scale = vecTemp;
                    F32* bias = vecTemp + channelCur;
                    for (U32 m = 0; m < channelCur; m++) {
                        scale[m] = 1.0 / stdValue[m]; // This is the first possible source of a meaningful scale, so just initilize
                        bias[m] = (bias[m] - meanPtr[m]) / stdValue[m];
                    }
                } else {
                    int weightDataSize = spec->ws[convWeightIndex].bytes_of_weight / bytesOf(spec->ws[convWeightIndex].mdt);
                    int weightPerChannel = weightDataSize / channelCur;
                    // NCHW
                    for (U32 m = 0; m < channelCur; m++) {
                        F32* convWeightPerChannel =  weightTemp + weightPerChannel * m;
                        for (int n = 0; n < weightPerChannel; n++) {
                            convWeightPerChannel[n] /= stdValue[m];
                        }
                        vecTemp[m] = (vecTemp[m] - meanPtr[m]) / stdValue[m];
                    }
                }
                // free BN memory
                if(spec->ws[bnWeightIndex].weight != nullptr) {
                    spec->ws[bnWeightIndex].bytes_of_weight = 0;
                    delete [] spec->ws[bnWeightIndex].weight;
                    spec->ws[bnWeightIndex].weight = nullptr;
                }
                if(spec->ws[bnWeightIndex].vec != nullptr) {
                    spec->ws[bnWeightIndex].bytes_of_vec = 0;
                    delete [] spec->ws[bnWeightIndex].vec;
                    spec->ws[bnWeightIndex].vec = nullptr;
                }
                setOperatorInvalid(spec, bnOpIndex);
            }
        }
        return hasOptimized;
    }
};
#endif
