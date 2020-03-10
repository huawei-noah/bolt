// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CONVSCALEOPTIMIZER
#define _H_CONVSCALEOPTIMIZER

#include <vector>
#include <string>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class ConvScaleOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 3;
        OperatorType queryOps[queryNum] = {OT_Conv, OT_FC, OT_Deconvolution};
 
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Scale) {
                int scaleOpIndex = i;
                if (spec->ops[scaleOpIndex].num_inputs > 1){
                    std::cout << "[WARNING] encounter unoptimize Scale layer(multi-inputs): " << spec->ops[i].name <<std::endl;
                    continue;
                }
                int convOpIndex = searchOperatorIndexBackward(spec, i-1, queryOps, queryNum);
                if (convOpIndex == -1) {
                    std::cout << "[WARNING] encounter unoptimize Scale layer(no Conv before): " << spec->ops[i].name <<std::endl;
                    continue;
                }

                // tensor relationship rewrite
                str_copy(spec->ops[convOpIndex].output_tensors_name[0], spec->ops[scaleOpIndex].output_tensors_name[0], NAME_LEN);
                hasOptimized = true;

                // scale
                int scaleWeightIndex = searchWeightIndex(spec, spec->ops[scaleOpIndex].name);
                CHECK_REQUIREMENT(scaleWeightIndex >= 0);
                CHECK_REQUIREMENT(spec->ws[scaleWeightIndex].mdt == DT_F32);
                U32 channelAlpha = spec->ws[scaleWeightIndex].bytes_of_weight / bytesOf(spec->ws[scaleWeightIndex].mdt);
                U32 channelBeta  = spec->ws[scaleWeightIndex].bytes_of_vec / bytesOf(spec->ws[scaleWeightIndex].mdt);
                U32 channelCur = UNI_MAX(channelAlpha, channelBeta);
                F32* alphaPtr = (F32*)spec->ws[scaleWeightIndex].weight;
                F32* betaPtr = (F32*)spec->ws[scaleWeightIndex].vec;

                // conv
                int convWeightIndex = searchWeightIndex(spec, spec->ops[convOpIndex].name);
                CHECK_REQUIREMENT(convWeightIndex >= 0);
                // mdt can now be DT_BIN01 or DT_BIN11
                U32 isBNN = 0;
                if (spec->ws[convWeightIndex].mdt==DT_BIN01 || spec->ws[convWeightIndex].mdt==DT_BIN11) {
                    isBNN = 1;
                }
                F32* weightTemp = (F32*)spec->ws[convWeightIndex].weight;
                if (spec->ws[convWeightIndex].vec == nullptr) {
                    spec->ws[convWeightIndex].bytes_of_vec = channelCur * sizeof(F32);
                    if (isBNN == 1) {
                        spec->ws[convWeightIndex].bytes_of_vec *= 2;
                    }
                    spec->ws[convWeightIndex].vec = (U8 *)mt_new_storage(spec->ws[convWeightIndex].bytes_of_vec);
                    memset(spec->ws[convWeightIndex].vec, 0, spec->ws[convWeightIndex].bytes_of_vec);
                }
                F32* vecTemp = (F32*)spec->ws[convWeightIndex].vec;
                if (isBNN == 1) {
                    F32 *scale = vecTemp;
                    F32 *bias = vecTemp + channelCur;
                    for (U32 m = 0; m < channelCur; m++) {
                        if (scale[m] == 0) {
                            scale[m] = alphaPtr[m];
                        } else {
                            scale[m] *= alphaPtr[m];
                        }
                        bias[m] *= alphaPtr[m];
                        if (betaPtr != nullptr) {
                            bias[m] += betaPtr[m];
                        }
                    }
                } else {
                    int weightDataSize = spec->ws[convWeightIndex].bytes_of_weight / bytesOf(spec->ws[convWeightIndex].mdt);
                    int weightPerChannel = weightDataSize / channelCur;
                    // NCHW
                    for (U32 m = 0; m < channelCur; m++){
                        F32* convWeightPerChannel =  weightTemp + weightPerChannel * m;
                        if (alphaPtr != nullptr) {
                            for(int n = 0; n < weightPerChannel; n++){
                                convWeightPerChannel[n] *= alphaPtr[m];
                            }
                            vecTemp[m] = alphaPtr[m] * vecTemp[m];
                        }
                        if (betaPtr != nullptr) {
                            vecTemp[m] += betaPtr[m];
                        }
                    }
                }
                // free scale memory
                if (spec->ws[scaleWeightIndex].weight != nullptr) {
                    spec->ws[scaleWeightIndex].bytes_of_weight = 0;
                    delete [] spec->ws[scaleWeightIndex].weight;
                    spec->ws[scaleWeightIndex].weight = nullptr;
                }
                if(spec->ws[scaleWeightIndex].vec != nullptr) {
                    spec->ws[scaleWeightIndex].bytes_of_vec = 0;
                    delete [] spec->ws[scaleWeightIndex].vec;
                    spec->ws[scaleWeightIndex].vec = nullptr;
                }
                setOperatorInvalid(spec, scaleOpIndex);
            }
        }
        return hasOptimized;
    }
};
#endif
