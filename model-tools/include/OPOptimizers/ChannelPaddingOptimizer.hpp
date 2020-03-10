// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CHANNELPADDINGOPTIMIZER
#define _H_CHANNELPADDINGOPTIMIZER

#include <vector>
#include <string>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class ChannelPaddingOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        bool hasOptimized = false;  // If padding optimization has never been done, we do not need to check the number of input channels
        for (int i = 0; i< spec->num_operator_specs; i++) {
            bool padding = false;  // Whether to check input channels and actually pad
            bool optimizeOC = false;
            U32 numKernels = 0;
            U32 kernelSizeH = 0;
            U32 kernelSizeW = 0;
            if (spec->ops[i].type == OT_Conv || spec->ops[i].type == OT_Deconvolution) {
                if (spec->ops[i].ps.conv_spec.convolution_type != Convolution_Pointwise && spec->ops[i].ps.conv_spec.convolution_type != Convolution_Deconvolution) {
                    // Depthwise not supported for the time being
                    continue;
                }

                numKernels = spec->ops[i].ps.conv_spec.num_kernels;
                kernelSizeH = spec->ops[i].ps.conv_spec.kernel_size_h;
                kernelSizeW = spec->ops[i].ps.conv_spec.kernel_size_w;
                if (numKernels % 8 != 0) {  // Check output channels
                    optimizeOC = true;
                }
                padding = hasOptimized || optimizeOC;  // If padding has been done before, we need to check the input channels as well
            } else if (spec->ops[i].type == OT_FC) {
                numKernels = spec->ops[i].ps.fc_spec.num_outputs;
                kernelSizeH = 1;
                kernelSizeW = 1;
                padding = hasOptimized;
            } else {
                continue;
            }

            if (padding) {
                int weightIndex = searchWeightIndex(spec, spec->ops[i].name);
                CHECK_REQUIREMENT(weightIndex >= 0);
                CHECK_REQUIREMENT(spec->ws[weightIndex].mdt == DT_F32);  // BNN not supported for the time being
                U32 weightSize = spec->ws[weightIndex].bytes_of_weight / bytesOf(spec->ws[weightIndex].mdt);
                U32 inputChannels = weightSize / (numKernels * kernelSizeH * kernelSizeW);
                if (inputChannels % 8 == 0 && false == optimizeOC) {
                    continue;
                }

                U32 numKernelsNew = optimizeOC ? ((numKernels / 8 + 1) * 8) : numKernels;
                U32 inputChannelsNew = (inputChannels % 8) ? ((inputChannels / 8 + 1) * 8) : inputChannels;

                U8 *weight = spec->ws[weightIndex].weight;
                U8 *vec = spec->ws[weightIndex].vec;
                U32 vecBytes = spec->ws[weightIndex].bytes_of_vec;
                spec->ws[weightIndex].bytes_of_weight = bytesOf(spec->ws[weightIndex].mdt)
                                                           * numKernelsNew * inputChannelsNew * kernelSizeH * kernelSizeW;
                spec->ws[weightIndex].bytes_of_vec = bytesOf(spec->ws[weightIndex].mdt) * numKernelsNew;
                spec->ws[weightIndex].weight = (U8 *)mt_new_storage(spec->ws[weightIndex].bytes_of_weight);
                spec->ws[weightIndex].vec = (U8 *)mt_new_storage(spec->ws[weightIndex].bytes_of_vec);
                memset(spec->ws[weightIndex].weight, 0, spec->ws[weightIndex].bytes_of_weight);
                memset(spec->ws[weightIndex].vec, 0, spec->ws[weightIndex].bytes_of_vec);
                if (spec->ops[i].type == OT_Conv)
                    spec->ops[i].ps.conv_spec.num_kernels = numKernelsNew;
                if (spec->ops[i].type == OT_FC)
                    spec->ops[i].ps.fc_spec.num_outputs = numKernelsNew;
                // process weight
                U32 blockSize = bytesOf(spec->ws[weightIndex].mdt) * kernelSizeH * kernelSizeW;
                for (U32 oc = 0; oc < numKernels; oc++) {
                    for (U32 ic = 0; ic < inputChannels; ic++) {
                        U32 oldIndex = (oc * inputChannels + ic) * blockSize;
                        U32 newIndex = (oc * inputChannelsNew + ic) * blockSize;
                        memcpy(spec->ws[weightIndex].weight + newIndex, weight + oldIndex, blockSize);
                    }
                }
                delete [] weight;
                // process bias
                if(vec != nullptr) {
                    memcpy(spec->ws[weightIndex].vec, vec, vecBytes);
                    delete [] vec;
                }

                hasOptimized = true;
            }
        }
        return hasOptimized;
    }
};
#endif
