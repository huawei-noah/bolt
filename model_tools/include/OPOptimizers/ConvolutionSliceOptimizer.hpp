// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CONVOLUTIONSLICEOPTIMIZER
#define _H_CONVOLUTIONSLICEOPTIMIZER

#include "OPOptimizer.hpp"

/*  fuse conv + slice -> conv(cropped)

          |               
         conv           
          |                        |
        slice         ---->       conv
          |                        |        

*/

inline void set_padding(char *mask, int *crop, int *p, int s)
{
    if (*mask == 0 && *crop >= 0) {
        if (*crop <= *p / s) {
            *p -= *crop * s;
            *crop = 0;
        } else {
            *crop = *crop - *p / s;
            *p = 0;
        }
        if (*crop == 0) {
            *mask = 1;
        }
    }
}

class ConvolutionSliceOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Conv) {
                std::string curOut = spec->ops[i].output_tensors_name[0];
                auto nextOpIndexes = searchOperatorIndexByInput(
                    spec, curOut, i + 1, spec->num_operator_specs, false);

                if ((nextOpIndexes.size() != 1) || (-1 == nextOpIndexes[0].first) ||
                    (OT_TfSlice != spec->ops[nextOpIndexes[0].first].type)) {
                    continue;
                }

                int sliceIndex = nextOpIndexes[0].first;
                U32 dims = 4;
                int p = 0;
                int cropLength = 0;
                if (spec->ops[sliceIndex].ps.tfslice_spec.begin[dims - 1] > 0) {
                    p = spec->ops[i].ps.conv_spec.padding_left;
                    set_padding(&spec->ops[sliceIndex].ps.tfslice_spec.begin_mask[dims - 1],
                        &spec->ops[sliceIndex].ps.tfslice_spec.begin[dims - 1], &p,
                        spec->ops[i].ps.conv_spec.stride_w);
                    spec->ops[i].ps.conv_spec.padding_left = p;
                }
                if (spec->ops[sliceIndex].ps.tfslice_spec.begin[dims - 2] > 0) {
                    p = spec->ops[i].ps.conv_spec.padding_top;
                    set_padding(&spec->ops[sliceIndex].ps.tfslice_spec.begin_mask[dims - 2],
                        &spec->ops[sliceIndex].ps.tfslice_spec.begin[dims - 2], &p,
                        spec->ops[i].ps.conv_spec.stride_h);
                    spec->ops[i].ps.conv_spec.padding_top = p;
                }
                if (spec->ops[sliceIndex].ps.tfslice_spec.end[dims - 1] < 0) {
                    p = spec->ops[i].ps.conv_spec.padding_right;
                    cropLength = -spec->ops[sliceIndex].ps.tfslice_spec.end[dims - 1];
                    set_padding(&spec->ops[sliceIndex].ps.tfslice_spec.end_mask[dims - 1],
                        &cropLength, &p, spec->ops[i].ps.conv_spec.stride_w);
                    spec->ops[sliceIndex].ps.tfslice_spec.end[dims - 1] = -cropLength;
                    spec->ops[i].ps.conv_spec.padding_right = p;
                }
                if (spec->ops[sliceIndex].ps.tfslice_spec.end[dims - 2] < 0) {
                    p = spec->ops[i].ps.conv_spec.padding_bottom;
                    cropLength = -spec->ops[sliceIndex].ps.tfslice_spec.end[dims - 2];
                    set_padding(&spec->ops[sliceIndex].ps.tfslice_spec.end_mask[dims - 2],
                        &cropLength, &p, spec->ops[i].ps.conv_spec.stride_h);
                    spec->ops[i].ps.conv_spec.padding_bottom = p;
                    spec->ops[sliceIndex].ps.tfslice_spec.end[dims - 2] = -cropLength;
                }

                hasOptimized = true;
                for (U32 idx = 0; idx < dims; ++idx) {
                    if (spec->ops[sliceIndex].ps.tfslice_spec.begin[idx] == 0) {
                        spec->ops[sliceIndex].ps.tfslice_spec.begin_mask[idx] = 1;
                    }
                    if (spec->ops[sliceIndex].ps.tfslice_spec.begin_mask[idx] != 1 ||
                        spec->ops[sliceIndex].ps.tfslice_spec.end_mask[idx] != 1) {
                        hasOptimized = false;
                    }
                }

                if (hasOptimized) {
                    setOperatorInvalid(spec, sliceIndex, true);
                }
            }
        }
        return hasOptimized;
    }
};
#endif
