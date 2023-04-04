// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_NON_MAX_SUPPRESSION_TENSOR_COMPUTING
#define _H_NON_MAX_SUPPRESSION_TENSOR_COMPUTING

#include "parameter_spec.h"
#include "uni.h"
#include <vector>
#include <algorithm>

typedef struct {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    unsigned int label;
    float score;
    unsigned int index;
} BoxRect;

inline F32 intersectionarea(const BoxRect &a, const BoxRect &b)
{
    if (a.xmin >= b.xmax || a.xmax <= b.xmin || a.ymin >= b.ymax || a.ymax <= b.ymin) {
        return 0.f;
    }
    F32 inter_width = UNI_MIN(a.xmax, b.xmax) - UNI_MAX(a.xmin, b.xmin);
    F32 inter_height = UNI_MIN(a.ymax, b.ymax) - UNI_MAX(a.ymin, b.ymin);
    return inter_width * inter_height;
}

inline std::vector<I32> nms_pickedboxes(const std::vector<BoxRect> &boxes, F32 nms_threshold)
{
    I32 n = boxes.size();
    std::vector<F32> areas(n);
    for (I32 i = 0; i < n; i++) {
        const BoxRect &box = boxes[i];
        F32 width = box.xmax - box.xmin;
        F32 height = box.ymax - box.ymin;
        areas[i] = width * height;
    }
    std::vector<I32> picked;
    picked.reserve(n);
    for (I32 i = 0; i < n; i++) {
        const BoxRect &a = boxes[i];
        bool keep = true;
        for (U32 j = 0; j < picked.size(); j++) {
            const BoxRect &b = boxes[picked[j]];
            F32 inter_area = intersectionarea(a, b);
            F32 union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            picked.push_back(i);
        }
    }
    return picked;
}
#endif
