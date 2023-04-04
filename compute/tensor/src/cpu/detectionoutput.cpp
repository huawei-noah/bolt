// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/tensor_computing_cpu.h"
#include "cpu/non_max_suppression.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif

template <typename T>
EE detectionoutput_kernel(std::vector<void *> input,
    T *output,
    U32 priorbox_width,
    U32 num_class,
    F32 nms_threshold,
    U32 nms_top_k,
    U32 keep_top_k,
    F32 confidence_threshold,
    Arch arch)
{
    T *location = (T *)input[0];
    T *confidence = (T *)input[1];
    T *priorbox = (T *)input[2];

    U32 num_total_priorbox = priorbox_width / 4;

    T *variance = priorbox + priorbox_width;
    std::vector<T> xmin(num_total_priorbox);
    std::vector<T> ymin(num_total_priorbox);
    std::vector<T> xmax(num_total_priorbox);
    std::vector<T> ymax(num_total_priorbox);
    if (0) {
#ifdef _USE_GENERAL
    } else if (IS_GENERAL(arch)) {
        CHECK_STATUS(decode_priorbox_general<T>(location, priorbox, variance, num_total_priorbox,
            xmin.data(), ymin.data(), xmax.data(), ymax.data()));
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        CHECK_STATUS(decode_priorbox_x86<T>(location, priorbox, variance, num_total_priorbox,
            xmin.data(), ymin.data(), xmax.data(), ymax.data()));
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        CHECK_STATUS(decode_priorbox_arm<T>(location, priorbox, variance, num_total_priorbox,
            xmin.data(), ymin.data(), xmax.data(), ymax.data()));
#endif
    } else {
        return NOT_SUPPORTED;
    }

    std::vector<BoxRect> boxrects((num_class - 1) * nms_top_k);
    U32 count = 0;
    // class 0 is background
    for (U32 i = 1; i < num_class; i++) {
        std::vector<BoxRect> class_boxrects;
        for (U32 j = 0; j < num_total_priorbox; j++) {
            F32 score = confidence[j * num_class + i];
            if (score > confidence_threshold) {
                BoxRect b = {xmin[j], ymin[j], xmax[j], ymax[j], i, score, j};
                class_boxrects.push_back(b);
            }
        }

        // sort the boxes with scores
        std::stable_sort(class_boxrects.begin(), class_boxrects.end(),
            [&](const BoxRect &a, const BoxRect &b) { return (a.score > b.score); });
        if (nms_top_k < class_boxrects.size()) {
            class_boxrects.resize(nms_top_k);
        }

        // apply nms
        std::vector<I32> picked = nms_pickedboxes(class_boxrects, nms_threshold);
        for (U32 j = 0; j < picked.size(); j++) {
            I64 picked_box = picked[j];
            boxrects[count++] = class_boxrects[picked_box];
        }
    }
    boxrects.resize(count);

    std::stable_sort(boxrects.begin(), boxrects.end(),
        [&](const BoxRect &a, const BoxRect &b) { return (a.score > b.score); });
    if (keep_top_k < boxrects.size()) {
        boxrects.resize(keep_top_k);
    }

    U32 num_detected = boxrects.size();
    // the first box contains the number of availble boxes in the first element.
    output[0] = num_detected;
    output[1] = output[2] = output[3] = output[4] = output[5] = 0;
    for (U32 i = 0, j = 6; i < num_detected; i++) {
        BoxRect b = boxrects[i];
        output[j++] = b.label;
        output[j++] = b.score;
        output[j++] = b.xmin;
        output[j++] = b.ymin;
        output[j++] = b.xmax;
        output[j++] = b.ymax;
    }
    return SUCCESS;
}

EE detectionoutput_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    DetectionOutputParamSpec p,
    TensorDesc outputDesc,
    void *output,
    Arch arch)
{
    UNUSED(outputDesc);
    if (nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.size() != 3) {
        CHECK_STATUS(NOT_MATCH);
    }
    DataType idt0 = inputDesc[0].dt;
    U32 ilens2 = inputDesc[2].dims[0];
    EE ret = NOT_SUPPORTED;
    switch (idt0) {
#ifdef _USE_FP32
        case DT_F32:
            ret = detectionoutput_kernel(input, (F32 *)output, ilens2, p.num_class, p.nms_threshold,
                p.nms_top_k, p.keep_top_k, p.confidence_threshold, arch);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = detectionoutput_kernel(input, (F16 *)output, ilens2, p.num_class, p.nms_threshold,
                p.nms_top_k, p.keep_top_k, p.confidence_threshold, arch);
            break;
#endif
        default:
            break;
    }
    return ret;
}
