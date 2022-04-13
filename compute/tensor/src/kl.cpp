// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tensor_computing.h"
#include "cpu/cpu_functions.h"

std::vector<F32> compress_histogram(std::vector<F32> &histogram, F32 numPerBin, F32 last_max)
{
    std::vector<F32> newhistogram(2048, 0);
    for (U32 q = 0; q < ceil(2048 / numPerBin); q++) {
        F32 start = q * numPerBin;
        F32 end = start + numPerBin;
        int left = ceil(start);
        if (left > start) {
            newhistogram[q] += ((F32)left - start) * histogram[left - 1];
        }
        if (end <= last_max) {
            int right = floor(end);
            if (right < end) {
                newhistogram[q] += (end - (F32)right) * histogram[right];
            }

            for (int k = left; k < right; k++) {
                newhistogram[q] += histogram[k];
            }
        } else {
            for (int k = left; k < 2048; k++) {
                newhistogram[q] += histogram[k];
            }
        }
    }
    histogram.assign(newhistogram.begin(), newhistogram.end());
    return histogram;
}

F32 compute_KLD(U32 len, const F32 *p, const F32 *q)
{
    F32 kld = 0;

    for (U32 i = 0; i < len; i++) {
        if (0 != p[i]) {
            if (0 == q[i]) {
                kld += 1;
            } else {
                kld += p[i] * log(p[i] / q[i]);
            }
        }
    }

    return kld;
}

std::vector<F32> compute_scale_with_KL(std::vector<F32> &histogram, F32 interval)
{
    ArraySumFunction sum_func = get_array_sum_function(CPU_GENERAL);
    ArrayScaleFunction scale_func = get_array_scale_function(CPU_GENERAL);
    std::vector<F32> scale;
    const int BINS = 2048;
    F32 histoSum = sum_func(DT_F32, histogram.data(), BINS);

    F32 minKLD = 2048;
    int bestThreshold = 128;
    F32 sumBin = sum_func(DT_F32, histogram.data(), 128);
    UNI_DEBUG_LOG("First 128 bins contain %f of values", sumBin);
    F32 sumOver = histoSum - sumBin;

    for (U32 i = 128; i < 2048; i++) {
        std::vector<F32> clipDist(histogram.begin(), histogram.begin() + i);
        clipDist[i - 1] += sumOver;
        sumOver -= histogram[i];  // Prepare for next round
        scale_func(DT_F32, clipDist.data(), clipDist.data(), i, 1 / histoSum, 0);

        std::vector<F32> quantDist(128, 0);

        F32 numPerBin = (F32)i / 128.0;

        for (U32 j = 0; j < 128; j++) {
            F32 start = j * numPerBin;
            F32 end = start + numPerBin;

            int left = ceil(start);
            if (left > start) {
                quantDist[j] += ((F32)left - start) * histogram[left - 1];
            }

            int right = floor(end);
            if (right < end) {
                quantDist[j] += (end - (F32)right) * histogram[right];
            }

            for (int k = left; k < right; k++) {
                quantDist[j] += histogram[k];
            }
        }

        std::vector<F32> qExpand(i, 0);

        for (U32 j = 0; j < 128; j++) {
            F32 start = j * numPerBin;
            F32 end = start + numPerBin;

            F32 count = 0;

            int left = ceil(start);
            if (left > start && 0 != histogram[left - 1]) {
                count += (F32)left - start;
            }

            int right = floor(end);
            if (right < end && 0 != histogram[right]) {
                count += end - (F32)right;
            }

            for (int k = left; k < right; k++) {
                if (0 != histogram[k]) {
                    count += 1;
                }
            }

            F32 expandVal = quantDist[j] / count;

            if (left > start && 0 != histogram[left - 1]) {
                qExpand[left - 1] += expandVal * ((F32)left - start);
            }

            if (right < end && 0 != histogram[right]) {
                qExpand[right] += expandVal * (end - (F32)right);
            }

            for (int k = left; k < right; k++) {
                if (0 != histogram[k]) {
                    qExpand[k] += expandVal;
                }
            }
        }
        F32 kld = compute_KLD(i, clipDist.data(), qExpand.data());

        if (kld < minKLD) {
            minKLD = kld;
            bestThreshold = i;
        }
    }
    UNI_DEBUG_LOG(" %d/2048\n", bestThreshold);
    F32 threshold = (F32)bestThreshold * interval;
    F32 quantScale = 127.0 / threshold;
    scale.push_back(quantScale);
    return scale;
}
