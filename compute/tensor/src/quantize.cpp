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
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP16
#include "cpu/arm/fp16/arm_functions_fp16.h"
#endif
#ifdef _USE_FP32
#include "cpu/arm/fp32/arm_functions_fp32.h"
#endif
#endif

EE quantize_tensor(TensorDesc dDesc, const void *data, TensorDesc *qDesc, void *qData, void *scale)
{
    EE ret = NOT_SUPPORTED;
#ifdef _USE_NEON
    ret = quantize_tensor_arm(dDesc, data, qDesc, qData, scale);
#endif
    return ret;
}

#if defined(_USE_NEON) && defined(_USE_INT8)
void dequantize_int8_to_fp16(U32 len, INT8 *q, F32 scale, F16 *d)
{
    F16 factor = 1 / scale;
    int i = 0;
    for (; i < ((int)len) - 15; i += 16) {
        int8x8_t in0 = vld1_s8(q + i);
        int8x8_t in1 = vld1_s8(q + i + 8);
        int16x8_t s0 = vmovl_s8(in0);
        int16x8_t s1 = vmovl_s8(in1);
        float16x8_t f0 = vcvtq_f16_s16(s0);
        float16x8_t f1 = vcvtq_f16_s16(s1);
        f0 = vmulq_n_f16(f0, factor);
        f1 = vmulq_n_f16(f1, factor);
        vst1q_f16(d + i, f0);
        vst1q_f16(d + i + 8, f1);
    }

    for (; i < (int)len; i++) {
        d[i] = q[i] * factor;
    }
}

void dequantize_int32_to_fp16(U32 len, I32 *q, F32 scale, F16 *d, U32 biasLen, F16 *biasPtr)
{
    if (0 != biasLen) {
        CHECK_REQUIREMENT(nullptr != biasPtr);
        CHECK_REQUIREMENT(len % biasLen == 0);
    }
    float16x4_t bias[4];

    F32 factor = 1 / scale;
    if (biasLen % 4 == 0) {
        int i = 0;
        for (; i < ((int)len) - 15; i += 16) {
            int32x4_t in0 = vld1q_s32(q + i);
            int32x4_t in1 = vld1q_s32(q + i + 4);
            int32x4_t in2 = vld1q_s32(q + i + 8);
            int32x4_t in3 = vld1q_s32(q + i + 12);
            if (0 != biasLen) {
                U32 offset = i % biasLen;
                for (U32 j = 0; j < 4; j++) {
                    bias[j] = vld1_f16(biasPtr + offset);
                    offset += 4;
                    if (offset >= biasLen) {
                        offset = 0;
                    }
                }
            }
            float32x4_t f0 = vcvtq_f32_s32(in0);
            float32x4_t f1 = vcvtq_f32_s32(in1);
            float32x4_t f2 = vcvtq_f32_s32(in2);
            float32x4_t f3 = vcvtq_f32_s32(in3);
            f0 = vmulq_n_f32(f0, factor);
            f1 = vmulq_n_f32(f1, factor);
            f2 = vmulq_n_f32(f2, factor);
            f3 = vmulq_n_f32(f3, factor);
            float16x4_t h0 = vcvt_f16_f32(f0);
            float16x4_t h1 = vcvt_f16_f32(f1);
            float16x4_t h2 = vcvt_f16_f32(f2);
            float16x4_t h3 = vcvt_f16_f32(f3);
            if (0 != biasLen) {
                h0 = vadd_f16(h0, bias[0]);
                h1 = vadd_f16(h1, bias[1]);
                h2 = vadd_f16(h2, bias[2]);
                h3 = vadd_f16(h3, bias[3]);
            }
            vst1_f16(d + i, h0);
            vst1_f16(d + i + 4, h1);
            vst1_f16(d + i + 8, h2);
            vst1_f16(d + i + 12, h3);
        }

        for (; i < (int)len; i++) {
            d[i] = q[i] * factor;
            if (0 != biasLen) {
                d[i] += biasPtr[i % biasLen];
            }
        }
    } else {
        for (int i = 0; i < ((int)len); i += biasLen) {
            int j = 0;
            for (; j < ((int)biasLen) - 3; j += 4) {
                int32x4_t in0 = vld1q_s32(q + i + j);
                bias[0] = vld1_f16(biasPtr + j);
                float32x4_t f0 = vcvtq_f32_s32(in0);
                f0 = vmulq_n_f32(f0, factor);
                float16x4_t h0 = vcvt_f16_f32(f0);
                h0 = vadd_f16(h0, bias[0]);
                vst1_f16(d + i + j, h0);
            }
            for (; j < (int)biasLen; j++) {
                d[i + j] = q[i + j] * factor + biasPtr[j];
            }
        }
    }
}

void update_histogram(U32 len, const F16 *data, int numBins, F32 interval, F32 *histo)
{
    for (U32 i = 0; i < len; i++) {
        F32 tmp = data[i];
        int index = floor(abs(tmp) / interval);
        if (index >= numBins) {
            index = numBins - 1;
        }
        histo[index] += 1;
    }
}

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
#endif

std::vector<F32> compute_scale_with_KL(std::vector<F32> &histogram, F32 interval)
{
    std::vector<F32> scale;
#ifdef _USE_INT8
    const int BINS = 2048;
    F32 histoSum = array_sum_f32(histogram.data(), BINS);
    array_scale_f32(histogram.data(), histogram.data(), BINS, 1 / histoSum, 0);

    F32 minKLD = 2048;
    int bestThreshold = 128;
    F32 sumBin = array_sum_f32(histogram.data(), 128);
    UNI_DEBUG_LOG("First 128 bins contain %f of values", sumBin);
    F32 sumOver = 1 - sumBin;

    for (U32 i = 128; i < 2048; i++) {
        std::vector<F32> clipDist(histogram.begin(), histogram.begin() + i);
        clipDist[i - 1] += sumOver;
        sumOver -= histogram[i];  // Prepare for next round

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
    F32 quantScale = 127.99 / threshold;
    scale.push_back(quantScale);
#endif
    return scale;
}
