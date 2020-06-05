// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/arm/fp16/tensor_computing_fp16.h"
#include "cpu/arm/fp16/convolution_winograd_transform.h"

inline EE deconvolution_transform_filter_kernel_fp16(TensorDesc filterDesc, const F16* filterArray,
    TensorDesc *ftmDesc, F16* ftmArray,
    DataFormat ftmDataFormat)
{
    // Procedure should be the same, but fhfw is reversed
    if (nullptr == filterArray || nullptr == ftmDesc || nullptr == ftmArray) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (fdf == ftmDataFormat) {
        *ftmDesc = filterDesc;
        memcpy(ftmArray, filterArray, fn*fc*fh*fw*bytesOf(fdt));
        return SUCCESS;
    }
    if (fdf != DF_NCHW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    EE ret = SUCCESS;
    switch (ftmDataFormat) {
        case DF_NHWCN16: {
            /*
             *  CNHW => NHWCN16
             *  if there is remainder, it should be NHWCN8
             */
            U32 oc = fc / 16;
            U32 hwMax = fh * fw - 1;
            for (U32 o = 0; o < oc; o++) {
                for (U32 hw = 0; hw < fh*fw; hw++) {
                    for (U32 c = 0; c < fn; c++) {
                        for (U32 o16 = 0; o16 < 16; o16++) {
                            ftmArray[o*fh*fw*fn*16 + hw*fn*16 + c*16 + o16] = filterArray[c*fc*fh*fw + (o*16+o16)*fh*fw + hwMax - hw];
                        }
                    }
                }
            }
            if (fc != oc * 16) {
                for (U32 hw = 0; hw < fh*fw; hw++) {
                    for (U32 c = 0; c < fn; c++) {
                        for (U32 o8 = 0; o8 < 8; o8++) {
                            ftmArray[(oc*16)*fh*fw*fn + hw*fn*8 + c*8 + o8] = filterArray[c*fc*fh*fw + (oc*16+o8)*fh*fw + hwMax - hw];
                        }
                    }
                }
            }
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fc, fn, fh, fw);
            break;
        }
        case DF_HWNCN16: {
            /*
             *  CNHW => NHWCN16 + NHWCN8 if there is remainder divided by 16
             */
            const U32 hwMax = 8;
            
            for (U32 o = 0; o < fc/16; o++) {
                for (U32 c = 0; c < fn; c++) {
                    U32 f_off_0 = c*fc*fh*fw + (o*16)*fh*fw;
                    U32 f_off_1 = c*fc*fh*fw + (o*16+8)*fh*fw;
                    U32 ftm_off_0 = o*36*fn*16 + c*16;
                    U32 ftm_off_1 = o*36*fn*16 + c*16 + 8;
                    F16 F[9][8];
                    F16 *F_ptr[9];
                    F16 *Fw[36];

                    for (U32 hw = 0; hw < 9; hw++) {
                        for (U32 oo = 0; oo < 8; oo++) {
                            F[hw][oo] = filterArray[f_off_0 + hwMax - hw + oo*fh*fw];
                        }
                        F_ptr[hw] = F[hw];
                    }
                    for (U32 hw = 0; hw < 36; hw++) {
                        Fw[hw] = ftmArray + ftm_off_0 + hw*fn*16;
                    }
                    trans_W_4x4_3x3(Fw, F_ptr);
                    for (U32 hw = 0; hw < 9; hw++) {
                        for (U32 oo = 0; oo < 8; oo++) {
                            F[hw][oo] = filterArray[f_off_1 + hwMax - hw + oo*fh*fw];
                        }
                        F_ptr[hw] = F[hw];
                    }
                    for (U32 hw = 0; hw < 36; hw++) {
                        Fw[hw] = ftmArray + ftm_off_1 + hw*fn*16;
                    }
                    trans_W_4x4_3x3(Fw, F_ptr);
                }
            }
            U32 oc = (fc / 16) * 16;
            if (oc != fc) {
                for (U32 c = 0; c < fn; c++) {
                    U32 f_off_0 = c*fc*fh*fw + oc*fh*fw;
                    U32 ftm_off_0 = oc*36*fn + c*8;
                    F16 F[9][8];
                    F16 *F_ptr[9];
                    F16 *Fw[36];
                    for (U32 hw = 0; hw < 9; hw++) {
                        for (U32 oo = 0; oo < 8; oo++) {
                            F[hw][oo] = filterArray[f_off_0 + hwMax - hw + oo*fh*fw];
                        }
                        F_ptr[hw] = F[hw];
                    }
                    for (U32 hw = 0; hw < 36; hw++) {
                        Fw[hw] = ftmArray + ftm_off_0 + hw*fn*8;
                    }
                    trans_W_4x4_3x3(Fw, F_ptr);
                }
            }
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fc, fn, 6, 6);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE deconvolution_transform_filter_fp16(TensorDesc filterDesc, const F16* filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, F16* filterTransformed)
{
    DataFormat ftmDataFormat;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            ftmDataFormat = DF_HWNCN16;
            break;
        case CONVOLUTION_ALGORITHM_GEMM_ICNCHW:
            ftmDataFormat = DF_NHWCN16;
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            ftmDataFormat = DF_NHWCN16;
            break;
        default:
            return NOT_MATCH;
    }
    EE ret = deconvolution_transform_filter_kernel_fp16(filterDesc, filter, ftmDesc, filterTransformed, ftmDataFormat);
    CHECK_STATUS(ret);
    return ret;
}
