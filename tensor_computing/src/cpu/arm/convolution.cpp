// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string.h>
#include <float.h>
#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif
#ifdef _USE_INT8
#include "cpu/arm/int8/tensor_computing_int8.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/bnn/tensor_computing_bnn.h"
#endif
#include "ut_util.h"
#include "tensor_computing_library_algorithm_search.h"

EE convolution_infer_forward_algorithm_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm, DataType targetDataType)
{
    UNUSED(outputDesc);
    if (nullptr == algorithm)
        CHECK_STATUS(NULL_POINTER);
    if (*algorithm != CONVOLUTION_ALGORITHM_NULL)
        return SUCCESS;
    if (policy == CONVOLUTION_LIBRARY_SEARCH) {
        if (libraryAlgorithmMap.size() == 0) {
            loadLibraryAlgorithmMapFromTxt();
        }
        std::string name = "convolution_cpu_" + getConvolutionAlgorithmMapNameFromInput(inputDesc,
            filterDesc, convDesc, targetDataType);
        if (libraryAlgorithmMap.find(name) != libraryAlgorithmMap.end()) {
            *algorithm = (ConvolutionForwardAlgorithm)libraryAlgorithmMap[name];
            return SUCCESS;
        } else {
            policy = CONVOLUTION_FASTEST;
        }
    }

    EE ret = SUCCESS;
    if (policy == CONVOLUTION_FASTEST) {
        DataType idt, fdt;
        DataFormat idf, fdf;
        U32 in, ic, ih, iw;
        U32 fn, fc, fh, fw;
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
        U32 strideH = convDesc.stride_h;
        U32 strideW = convDesc.stride_w;
        U32 paddingT = convDesc.padding_top;
        U32 paddingB = convDesc.padding_bottom;
        U32 paddingL = convDesc.padding_left;
        U32 paddingR = convDesc.padding_right;
        U32 dilateH = convDesc.dilatedRate_h;
        U32 dilateW = convDesc.dilatedRate_w;
        if (dilateH > 1 || dilateW > 1) {
            *algorithm = CONVOLUTION_ALGORITHM_GEMM;
            return SUCCESS;
        }

        if (ic % 8 != 0) {
            *algorithm = CONVOLUTION_ALGORITHM_GEMM_ICNCHW;
        } else if (fh == 3 && fw == 3 && strideH == 1 && strideW == 1 && paddingT == 1 && paddingB == 1 && paddingL == 1 && paddingR == 1) {
            *algorithm = CONVOLUTION_ALGORITHM_WINOGRAD;
        } else {
            *algorithm = CONVOLUTION_ALGORITHM_GEMM;
        }

        switch (targetDataType) {
            case DT_BIN01: {
                *algorithm = CONVOLUTION_ALGORITHM_BNN;
                break;
            }
            case DT_BIN11: {
                *algorithm = CONVOLUTION_ALGORITHM_BNN;
                break;
            }
            default:
                break;
        }
    } else if (policy == CONVOLUTION_TUNNING) {
        std::vector<ConvolutionForwardAlgorithm> convolutionAlgorithms;
        U32 filterBytes = 0;
        U32 tmpBytes = 0;
        for (U32 i = 0; i < convolutionAlgorithms.size(); i++) {
            U32 bytes = 0;
            CHECK_STATUS(convolution_transform_filter_bytes_arm(filterDesc, convolutionAlgorithms[i], &bytes));
            filterBytes = (bytes > filterBytes) ? bytes : filterBytes;
            CHECK_STATUS(convolution_infer_forward_tmp_bytes_arm(inputDesc, filterDesc, outputDesc,
                convDesc, convolutionAlgorithms[i], &bytes));
            tmpBytes = (bytes > tmpBytes) ? bytes : tmpBytes;
        }
        TensorDesc biasDesc  = tensor1d(filterDesc.dt, outputDesc.dims[3]);
        TensorDesc scaleDesc = tensor1d(DT_F32, outputDesc.dims[2]);
        U8 *input  = ut_input_v(tensorNumElements(inputDesc), inputDesc.dt, UT_INIT_RANDOM);
        U8 *filter = ut_input_v(tensorNumElements(filterDesc), filterDesc.dt, UT_INIT_RANDOM);
        U8 *filterTransformed = ut_input_v(filterBytes/bytesOf(filterDesc.dt), filterDesc.dt, UT_INIT_RANDOM);
        U8 *bias   = ut_input_v(tensorNumElements(biasDesc), biasDesc.dt, UT_INIT_RANDOM);
        U8 *scale  = ut_input_v(tensorNumElements(scaleDesc), scaleDesc.dt, UT_INIT_RANDOM);
        U8 *tmp    = ut_input_v(tmpBytes/bytesOf(inputDesc.dt), inputDesc.dt, UT_INIT_ZERO);
        U8 *output = ut_input_v(tensorNumElements(outputDesc), outputDesc.dt, UT_INIT_ZERO);
        U32 algorithmIndex = 0;
        for (U32 i = 0; i < convolutionAlgorithms.size(); i++) {
            TensorDesc ftmDesc;
            CHECK_STATUS(convolution_transform_filter_arm(filterDesc, filter,
                convolutionAlgorithms[i],
                &ftmDesc, filterTransformed));

            memset(tmp, 0, tmpBytes);
            double timeStart = ut_time_ms();
            CHECK_STATUS(convolution_arm(inputDesc, input,
                ftmDesc, filterTransformed,
                convDesc,
                convolutionAlgorithms[i],
                scaleDesc, scale,
                biasDesc, bias,
                tmpBytes, tmp,
                outputDesc, output,
                ACTIVATION_RELU,
                ARM_A76));
            double timeEnd = ut_time_ms();
            double timeMin = FLT_MAX;
            if (timeMin > timeEnd - timeStart) {
                timeMin = timeEnd - timeStart;
                algorithmIndex = i;
            }
        }
        free(input);
        free(filter);
        free(filterTransformed);
        free(bias);
        free(scale);
        free(tmp);
        free(output);
        *algorithm = convolutionAlgorithms[algorithmIndex];
        ret = SUCCESS;;
    } else {
        ret = NOT_SUPPORTED;
    }
    return ret;
}

EE convolution_transform_filter_bytes_arm(TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32* bytes)
{
    if (nullptr == bytes)
        CHECK_STATUS(NULL_POINTER);
    EE ret = SUCCESS;
    
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            *bytes = fn * fc * 6 * 6;
            break;
        case CONVOLUTION_ALGORITHM_DIRECT:
            *bytes = fn * fc * fh * fw;
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            *bytes = fn * fc * fh * fw;
            break;
        case CONVOLUTION_ALGORITHM_GEMM_ICNCHW:
            *bytes = fn * fc * fh * fw;
            break;
        case CONVOLUTION_ALGORITHM_BNN:
            *bytes = fn * fc * fh * fw;
            break;
        default:
            return NOT_SUPPORTED;
    }
    *bytes *= bytesOf(fdt);

    switch (filterDesc.dt) {
        case DT_BIN01: {
            *bytes /= 8;
            break;
        }
        case DT_BIN11: {
            *bytes /= 8;
            break;
        }
        default:
            break;
    }
    *bytes += 32;
    return ret;
}

EE convolution_transform_filter_arm(TensorDesc filterDesc, const void* filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, void* filterTransformed)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = convolution_transform_filter_fp32(filterDesc, (F32*)filter, algorithm, ftmDesc, (F32*)filterTransformed);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = convolution_transform_filter_fp16(filterDesc, (F16*)filter, algorithm, ftmDesc, (F16*)filterTransformed);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = convolution_transform_filter_int8(filterDesc, filter, algorithm, ftmDesc, filterTransformed);
            break;
        }
        case DT_F16_8Q: {
            ret = convolution_transform_filter_int8(filterDesc, filter, algorithm, ftmDesc, filterTransformed);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_BIN01: {
            ret = convolution_transform_filter_bnn(filterDesc, (BIN8*)filter, ftmDesc, (BIN8*)filterTransformed);
            break;
        }
        case DT_BIN11: {
            ret = convolution_transform_filter_bnn(filterDesc, (BIN8*)filter, ftmDesc, (BIN8*)filterTransformed);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = convolution_infer_forward_tmp_bytes_fp32(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = convolution_infer_forward_tmp_bytes_fp16(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = convolution_infer_forward_tmp_bytes_int8(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_BIN01: {
            ret = convolution_infer_forward_tmp_bytes_bnn(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
        case DT_BIN11: {
            ret = convolution_infer_forward_tmp_bytes_bnn(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;

}

EE convolution_arm(TensorDesc inputDesc, void* input,
    TensorDesc filterDesc, const void* filter,
    ConvolutionDesc convDesc,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc scaleDesc, const void* scale,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output,
    ActivationMode activationMode,
    Arch arch)
{
    EE ret = SUCCESS;
    UNUSED(scaleDesc);
    UNUSED(scale);
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = convolution_fp32(inputDesc, (F32*)input,
                                   filterDesc, (F32*)filter,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (F32*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, (F32*)output,
                                   activationMode,
                                   arch);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = convolution_fp16(inputDesc, (F16*)input,
                                   filterDesc, (F16*)filter,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (F16*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, (F16*)output,
                                   activationMode,
                                   arch);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = convolution_int8(inputDesc, (INT8*)input,
                                   filterDesc, (INT8*)filter,
                                   (F16*)scale,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (F16*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, output,
                                   activationMode,
                                   arch);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_BIN01: {
            ret = convolution_bnn(inputDesc, (F16*)input,
                                         filterDesc, (BIN8*)filter,
                                         convDesc,
                                         scaleDesc, (F16*)scale,
                                         biasDesc, (F16*)bias,
                                         tmpBytes, tmp,
                                         outputDesc, (F16*)output,
                                         activationMode,
                                         arch);
            break;
        }
        case DT_BIN11: {
            ret = convolution_bnn(inputDesc, (F16*)input,
                                       filterDesc, (BIN8*)filter,
                                       convDesc,
                                       scaleDesc, (F16*)scale,
                                       biasDesc, (F16*)bias,
                                       tmpBytes, tmp,
                                       outputDesc, (F16*)output,
                                       activationMode,
                                       arch);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
