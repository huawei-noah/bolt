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

#include "tensor_computing.h"
#include "blas_enhance.h"
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

// input format: NCHW|NCHWC8|NORMAL
// weight(filter) format: NORMAL
// result format: NORMAL

inline EE fully_connected_infer_output_size_cpu(
    TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc *outputDesc)
{
    if (outputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fh, fw;
    if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &iw));
        ic = 1;
        ih = 1;
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        if (idf != DF_NCHW && idf != DF_NCHWC8) {
            CHECK_STATUS(NOT_MATCH);
        }
    } else {
        return NOT_MATCH;
    }

    CHECK_REQUIREMENT(tensorIs2d(filterDesc));
    CHECK_STATUS(tensor2dGet(filterDesc, &fdt, &fdf, &fh, &fw));
    if (fdf != DF_TRANSPOSE) {
        CHECK_STATUS(NOT_MATCH);
    }

    if (fw != ic * ih * iw) {
        CHECK_STATUS(NOT_MATCH);
    }

    *outputDesc = tensor2df(idt, DF_NORMAL, in, fh);
    return SUCCESS;
}

EE fully_connected_infer_output_size(
    Tensor *inputTensor, Tensor filterTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc filterDesc = filterTensor.get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = fully_connected_infer_output_size_mali(
            inputDesc, filterDesc, &outputDesc, &gclmemInputDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    } else {
        ret = fully_connected_infer_output_size_cpu(inputDesc, filterDesc, &outputDesc);
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE fully_connected_infer_forward_algorithm(
    Tensor inputTensor, Tensor filterTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc filterDesc = filterTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        ret = fully_connected_infer_forward_algorithm_mali(
            ((MaliPara_t)(archInfo->archPara))->handle, inputDesc, filterDesc, outputDesc,
            gclmemInputDesc, gclmemOutputDesc, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        UNUSED(inputTensor);
        UNUSED(filterTensor);
        UNUSED(outputTensor);
    }
    return ret;
}
EE fully_connected_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor filterTensor, U32 *bytes, ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc filterDesc = filterTensor.get_desc();
    // Match dt in int8 inference
    inputDesc.dt = filterDesc.dt;

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        ret = fully_connected_infer_forward_tmp_bytes_mali(
            inputDesc, filterDesc, bytes, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        if (bytes == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        if (tensorIs2d(inputDesc)) {
            CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &iw));
            ic = ih = 1;
        } else if (tensorIs4d(inputDesc)) {
            CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        } else {
            return NOT_MATCH;
        }

        if (in != 1) {
            // call gemm
            TensorDesc in_desc = tensor2df(idt, DF_NORMAL, in, ic * ih * iw);
            ret = matrix_matrix_multiply_tmp_bytes(in_desc, filterDesc, bytes, archInfo->arch);
        } else {
            // call gemv
            TensorDesc in_desc = tensor1d(idt, ic * ih * iw);
            ret = matrix_vector_multiply_tmp_bytes(filterDesc, in_desc, bytes, archInfo->arch);
        }
        if (DT_I8 == filterDesc.dt) {
            if (DT_F16 == inputTensor.get_desc().dt) {
                *bytes += tensorNumBytes(inputDesc);
            }
            *bytes += filterDesc.dims[0] * bytesOf(DT_I32);       // Bias
            *bytes += in * filterDesc.dims[1] * bytesOf(DT_I32);  // Results before quantization
        }
    }
    return ret;
}

EE fully_connected_transform_filter_bytes(Tensor filterTensor, U32 *bytes, ArchInfo_t archInfo)
{
    TensorDesc filterDesc = filterTensor.get_desc();

    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        CHECK_STATUS(fully_connected_transform_filter_bytes_mali(filterDesc,
            ((MaliPara_t)(archInfo->archPara))->gclmemFilterDesc, bytes,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo));
#endif
    } else {
        if (bytes == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }
        *bytes = tensorNumBytes(filterDesc) + 32;
    }
    return SUCCESS;
}

template <typename T>
EE fully_connected_transform_filter_kernel(TensorDesc inputDesc,
    TensorDesc filterDesc,
    const void *filter,
    TensorDesc *ftmDesc,
    void *filterTransformed)
{
    if (filter == nullptr || ftmDesc == nullptr || filterTransformed == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fh, fw;
    if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &iw));
        ic = ih = 1;
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    } else {
        return NOT_MATCH;
    }
    CHECK_STATUS(tensor2dGet(filterDesc, &fdt, &fdf, &fh, &fw));

    if (fw != ic * ih * iw) {
        CHECK_STATUS(NOT_MATCH);
    }
    bool need_transpose = false;
    if (in > 1) {
        need_transpose = true;
    }

    if (idf == DF_NCHW || idf == DF_NORMAL) {
        if (need_transpose) {
            T *f_ptr = (T *)filter;
            T *ftm_ptr = (T *)filterTransformed;
            for (U32 h = 0; h < fh; h++) {
                for (U32 w = 0; w < fw; w++) {
                    U32 f_index = h * fw + w;
                    U32 ftm_index = w * fh + h;
                    ftm_ptr[ftm_index] = f_ptr[f_index];
                }
            }
        } else {
            memcpy(filterTransformed, filter, tensorNumBytes(filterDesc));
        }
    } else if (idf == DF_NCHWC8) {
        U32 align = 8;
        if (ic % align != 0) {
            align = 1;
        }
        U32 ic_new = ic / align;
        T *f_ptr = (T *)filter;
        T *ftm_ptr = (T *)filterTransformed;
        for (U32 h = 0; h < fh; h++) {
            for (U32 w = 0; w < fw; w++) {
                U32 i_n = w / (ic * ih * iw);
                U32 remain = w % (ic * ih * iw);
                U32 i_c = remain / (ih * iw);
                remain = remain % (ih * iw);
                U32 i_h = remain / iw;
                U32 i_w = remain % iw;
                U32 i_c_outer = i_c / align;
                U32 i_c_inner = i_c % align;
                U32 h_new = h;
                U32 w_new = (((i_n * ic_new + i_c_outer) * ih + i_h) * iw + i_w) * align + i_c_inner;
                U32 ld = fw;
                if (need_transpose) {
                    U32 tmp = h_new;
                    h_new = w_new;
                    w_new = tmp;
                    ld = fh;
                }
                U32 f_index = h * fw + w;
                U32 ftm_index = h_new * ld + w_new;
                ftm_ptr[ftm_index] = f_ptr[f_index];
            }
        }
    } else {
        return NOT_MATCH;
    }

    U32 fh_after = fh;
    U32 fw_after = fw;
    if (need_transpose) {
        fh_after = fw;
        fw_after = fh;
    }
    *ftmDesc = tensor2df(fdt, DF_NORMAL, fh_after, fw_after);
    return SUCCESS;
}

EE fully_connected_transform_filter(
    Tensor inputTensor, Tensor filterTensor, Tensor *ftmTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc filterDesc = filterTensor.get_desc();
    void *filter = get_ptr_from_tensor(filterTensor, arch);
    TensorDesc ftmDesc = ftmTensor->get_desc();
    void *filterTransformed = get_ptr_from_tensor(*ftmTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        ret = fully_connected_transform_filter_mali(((MaliPara_t)(archInfo->archPara))->handle,
            filterDesc, (GCLMem_t)filter, &ftmDesc, (GCLMem_t)filterTransformed,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        switch (filterDesc.dt) {
#ifdef _USE_FP16
            case DT_F16: {
                ret = fully_connected_transform_filter_kernel<F16>(
                    inputDesc, filterDesc, filter, &ftmDesc, filterTransformed);
                break;
            }
#endif
#ifdef _USE_FP32
            case DT_F32: {
                ret = fully_connected_transform_filter_kernel<F32>(
                    inputDesc, filterDesc, filter, &ftmDesc, filterTransformed);
                break;
            }
#endif
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    }
    ftmTensor->resize(ftmDesc);
    return ret;
}

EE fully_connected(Tensor inputTensor,
    Tensor filterTensor,
    Tensor biasTensor,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc filterDesc = filterTensor.get_desc();
    void *filter = get_ptr_from_tensor(filterTensor, arch);
    TensorDesc biasDesc = biasTensor.get_desc();
    void *bias = get_ptr_from_tensor(biasTensor, arch);
    U32 tmpBytes = tmpTensor.bytes();
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        ret = fully_connected_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, filterDesc, (GCLMem_t)filter, biasDesc, (GCLMem_t)bias, tmpBytes,
            (GCLMem_t)tmp, outputDesc, (GCLMem_t)output,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        if (input == nullptr || filter == nullptr || output == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }

#ifdef _USE_INT8
        F32 scaleI = inputTensor.get_scale();
        if (DT_I8 == filterDesc.dt) {
            if (DT_F16 == inputDesc.dt) {
                F16 *inD = (F16 *)input;
                INT8 *inQ = (INT8 *)tmp;
                F16 scale = scaleI;
                quantize_tensor(inputDesc, inD, &inputDesc, inQ, &scale);
                scaleI = scale;
                input = (U8 *)tmp;
                tmp = (U8 *)tmp + tensorNumBytes(inputDesc);
            }
            if (nullptr != bias) {
                if (DT_F16 == outputDesc.dt) {  // dequantize and then add bias
                    bias = nullptr;
                } else {
                    CHECK_REQUIREMENT(DT_I8 == outputDesc.dt);
                    biasDesc.dt = DT_I32;
                    F16 *biasF = (F16 *)bias;
                    I32 *biasI = (I32 *)tmp;
                    F32 scale = scaleI * filterTensor.get_scale();
                    for (U32 i = 0; i < tensorNumElements(biasDesc); i++) {
                        biasI[i] = round(scale * biasF[i]);
                    }
                    bias = tmp;
                    tmp = (U8 *)tmp + tensorNumBytes(biasDesc);
                }
            }
            outputDesc.dt = DT_I32;
            output = tmp;
            tmp = (U8 *)tmp + tensorNumBytes(outputDesc);
        }
#endif

        U32 in, ic, ih, iw;
        U32 oh, ow;
        U32 fh, fw, bw;
        DataType idt, fdt, odt, bdt;
        DataFormat idf, fdf, odf, bdf;
        if (tensorIs2d(inputDesc)) {
            CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &iw));
            ic = ih = 1;
        } else if (tensorIs4d(inputDesc)) {
            CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        } else {
            CHECK_STATUS(NOT_MATCH);
        }

        CHECK_REQUIREMENT(tensorIs2d(filterDesc));
        CHECK_STATUS(tensor2dGet(filterDesc, &fdt, &fdf, &fh, &fw));
        CHECK_STATUS(tensor2dGet(outputDesc, &odt, &odf, &oh, &ow));

        if (bias != nullptr) {
            CHECK_STATUS(tensor1dGet(biasDesc, &bdt, &bdf, &bw));
            if (bw != ow) {
                CHECK_STATUS(NOT_MATCH);
            } else {
                U8 *outArray = (U8 *)output;
                U32 size = tensorNumBytes(biasDesc);
                for (U32 i = 0; i < in; i++) {
                    memcpy(outArray + i * size, bias, size);
                }
            }
        } else {
            memset(output, 0, tensorNumBytes(outputDesc));
        }
        if (in == 1 &&
            fdf != targetFormat4MatrixB(fdt)) {  // If weight is transformed for mmm, don't run as mvm
            TensorDesc vectorDesc = tensor1d(idt, ic * ih * iw);
            TensorDesc resultDesc = tensor1d(odt, ow);
            ret = matrix_vector_multiply(filterDesc, filter, vectorDesc, input, tmpBytes, tmp,
                resultDesc, output, archInfo->arch);
        } else {
            TensorDesc in_desc = tensor2df(idt, DF_NORMAL, in, ic * ih * iw);
            ret = matrix_matrix_multiply(in_desc, input, filterDesc, filter, tmpBytes, tmp,
                outputDesc, output, archInfo->arch);
        }
#ifdef _USE_INT8
        F32 scale = scaleI * filterTensor.get_scale();
        if (DT_I8 == filterDesc.dt) {
            if (DT_I8 == outputTensor.get_desc().dt) {
                CHECK_STATUS(quantize_tensor(outputDesc, output, &outputDesc,
                    get_ptr_from_tensor(outputTensor, arch), &scale));
                outputTensor.set_scale(scale);
            } else {
                CHECK_REQUIREMENT(DT_F16 == outputTensor.get_desc().dt);
                F16 *biasF = (F16 *)get_ptr_from_tensor(biasTensor, arch);
                U32 biasLen = nullptr == biasF ? 0 : tensorNumElements(biasDesc);
                dequantize_int32_to_fp16(tensorNumElements(outputDesc), (I32 *)output, scale,
                    (F16 *)get_ptr_from_tensor(outputTensor, arch), biasLen, biasF);
            }
        }
#endif
    }
    return ret;
}
