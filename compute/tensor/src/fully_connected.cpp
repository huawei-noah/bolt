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
#include "blas_enhance.h"
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif
#ifdef _USE_CPU
#include "cpu/tensor_computing_cpu.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif

inline EE fully_connected_infer_output_size_cpu(
    TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc *outputDesc)
{
    DataType fdt;
    DataFormat fdf;
    U32 fh, fw;
    CHECK_REQUIREMENT(tensorIs2d(filterDesc));
    CHECK_STATUS(tensor2dGet(filterDesc, &fdt, &fdf, &fh, &fw));
    if (fdf != DF_TRANSPOSE) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 sum = 1;
    int last = -1;
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        sum *= inputDesc.dims[i];
        if (sum == fw) {
            last = i;
            break;
        }
    }
    if (last == -1) {
        return NOT_MATCH;
    }
    outputDesc->dims[0] = fh;
    for (U32 j = last + 1, i = 1; j < inputDesc.nDims; j++, i++) {
        outputDesc->dims[i] = inputDesc.dims[j];
    }
    outputDesc->dt = inputDesc.dt;
    outputDesc->nDims = inputDesc.nDims - last;
    outputDesc->df = getTensorDefaultDataFormat(outputDesc->nDims);
    return SUCCESS;
}

EE fully_connected_infer_output_size(
    Tensor *inputTensor, Tensor filterTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc filterDesc = filterTensor.get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        OclMemory *inputMem = (OclMemory *)inputTensor->get_memory();
        OclMemory *outputMem = (OclMemory *)outputTensor->get_memory();
        ret = fully_connected_padding_input_mali(
            inputDesc, filterDesc, &outputDesc, inputMem, outputMem);
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
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
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
        ret = SUCCESS;
    }
    return ret;
}

EE fully_connected_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor filterTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    // Match dt in int8 inference
    // inputDesc.dt = filterDesc.dt;
    if (bytes == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc filterDesc = filterTensor.get_desc();
    TensorDesc outputDesc = outputTensor.get_desc();

    EE ret = NOT_SUPPORTED;
    Arch arch = archInfo->arch;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        ret = fully_connected_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, outputDesc,
            gclmemInputDesc, bytes, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        U32 fw = outputDesc.dims[0];
        CHECK_REQUIREMENT(tensorIs2d(filterDesc));
        U32 fh = tensorNumElements(filterDesc) / fw;
        U32 M = tensorNumElements(inputDesc) / fh;
        if (M != 1) {
            // call gemm
            TensorDesc in_desc = tensor2df(inputDesc.dt, DF_NORMAL, M, fh);
            ret = matrix_matrix_multiply_tmp_bytes(in_desc, filterDesc, bytes, arch);
        } else {
            // call gemv
            TensorDesc in_desc = tensor1d(inputDesc.dt, fh);
            ret = matrix_vector_multiply_tmp_bytes(filterDesc, in_desc, bytes, arch);
        }
        if (inputDesc.df == DF_NCHWC8) {
            U32 ihiw = 1;
            for (int i = inputDesc.nDims - 3; i >= 0; i--) {
                ihiw *= inputDesc.dims[i];
            }
            if (ihiw > 1) {
                *bytes += tensorNumBytes(inputDesc);
            }
        }
        if (DT_I8 == filterDesc.dt || DT_F32_8Q == filterDesc.dt) {
            if (DT_I8 != inputTensor.get_desc().dt) {
                *bytes += tensorNumBytes(inputDesc);
            }
            *bytes += fw * bytesOf(DT_I32);                             // Bias
            *bytes += tensorNumElements(outputDesc) * bytesOf(DT_I32);  // Results before quantization
        }
    }
    return ret;
}
EE fully_connected_transform_filter_bytes(Tensor filterTensor, void *bytes, ArchInfo_t archInfo)
{
    if (bytes == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    TensorDesc filterDesc = filterTensor.get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        ret = fully_connected_transform_filter_bytes_mali(
            filterDesc, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, (TensorDesc *)bytes);
#endif
    } else {
        U32 *p = (U32 *)bytes;
        *p = tensorNumBytes(filterDesc);
        //        U32 fh, fw;
        //        DataType fdt;
        //        DataFormat fdf;
        //        CHECK_REQUIREMENT(tensorIs2d(filterDesc));
        //        if (filterDesc.df == DF_TRANSPOSE) {
        //            CHECK_STATUS(tensor2dGet(filterDesc, &fdt, &fdf, &fw, &fh));
        //        } else if (filterDesc.df == DF_NORMAL) {
        //            CHECK_STATUS(tensor2dGet(filterDesc, &fdt, &fdf, &fh, &fw));
        //        } else {
        //            return NOT_SUPPORTED;
        //        }
        //
        //        U32 *size = (U32 *)bytes;
        //        *size = 0;
        //        U32 alignments = 1;
        //        if (IS_ARM(archInfo->arch)) {
        //            alignments = 4;
        //        } else if (IS_X86(archInfo->arch)) {
        //            alignments = 8;
        //#ifdef _USE_INT8
        //            alignments = 16;
        //            fh = UNI_ALIGN(fh, 8);
        //            // for x86 int8 offset
        //            *size += UNI_MAX(fw, fh) * bytesOf(DT_I32);
        //#endif
        //        }
        //        fw = UNI_ALIGN(fw, alignments);
        //        *size += fw * fh + UNI_SOFTPIPELINE_PREFETECH;
        //        *size *= bytesOf(fdt);
        ret = SUCCESS;
    }
    return ret;
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

    DataType fdt;
    DataFormat fdf;
    U32 fh, fw;
    CHECK_STATUS(tensor2dGet(filterDesc, &fdt, &fdf, &fh, &fw));
    if (fdf != DF_TRANSPOSE) {
        return NOT_MATCH;
    }
    U32 M = tensorNumElements(inputDesc) / fw;
    U32 ihiw = 1;
    if (inputDesc.df == DF_NCHWC8) {
        for (int i = inputDesc.nDims - 3; i >= 0; i--) {
            ihiw *= inputDesc.dims[i];
        }
    }
    bool need_transpose = false;
    if (M > 1) {
        need_transpose = true;
    }

    DataFormat idf = inputDesc.df;
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
        UNI_MEMCPY(filterTransformed, filter, tensorNumBytes(filterDesc));
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
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
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
    std::vector<Tensor> tmpTensors,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc filterDesc = filterTensor.get_desc();
    void *filter = get_ptr_from_tensor(filterTensor, arch);
    void *bias = get_ptr_from_tensor(biasTensor, arch);
    TensorDesc biasDesc;
    if (bias) {
        biasDesc = biasTensor.get_desc();
    }
    U32 tmpBytes = tmpTensors[0].bytes();
    void *tmp = get_ptr_from_tensor(tmpTensors[0], arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        std::vector<GCLMem_t> tmpVec(2, NULL);
        for (U32 i = 0; i < tmpTensors.size(); i++) {
            tmpVec[i] = (GCLMem_t)get_ptr_from_tensor(tmpTensors[i], arch);
        }
        ret = fully_connected_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, filterDesc, (GCLMem_t)filter, biasDesc, (GCLMem_t)bias, tmpBytes,
            tmpVec, outputDesc, (GCLMem_t)output,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        if (input == nullptr || filter == nullptr || output == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }

        if (inputDesc.df == DF_NCHWC8) {
            U32 ihiw = 1;
            for (int i = inputDesc.nDims - 3; i >= 0; i--) {
                ihiw *= inputDesc.dims[i];
            }
            if (ihiw > 1) {
                TensorDesc tmpInputDesc = inputDesc;
                tmpInputDesc.df = DF_NCHW;
                transformToNCHW(inputDesc, input, tmpInputDesc, tmp);
                input = tmp;
                tmp = (U8 *)tmp + tensorNumBytes(tmpInputDesc);
                tmpBytes -= tensorNumBytes(tmpInputDesc);
                inputDesc.df = DF_NCHW;
            }
        }

        U32 fw = outputDesc.dims[0];
        CHECK_REQUIREMENT(tensorIs2d(filterDesc));
        U32 fh = tensorNumElements(filterDesc) / fw;
        U32 M = tensorNumElements(inputDesc) / fh;

        F32 *scale = nullptr;
        DataType idt = inputDesc.dt;
        DataType fdt = filterDesc.dt;
        DataType odt = outputDesc.dt;

#ifdef _USE_INT8
        F32 scaleI = inputTensor.get_scale();
        F32 scaleO = outputTensor.get_scale();
        F32 scaleArray[2] = {-1, -1};
        if (DT_I8 == filterDesc.dt) {
            TensorDesc qIDesc = inputDesc;
            TensorDesc qODesc = outputDesc;
            if (IS_X86(arch)) {
                qIDesc.dt = DT_U8_Q;
                if (outputDesc.dt == DT_F32) {
                    scale = &scaleO;
                } else if ((outputDesc.dt == DT_U8_Q) || (outputDesc.dt == DT_I8)) {
                    if (scaleO > 0) {
                        scale = scaleArray;
                        scale[1] = scaleO;
                    } else {
                        qODesc.dt = DT_I32;
                    }
                }
            } else {
                qIDesc.dt = DT_I8;
                qODesc.dt = DT_I32;
            }
            CHECK_REQUIREMENT(idt == qIDesc.dt);
            scaleO = scaleI * filterTensor.get_scale();

            if (IS_X86(arch)) {
#ifdef _USE_X86
                U8 *offsetC = (U8 *)tmp;
                if (outputDesc.dt != qODesc.dt) {
                    offsetC += tensorNumBytes(qODesc);
                }
		void *transOffsetC = (void *)((U8 *)filter + UNI_ALIGN(fw, 16) * UNI_ALIGN(fh, 8));
                CHECK_STATUS(quantize_bias_offsetC(
                    bias, biasDesc, DT_I32, transOffsetC, filterDesc, &scaleO, offsetC));
                bias = nullptr;
                if (outputDesc.dt == DT_U8_Q && outputTensor.get_scale() > 0) {
                    scale[1] = scale[1] / scaleO;
                }
#endif
            } else if (nullptr != bias) {
                if (DT_F16 == outputDesc.dt ||
                    DT_F32 == outputDesc.dt) {  // dequantize and then add bias
                    bias = nullptr;
                } else {
                    CHECK_REQUIREMENT(DT_I8 == outputDesc.dt);
                    biasDesc.dt = DT_I32;
                    I32 *biasI = (I32 *)tmp;
#ifdef _USE_FP16
                    F16 *biasF = (F16 *)bias;
#else
                    F32 *biasF = (F32 *)bias;
#endif
                    for (U32 i = 0; i < tensorNumElements(biasDesc); i++) {
                        biasI[i] = round(scaleO * biasF[i]);
                    }
                    bias = tmp;
                    tmp = (U8 *)tmp + tensorNumBytes(biasDesc);
                }
            }
            outputDesc = qODesc;
            if (outputDesc.dt != outputTensor.get_desc().dt) {
                output = tmp;
                tmp = (U8 *)tmp + tensorNumBytes(outputDesc);
            }
        }
#endif

        if (bias != nullptr) {
            DataType bdt;
            DataFormat bdf;
            U32 bw;
            CHECK_STATUS(tensor1dGet(biasDesc, &bdt, &bdf, &bw));
            if (bw != fw) {
                CHECK_STATUS(NOT_MATCH);
            } else {
                U8 *outArray = (U8 *)output;
                U32 size = tensorNumBytes(biasDesc);
#if defined(_USE_OPENMP)
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
                for (U32 i = 0; i < M; i++) {
                    UNI_MEMCPY(outArray + i * size, bias, size);
                }
            }
        } else {
            UNI_MEMSET(output, 0, tensorNumBytes(outputDesc));
        }

        // If weight is transformed for mmm, don't run as mvm
        if (M == 1 && filterDesc.df != matrix_matrix_multiply_rhs_format(fdt)) {
            TensorDesc vectorDesc = tensor1d(idt, fh);
            TensorDesc resultDesc = tensor1d(odt, fw);
            if (IS_GENERAL(archInfo->arch)) {
                filterDesc.df = DF_NORMAL;
            }
            ret = matrix_vector_multiply(filterDesc, filter, vectorDesc, input, tmpBytes, tmp,
                resultDesc, output, scale, archInfo->arch);
        } else {
            TensorDesc in_desc = tensor2df(idt, DF_NORMAL, M, fh);
            TensorDesc out_desc = tensor2df(outputDesc.dt, DF_NORMAL, M, fw);
            ret = matrix_matrix_multiply(in_desc, input, filterDesc, filter, tmpBytes, tmp,
                out_desc, output, scale, archInfo->arch);
        }

#ifdef _USE_INT8
        if (outputTensor.get_desc().dt != outputDesc.dt) {
            if (DT_I8 == outputTensor.get_desc().dt || DT_U8_Q == outputTensor.get_desc().dt) {
                F32 scales[2] = {-1, -1};  // 0 is outputScale, 1 is computeScale
                scales[0] = outputTensor.get_scale();
                scales[1] = scaleO;
                TensorDesc qDesc = outputTensor.get_desc();
                CHECK_STATUS(quantize_cpu(outputDesc, output, &qDesc,
                    get_ptr_from_tensor(outputTensor, arch), scales, arch));
                outputTensor.set_scale(scales[0]);
            } else {
                Tensor tmpOutput;
                tmpOutput.resize(outputDesc);
                std::shared_ptr<U8> shared_data((U8 *)output, [](U8 *ptr) {});
                ((CpuMemory *)(tmpOutput.get_memory()))->set_shared_ptr(shared_data);
                CHECK_STATUS(dequantize(tmpOutput, &scaleO, biasTensor, outputTensor, archInfo));
            }
        }
#endif
    }
    return ret;
}
