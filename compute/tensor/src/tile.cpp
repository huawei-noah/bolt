// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>
#include "tensor_computing.h"
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE tile_infer_output_size(
    Tensor *inputTensor, TileParamSpec tileParamSpec, Tensor *outputTensor, ArchInfo_t archInfo)
{
    auto inDim = inputTensor->get_desc();
    auto outDim = inDim;

    if ((int)inDim.nDims == tileParamSpec.num_repeats) {
        for (int i = 0; i < tileParamSpec.num_repeats; i++) {
            outDim.dims[tileParamSpec.num_repeats - 1 - i] =
                inDim.dims[tileParamSpec.num_repeats - 1 - i] * tileParamSpec.repeats[i];
        }
    } else {
        int axis = (tileParamSpec.axis >= 0) ? tileParamSpec.axis : tileParamSpec.axis + inDim.nDims;
        axis = inDim.nDims - 1 - axis;
        outDim.dims[axis] = outDim.dims[axis] * tileParamSpec.repeats[0];
    }
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        if (outDim.df == DF_NCHWC4) {
            outDim.df = DF_NCHW;
        }
#endif
    }
    outputTensor->resize(outDim);
    return SUCCESS;
}

EE tile_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        CHECK_STATUS(tile_infer_forward_tmp_bytes_mali(
            inputDesc, outputDesc, gclmemInputDesc, gclmemOutputDesc, bytes));
#endif
    } else {
        *bytes = 0;
    }
    return SUCCESS;
}

EE tile(Tensor inputTensor,
    TileParamSpec tileParamSpec,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc outputDesc = outputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    void *output = get_ptr_from_tensor(outputTensor, arch);
    DataType idt = inputDesc.dt;

    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        void *tmp = get_ptr_from_tensor(tmpTensor, arch);
        CHECK_STATUS(tile_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, tileParamSpec, (GCLMem_t)tmp, outputDesc, (GCLMem_t)output));
#endif
    } else {
        if (inputDesc.df == DF_NCHWC8) {
            inputDesc.dims[inputDesc.nDims - 2] /= 8;
            outputDesc.dims[inputDesc.nDims - 2] /= 8;
            inputDesc.dims[0] *= 8;
            outputDesc.dims[0] *= 8;
        }

        if (tileParamSpec.num_repeats != (int)inputDesc.nDims) {
            CHECK_REQUIREMENT(tileParamSpec.num_repeats == 1);
            int axis = (tileParamSpec.axis >= 0) ? tileParamSpec.axis
                                                 : tileParamSpec.axis + inputDesc.nDims;
            U32 tiles = tileParamSpec.repeats[0];
            for (int i = 0; i < (int)inputDesc.nDims; ++i) {
                tileParamSpec.repeats[i] = 1;
                if (axis == i) {
                    tileParamSpec.repeats[i] = tiles;
                }
            }
        }

        U32 repeat_num = 0;
        for (U32 i = 0; i < inputDesc.nDims; ++i) {
            repeat_num += (tileParamSpec.repeats[inputDesc.nDims - 1 - i] > 1);
        }
        if (repeat_num == 0) {
            UNI_MEMCPY(output, input, tensorNumBytes(inputDesc));
            return SUCCESS;
        }

        U8 *input_ptr = (U8 *)input;
        U8 *output_ptr = (U8 *)output;
        std::vector<U32> otile_size(inputDesc.nDims, outputDesc.dims[0] * bytesOf(idt));
        std::vector<U32> itile_size(inputDesc.nDims, inputDesc.dims[0] * bytesOf(idt));
        for (U32 i = 1; i < inputDesc.nDims; ++i) {
            otile_size[i] = otile_size[i - 1] * outputDesc.dims[i];
            itile_size[i] = itile_size[i - 1] * inputDesc.dims[i];
        }

        bool first_copy = true;
        for (U32 j = 0; j < inputDesc.nDims; ++j) {
            if (tileParamSpec.repeats[inputDesc.nDims - 1 - j] > 1) {
                U32 tiles = tileParamSpec.repeats[inputDesc.nDims - 1 - j];
                int loopOuter = itile_size[inputDesc.nDims - 1] / itile_size[j];
                if (first_copy) {
                    first_copy = false;
                    for (int i = 0; i < loopOuter; ++i) {
                        for (U32 ii = 0; ii < tiles; ++ii) {
                            UNI_MEMCPY(output_ptr + i * tiles * itile_size[j] + ii * itile_size[j],
                                input_ptr + i * itile_size[j], itile_size[j]);
                        }
                    }
                } else {
                    for (int i = loopOuter - 1; i >= 0; --i) {
                        for (U32 ii = 0; ii < tiles; ++ii) {
                            if (i != 0 || ii != 0) {
                                U32 copy_size = otile_size[j - 1] * inputDesc.dims[i];
                                UNI_MEMCPY(output_ptr + i * tiles * copy_size + ii * copy_size,
                                    output_ptr + i * copy_size, copy_size);
                            }
                        }
                    }
                }
            }
        }
    }
    return SUCCESS;
}
