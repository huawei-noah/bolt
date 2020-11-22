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

EE tile_infer_output_size(
    Tensor *inputTensor, TileParamSpec tileParamSpec, Tensor *outputTensor, ArchInfo_t archInfo)
{
    auto inDim = inputTensor->get_desc();
    auto outDim = inDim;
    if ((int)inDim.nDims == tileParamSpec.dimsSize) {
        for (int i = 0; i < tileParamSpec.dimsSize; i++) {
            outDim.dims[tileParamSpec.dimsSize - 1 - i] =
                inDim.dims[tileParamSpec.dimsSize - 1 - i] * tileParamSpec.repeatsInfo[i];
        }
    } else {
        if (tileParamSpec.axis == -1) {
            tileParamSpec.axis = 0;
        }
        outDim.dims[tileParamSpec.axis] =
            outDim.dims[tileParamSpec.axis] * tileParamSpec.repeatsInfo[0];
    }
    outputTensor->resize(outDim);
    return SUCCESS;
}

EE tile(Tensor inputTensor, TileParamSpec tileParamSpec, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    U8 *ptr = (U8 *)output;
    int element_size = bytesOf(inputDesc.dt);
    if (tileParamSpec.dimsSize == (int)inputDesc.nDims) {  //onnx model support
        ret = NOT_SUPPORTED;
    } else {  //caffe model support
        int axis = tileParamSpec.axis;
        if (axis == -1) {
            axis = 0;
        }
        int length = 1;
        for (U32 i = 0; i < inputDesc.nDims; i++) {
            length = length * inputDesc.dims[i];
        }
        if (axis == (int)inputDesc.nDims - 1) {
            for (int i = 0; i < tileParamSpec.repeatsInfo[0]; i++) {
                U8 *srcPtr = (U8 *)input;
                U8 *desPtr = ptr + element_size * length * i;
                memcpy(desPtr, srcPtr, element_size * length);
            }
            ret = SUCCESS;
        } else if (axis == 0) {
            int count = length / inputDesc.dims[axis];
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < tileParamSpec.repeatsInfo[0]; j++) {
                    U8 *srcPtr = (U8 *)input + element_size * inputDesc.dims[axis] * i;
                    U8 *desPtr = ptr +
                        element_size * inputDesc.dims[axis] * (tileParamSpec.repeatsInfo[0] * i + j);
                    memcpy(desPtr, srcPtr, element_size * inputDesc.dims[axis]);
                }
            }
            ret = SUCCESS;
        } else {
            ret = NOT_SUPPORTED;
        }
    }
    return ret;
}
