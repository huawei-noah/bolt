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
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

// [1, 10, 10] + [1, 10, 10] = [1, 10, 10]
// [1, 10, 1] + [1, 1, 10] = [1, 10, 10]
// [1, 20, 10] + [10] = [1, 20, 10]
inline EE eltwise_infer_output_size_cpu(std::vector<TensorDesc> inputDesc, TensorDesc* outputDesc)
{
    if (nullptr == outputDesc)
        CHECK_STATUS(NULL_POINTER);
    U32 num = inputDesc.size();
    if (num <= 0)
        return NOT_MATCH;

    U32 arrayDimMax = 0;
    for (U32 i = 1; i < num; i++) {
        if (inputDesc[i].nDims > inputDesc[arrayDimMax].nDims)
            arrayDimMax = i;
    }

    U32 dim = inputDesc[arrayDimMax].nDims;
    *outputDesc = inputDesc[arrayDimMax];

    // DF should either all be NCHWC8, or all be non-C8
    bool isC8 = DF_NCHWC8 == (*outputDesc).df;
    for (U32 i = 0; i < num; i++) {
        if (isC8) {
            CHECK_REQUIREMENT(DF_NCHWC8 == inputDesc[i].df);
        } else {
            CHECK_REQUIREMENT(DF_NCHWC8 != inputDesc[i].df);
        }
    }
    for (U32 i = 0; i < dim; i++) {
        for (U32 j = 0; j < num; j++) {
            if (inputDesc[j].nDims > i) {
                outputDesc->dims[i] = UNI_MAX(outputDesc->dims[i], inputDesc[j].dims[i]);
            }
        }
    }
    return SUCCESS;
}

EE eltwise_infer_output_size(std::vector<TensorDesc> inputDesc, TensorDesc* outputDesc, Arch arch, ExtInfo_t extInfo) {
    EE ret = NOT_SUPPORTED;
    if(arch == MALI){
#ifdef _USE_MALI
        ret = eltwise_infer_output_size_mali(inputDesc, outputDesc, extInfo->maliInfo.gclmemInputDesc, extInfo->maliInfo.gclmemOutputDesc);
#endif
    } else {
        ret = eltwise_infer_output_size_cpu(inputDesc, outputDesc);
    }    
    return ret;
}

EE eltwise(std::vector<TensorDesc> inputDesc, std::vector<void*> input,
    TensorDesc outputDesc, void* output, EltwiseMode eltwiseMode, Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = eltwise_general(inputDesc, input, outputDesc, output, eltwiseMode);
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = eltwise_arm(inputDesc, input, outputDesc, output, eltwiseMode);
#endif
#ifdef _USE_MALI
    } else if (arch == MALI) {
        ret = eltwise_mali(extInfo->maliInfo.handle, inputDesc, input, outputDesc, (GCLMem_t)output, eltwiseMode);
#endif
    }
    return ret;
}

