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
#ifdef _USE_MALI 
#include "gpu/mali/tensor_computing_mali.h"
#endif

inline EE bilateral_slice_apply_infer_output_size_cpu()
{
    return NOT_SUPPORTED;
}

EE bilateral_slice_apply_infer_output_size(TensorDesc inputDesc, TensorDesc guideDesc, TensorDesc gridDesc, BilateralSliceApplyDesc bilateralSliceApplyDesc,
    TensorDesc* outputDesc, Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if(arch == MALI) {
#ifdef _USE_MALI
        if(extInfo->maliInfo.gclmemInputDesc) {
            ret = bilateral_slice_apply_infer_output_size_mali(inputDesc, guideDesc, gridDesc, bilateralSliceApplyDesc, outputDesc,
                &extInfo->maliInfo.gclmemInputDesc[0], &extInfo->maliInfo.gclmemInputDesc[2], &extInfo->maliInfo.gclmemInputDesc[1], extInfo->maliInfo.gclmemOutputDesc);
        } else {
            ret = bilateral_slice_apply_infer_output_size_mali(inputDesc, guideDesc, gridDesc, bilateralSliceApplyDesc, outputDesc, 
                NULL, NULL, NULL, extInfo->maliInfo.gclmemOutputDesc);
        }
#endif
    }
    return ret;
}

EE bilateral_slice_apply_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc guideDesc, TensorDesc gridDesc, BilateralSliceApplyDesc bilateralSliceApplyDesc,
    U32* bytes, Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if(arch == MALI) {
#ifdef _USE_MALI
        ret = bilateral_slice_apply_infer_forward_tmp_bytes_mali(inputDesc, guideDesc, gridDesc, bilateralSliceApplyDesc, 
            extInfo->maliInfo.forwardRunInfo, bytes);
#endif
    }
    return ret;
}

EE bilateral_slice_apply(TensorDesc inputDesc, const void* input, TensorDesc guideDesc, const void* guide, 
                         TensorDesc gridDesc, const void* grid, BilateralSliceApplyDesc bilateralSliceApplyDesc,
                         U32 tmpBytes, const void* tmpBuf, TensorDesc outputDesc, 
                         const void* output, Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if(arch == MALI){
#ifdef _USE_MALI
        ret = bilateral_slice_apply_mali(extInfo->maliInfo.handle, inputDesc, (GCLMem_t)input, guideDesc, (GCLMem_t)guide, gridDesc, (GCLMem_t)grid, 
            bilateralSliceApplyDesc, extInfo->maliInfo.forwardRunInfo, tmpBytes, (GCLMem_t)tmpBuf, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}
