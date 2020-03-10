// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CONVOLUTION_DIRECT_MALI_FP16
#define _H_CONVOLUTION_DIRECT_MALI_FP16
#include "sys.h"
#include "tensor_desc.h"
#include "type.h"
#include "error.h"
#include "tensor_computing_type.h"

EE convolution_direct_infer_forward_algorithm_mali_fp16(GCLHandle_t          handle,
                                                        TensorDesc           inputDesc, 
                                                        TensorDesc           filterDesc, 
                                                        ConvolutionDesc      convDesc,
                                                        TensorDesc           outputDesc,
                                                        ConvolutionPolicy    policy, 
                                                        ActivationMode       activationMode,
                                                        ForwardRunInfoMali_t forwardRunInfo);

EE convolution_direct_transform_filter_bytes_mali_fp16(TensorDesc            filterDesc, 
                                                       ForwardRunInfoMali_t  forwardRunInfo,
                                                       GCLMemDesc_t          gclmemFilterDesc,
                                                       U32*                  bytes);

EE convolution_direct_transform_filter_mali_fp16(GCLHandle_t          handle,
                                                 TensorDesc           filterDesc,
                                                 GCLMem_t             filter,
                                                 ForwardRunInfoMali_t forwardRunInfo,
                                                 TensorDesc*          fltmemDesc,
                                                 GCLMem_t             fltmem);

EE convolution_direct_infer_forward_tmp_bytes_mali_fp16(TensorDesc            inputDesc, 
                                                        TensorDesc            filterDesc, 
                                                        TensorDesc            outputDesc,
                                                        ConvolutionDesc       convDesc, 
                                                        ForwardRunInfoMali_t  forwardRunInfo,
                                                        U32*                  bytes); 

EE convolution_direct_mali_fp16(GCLHandle_t          handle,
                                TensorDesc           inputDesc, 
                                const GCLMem_t       input,
                                TensorDesc           filterDesc, 
                                const GCLMem_t       filter,
                                ConvolutionDesc      convDesc,
                                ForwardRunInfoMali_t forwardRunInfo,
                                TensorDesc           biasDesc, 
                                const GCLMem_t       bias,
                                U32                  tmpBytes, 
                                GCLMem_t             tmpBuf,
                                TensorDesc           outputDesc, 
                                GCLMem_t             output,
                                ActivationMode       activationMode);

#endif
