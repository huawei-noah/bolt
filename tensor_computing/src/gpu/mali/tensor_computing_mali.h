// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_COMPUTING_MALI
#define _H_TENSOR_COMPUTING_MALI
#include "tensor_desc.h"
#include "tensor_computing_type.h"
EE tensor_computing_set_input_infer_tmpBuf_size_mali(GCLMem_t   input,
                                                     TensorDesc hostDesc,
                                                     U32*       tmpBufSize);


EE tensor_computing_set_input_mali(GCLHandle_t handle, 
                                   GCLMem_t    input, 
                                   TensorDesc  hostDesc, 
                                   const U8*   hostPtr, 
                                   GCLMem_t    tmpBuf, 
                                   bool        blocking);


EE tensor_computing_get_output_infer_tmpBuf_size_mali(const GCLMem_t input, 
                                                      TensorDesc     hostDesc, 
                                                      U32*           tmpBufSize);


EE tensor_computing_get_output_mali(GCLHandle_t    handle, 
                                    const GCLMem_t input, 
                                    TensorDesc     hostDesc,
                                    U8**           hostPtr,
                                    GCLMem_t       tmpBuf,
                                    bool           blocking);


EE pooling_mali(GCLHandle_t    handle,
                TensorDesc     inputDesc, 
                const GCLMem_t input,
                PoolingDesc    poolingDesc, 
                const void*    scale,
                TensorDesc     outputDesc,
                GCLMem_t       output);


EE pooling_infer_output_size_mali(TensorDesc   inputDesc,
                                  PoolingDesc  poolingDesc,
                                  TensorDesc*  outputDesc,
                                  GCLMemDesc_t gclmemInputDesc,
                                  GCLMemDesc_t gclmemOutputDesc);


EE convolution_infer_output_size_mali(TensorDesc           inputDesc,
                                      TensorDesc           filterDesc,
                                      ConvolutionDesc      convDesc,
                                      TensorDesc*          outputDesc,
                                      GCLMemDesc_t         gclmemInputDesc,
                                      GCLMemDesc_t         gclmemOutputDesc,
                                      ForwardRunInfoMali_t forwardRunInfo);


EE convolution_infer_forward_algorithm_mali(GCLHandle_t          handle,
                                            TensorDesc           inputDesc, 
                                            TensorDesc           filterDesc, 
                                            ConvolutionDesc      convDesc,
                                            TensorDesc           outputDesc,
                                            ConvolutionPolicy    policy, 
                                            ActivationMode       activationMode,
                                            ForwardRunInfoMali_t forwardRunInfo);


EE convolution_transform_filter_bytes_mali(TensorDesc            filterDesc, 
                                           ForwardRunInfoMali_t  forwardRunInfo,
                                           GCLMemDesc_t          gclmemFilterDesc,
                                           U32*                  bytes);


EE convolution_transform_filter_mali(GCLHandle_t          handle,
                                     TensorDesc           filterDesc,
                                     GCLMem_t             filter,
                                     ForwardRunInfoMali_t forwardRunInfo,
                                     TensorDesc*          fltmemDesc,
                                     GCLMem_t             fltmem);


EE convolution_infer_forward_tmp_bytes_mali(TensorDesc            inputDesc, 
                                            TensorDesc            filterDesc, 
                                            TensorDesc            outputDesc,
                                            ConvolutionDesc       convDesc, 
                                            ForwardRunInfoMali_t  forwardRunInfo,
                                            U32*                  bytes);


EE convolution_mali(GCLHandle_t          handle,
                    TensorDesc           inputDesc, 
                    const GCLMem_t       input,
                    TensorDesc           filterDesc, 
                    const GCLMem_t       filter,
                    ConvolutionDesc      convDesc,
                    ForwardRunInfoMali_t forwardRunInfo,
                    TensorDesc           scaleDesc, 
                    const GCLMem_t       scale,
                    TensorDesc           biasDesc, 
                    const GCLMem_t       bias,
                    U32                  tmpBytes, 
                    GCLMem_t             tmpBuf,
                    TensorDesc           outputDesc, 
                    GCLMem_t             output,
                    ActivationMode       activationMode);


EE depthwise_convolution_infer_output_size_mali(TensorDesc           inputDesc,
                                                TensorDesc           filterDesc,
                                                ConvolutionDesc      convDesc,
                                                TensorDesc*          outputDesc,
                                                GCLMemDesc_t         gclmemInputDesc,
                                                GCLMemDesc_t         gclmemOutputDesc,
                                                ForwardRunInfoMali_t forwardRunInfo);

EE depthwise_convolution_infer_forward_algorithm_mali(GCLHandle_t          handle,
                                                      TensorDesc           inputDesc, 
                                                      TensorDesc           filterDesc, 
                                                      TensorDesc           outputDesc,
                                                      ConvolutionDesc      convDesc,
                                                      ConvolutionPolicy    policy, 
                                                      ActivationMode       depthwisewiseActivationMode,
                                                      ActivationMode       pointwiseActivationMode,
                                                      ForwardRunInfoMali_t forwardRunInfo);

EE depthwise_convolution_transform_filter_bytes_mali(TensorDesc            filterDesc, 
                                                     ForwardRunInfoMali_t  forwardRunInfo,
                                                     GCLMemDesc_t          gclmemFilterDesc,
                                                     U32*                  bytes);

EE depthwise_convolution_transform_filter_mali(GCLHandle_t          handle,
                                               TensorDesc           filterDesc,
                                               GCLMem_t             filter,
                                               ForwardRunInfoMali_t forwardRunInfo,
                                               TensorDesc*          fltmemDesc,
                                               GCLMem_t             fltmem);

EE depthwise_convolution_infer_forward_tmp_bytes_mali(TensorDesc            inputDesc, 
                                                      TensorDesc            filterDesc, 
                                                      TensorDesc            outputDesc,
                                                      ConvolutionDesc       convDesc, 
                                                      ForwardRunInfoMali_t  forwardRunInfo,
                                                      U32*                  bytes);

EE depthwise_convolution_mali(GCLHandle_t          handle,
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
                              ActivationMode       depthwisewiseActivationMode,
                              ActivationMode       pointwiseActivationMode);

EE bilateral_slice_apply_infer_output_size_mali(TensorDesc              inputDesc,
                                                TensorDesc              guideDesc,
                                                TensorDesc              gridDesc,
                                                BilateralSliceApplyDesc bilateralSliceApplyDesc,
                                                TensorDesc*             outputDesc,
                                                GCLMemDesc_t            gclmemInputDesc,
                                                GCLMemDesc_t            gclmemGuideDesc,
                                                GCLMemDesc_t            gclmemGridDesc,
                                                GCLMemDesc_t            gclmemOutputDesc);

EE bilateral_slice_apply_infer_forward_tmp_bytes_mali(TensorDesc              inputDesc,
                                                      TensorDesc              guideDesc,
                                                      TensorDesc              gridDesc,
                                                      BilateralSliceApplyDesc bilateralSliceApplyDesc,
                                                      ForwardRunInfoMali_t    forwardRunInfo,
                                                      U32*                    bytes);

EE bilateral_slice_apply_mali(GCLHandle_t             handle,
                              TensorDesc              inputDesc,
                              const GCLMem_t          input,
                              TensorDesc              guideDesc,
                              const GCLMem_t          guide,
                              TensorDesc              gridDesc,
                              const GCLMem_t          grid,
                              BilateralSliceApplyDesc bilateralSliceApplyDesc,
                              ForwardRunInfoMali_t    forwardRunInfo,
                              U32                     tmpBytes,
                              GCLMem_t                tmpBuf,
                              TensorDesc              outputDesc,
                              GCLMem_t                output);


EE eltwise_infer_output_size_mali(std::vector<TensorDesc> inputDesc,
                                  TensorDesc*             outputDesc,
                                  GCLMemDesc_t            gclmemInputDesc,
                                  GCLMemDesc_t            gclmemOutputDesc);

EE eltwise_mali(GCLHandle_t             handle,
                std::vector<TensorDesc> inputDesc,
                std::vector<void*>      input,
                TensorDesc              outputDesc,
                GCLMem_t                output,
                EltwiseMode             eltwiseMode);

EE softmax_infer_output_size_mali(TensorDesc   inputDesc,
                                  TensorDesc*  outputDesc,
                                  GCLMemDesc_t gclmemInputDesc,
                                  GCLMemDesc_t gclmemOutputDesc);

EE softmax_mali(GCLHandle_t handle, 
                TensorDesc  inputDesc,
                GCLMem_t    input,
                TensorDesc  outputDesc,
                GCLMem_t    output);

EE activation_infer_output_size_mali(TensorDesc   inputDesc,
                                     TensorDesc*  outputDesc,
                                     GCLMemDesc_t gclmemInputDesc,
                                     GCLMemDesc_t gclmemOutputDesc);

EE activation_mali(GCLHandle_t    handle, 
                   TensorDesc     inputDesc,
                   GCLMem_t       input,
                   ActivationMode activationMode);

EE fully_connected_infer_output_size_mali(TensorDesc   inputDesc,
                                          TensorDesc   filterDesc,
                                          TensorDesc*  outputDesc,
                                          GCLMemDesc_t gclmemInputDesc,
                                          GCLMemDesc_t gclmemOutputDesc);

EE fully_connected_transform_filter_bytes_mali(TensorDesc   filterDesc, 
                                               GCLMemDesc_t gclmemFilterDesc,
                                               U32*         bytes);

EE fully_connected_transform_filter_mali(GCLHandle_t handle,
                                         TensorDesc  filterDesc,
                                         GCLMem_t    filter,
                                         TensorDesc* fltmemDesc,
                                         GCLMem_t    fltmem);

EE fully_connected_infer_forward_tmp_bytes_mali(TensorDesc inputDesc, 
                                                TensorDesc filterDesc, 
                                                U32*       bytes);

EE fully_connected_mali(GCLHandle_t    handle,
                        TensorDesc     inputDesc, 
                        const GCLMem_t input,
                        TensorDesc     filterDesc, 
                        const GCLMem_t filter,
                        TensorDesc     biasDesc, 
                        const GCLMem_t bias,
                        U32            tmpBytes, 
                        GCLMem_t       tmpBuf,
                        TensorDesc     outputDesc, 
                        GCLMem_t       output);


EE scale_infer_output_size_mali(TensorDesc   inputDesc,
                                TensorDesc*  outputDesc,
                                GCLMemDesc_t gclmemInputDesc,
                                GCLMemDesc_t gclmemOutputDesc);
                                         
EE scale_mali(GCLHandle_t handle,
              GCLMem_t    alpha,
              GCLMem_t    beta,
              TensorDesc  inputDesc,
              GCLMem_t    input,
              TensorDesc  outputDesc,
              GCLMem_t    output);

EE concat_infer_output_size_mali(std::vector<TensorDesc> inputDesc,
                                 TensorDesc*            outputDesc,
                                 U32                    concatDim,
                                 GCLMemDesc_t           gclmemInputDesc,
                                 GCLMemDesc_t           gclmemOutputDesc);

EE concat_mali(GCLHandle_t             handle,
               std::vector<TensorDesc> inputDesc,
               std::vector<void*>      input,
               GCLMem_t                inputScale,
               TensorDesc              outputDesc,
               GCLMem_t                output,
               GCLMem_t                outputScale,
               U32                     concatDim);
#endif


