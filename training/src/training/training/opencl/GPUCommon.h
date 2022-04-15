// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <training/common/Common.h>
#include <training/common/Types.h>
#include <training/opencl/OpenCLKernelManager.h>

namespace raul
{
namespace gpu
{

void batchnorm_forward(OpenCLKernelManager& manager,
                       const Name& caller,
                       raul::Dimension dim,
                       size_t batchSize,
                       size_t depth,
                       size_t height,
                       size_t width,
                       dtype momentum,
                       dtype eps,
                       bool useMomentum,
                       const cl::Buffer& input,
                       const cl::Buffer& beta,
                       const cl::Buffer& gamma,
                       cl::Buffer& mean,
                       cl::Buffer& var,
                       cl::Buffer& xHat,
                       cl::Buffer& varSqrt,
                       cl::Buffer& output,
                       cl::Buffer& meanEval,
                       cl::Buffer& varEval);

void batchnorm_test(OpenCLKernelManager& manager,
                    const Name& caller,
                    raul::Dimension dim,
                    size_t batchSize,
                    size_t depth,
                    size_t height,
                    size_t width,
                    dtype eps,
                    const cl::Buffer& input,
                    const cl::Buffer& beta,
                    const cl::Buffer& gamma,
                    const cl::Buffer& meanEval,
                    const cl::Buffer& varEval,
                    cl::Buffer& output);

void batchnorm_backward(OpenCLKernelManager& manager,
                        const Name& caller,
                        raul::Dimension dim,
                        size_t batchSize,
                        size_t depth,
                        size_t height,
                        size_t width,
                        const cl::Buffer& deltas,
                        const cl::Buffer& xHat,
                        const cl::Buffer& varSqrt,
                        const cl::Buffer& gamma,
                        cl::Buffer& prevDeltas);

void batchnorm_backward(OpenCLKernelManager& manager,
                        const Name& caller,
                        raul::Dimension dim,
                        size_t batchSize,
                        size_t depth,
                        size_t height,
                        size_t width,
                        const cl::Buffer& deltas,
                        const cl::Buffer& xHat,
                        const cl::Buffer& varSqrt,
                        const cl::Buffer& gamma,
                        cl::Buffer& prevDeltas,
                        cl::Buffer& nablaBeta,
                        cl::Buffer& nablaGamma);

void clampForward(OpenCLKernelManager& manager,
                  const Name& caller,
                  size_t batch,
                  size_t depth,
                  size_t height,
                  size_t width,
                  const dtype min,
                  const dtype max,
                  const cl::Buffer& in,
                  const cl::Buffer& out);
void clampBackward(OpenCLKernelManager& manager,
                   const Name& caller,
                   size_t batch,
                   size_t depth,
                   size_t height,
                   size_t width,
                   const dtype min,
                   const dtype max,
                   const cl::Buffer& in,
                   const cl::Buffer& deltas,
                   const cl::Buffer& prevLayerDelta);

void copy(OpenCLKernelManager& manager,
          const Name& caller,
          size_t sourceLen,
          size_t destinationLen,
          size_t sourceOffset,
          size_t destinationOffset,
          bool sumWithOldValues,
          const cl::Buffer& in,
          cl::Buffer& out);

void cumsum(OpenCLKernelManager& manager,
            const Name& caller,
            size_t x,
            size_t y,
            size_t z,
            size_t dimension,
            size_t size,
            const shape& outputStrides,
            const cl::Buffer& in,
            cl::Buffer& out,
            bool isForward = true);

void eltwiseSumOp(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, size_t inputNum, const std::array<cl::Buffer, 4>& in, cl::Buffer& out);
void eltwiseSumOp(OpenCLKernelManager& manager,
                  const Name& caller,
                  size_t oBatch,
                  size_t oDepth,
                  size_t oHeight,
                  size_t oWidth,
                  size_t inputNum,
                  const std::array<size_t, 4> iBatches,
                  const std::array<size_t, 4> iDepths,
                  const std::array<size_t, 4> iHeights,
                  const std::array<size_t, 4> iWidths,
                  const std::array<cl::Buffer, 4>& in,
                  cl::Buffer& out);
void eltwiseSubOp(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in0, const cl::Buffer& in1, cl::Buffer& out);
void eltwiseSubOp(OpenCLKernelManager& manager,
                  const Name& caller,
                  size_t oBatch,
                  size_t oDepth,
                  size_t oHeight,
                  size_t oWidth,
                  size_t iBatch0,
                  size_t iDepth0,
                  size_t iHeight0,
                  size_t iWidth0,
                  const cl::Buffer& in0,
                  size_t iBatch1,
                  size_t iDepth1,
                  size_t iHeight1,
                  size_t iWidth1,
                  const cl::Buffer& in1,
                  cl::Buffer& out);
void eltwiseMulOp(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, size_t inputNum, const std::array<cl::Buffer, 4>& in, cl::Buffer& out);
void eltwiseMulOp(OpenCLKernelManager& manager,
                  const Name& caller,
                  size_t oBatch,
                  size_t oDepth,
                  size_t oHeight,
                  size_t oWidth,
                  size_t inputNum,
                  const std::array<size_t, 4> iBatches,
                  const std::array<size_t, 4> iDepths,
                  const std::array<size_t, 4> iHeights,
                  const std::array<size_t, 4> iWidths,
                  const std::array<cl::Buffer, 4>& in,
                  cl::Buffer& out);
void eltwiseDivOp(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in0, const cl::Buffer& in1, cl::Buffer& out);
void eltwiseDivOp(OpenCLKernelManager& manager,
                  const Name& caller,
                  size_t oBatch,
                  size_t oDepth,
                  size_t oHeight,
                  size_t oWidth,
                  size_t iBatch0,
                  size_t iDepth0,
                  size_t iHeight0,
                  size_t iWidth0,
                  const cl::Buffer& in0,
                  size_t iBatch1,
                  size_t iDepth1,
                  size_t iHeight1,
                  size_t iWidth1,
                  const cl::Buffer& in1,
                  cl::Buffer& out);

void slice(OpenCLKernelManager& manager,
           const Name& caller,
           size_t axisNum,
           size_t iDepth,
           size_t iHeight,
           size_t iWidth,
           size_t axisMax,
           size_t numOfSlices,
           size_t inSize,
           size_t batchOff,
           size_t totalSize,
           const std::array<size_t, 4>& oWidth,
           const std::array<size_t, 4>& oHeight,
           const std::array<size_t, 4>& oOffset,
           const std::array<size_t, 4>& axisLen,
           const cl::Buffer& in,
           std::array<cl::Buffer, 4>& out);

void concat(OpenCLKernelManager& manager,
            const Name& caller,
            size_t axisNum,
            size_t oDepth,
            size_t oHeight,
            size_t oWidth,
            size_t axisMax,
            size_t numOfSlices,
            size_t outSize,
            size_t batchOff,
            const std::array<size_t, 4>& iWidth,
            const std::array<size_t, 4>& iHeight,
            const std::array<size_t, 4>& iOffset,
            const std::array<size_t, 4>& axisLen,
            const std::array<cl::Buffer, 4>& in,
            cl::Buffer& out);

void eltwiseMaxForwardOp(OpenCLKernelManager& manager,
                         const Name& caller,
                         size_t batch,
                         size_t depth,
                         size_t height,
                         size_t width,
                         size_t index,
                         const cl::Buffer& in,
                         cl::Buffer& indexes,
                         cl::Buffer& out);
void eltwiseMaxBackwardOp(OpenCLKernelManager& manager,
                          const Name& caller,
                          size_t batch,
                          size_t depth,
                          size_t height,
                          size_t width,
                          size_t index,
                          const cl::Buffer& deltas,
                          cl::Buffer& indexes,
                          cl::Buffer& prevDelta);
void eltwiseMinForwardOp(OpenCLKernelManager& manager,
                         const Name& caller,
                         size_t batch,
                         size_t depth,
                         size_t height,
                         size_t width,
                         size_t index,
                         const cl::Buffer& in,
                         cl::Buffer& indexes,
                         cl::Buffer& out);
void eltwiseMinBackwardOp(OpenCLKernelManager& manager,
                          const Name& caller,
                          size_t batch,
                          size_t depth,
                          size_t height,
                          size_t width,
                          size_t index,
                          const cl::Buffer& deltas,
                          cl::Buffer& indexes,
                          cl::Buffer& prevDelta);

void gaussianUpsamplingDistributionForward(OpenCLKernelManager& manager,
                                           const Name& caller,
                                           size_t batch,
                                           size_t depth,
                                           size_t height,
                                           size_t width,
                                           const cl::Buffer& values,
                                           const cl::Buffer& loc,
                                           const cl::Buffer& scale,
                                           cl::Buffer& out);
void gaussianUpsamplingDistributionBackward(OpenCLKernelManager& manager,
                                            const Name& caller,
                                            size_t batch,
                                            size_t depth,
                                            size_t height,
                                            size_t width,
                                            bool backwardForLoc,
                                            const cl::Buffer& values,
                                            const cl::Buffer& loc,
                                            const cl::Buffer& scale,
                                            const cl::Buffer& deltas,
                                            cl::Buffer& prevLayerDelta);

void initAlignment(OpenCLKernelManager& manager, const Name& caller, dtype val, size_t batch, size_t height, cl::Buffer& out);

void iota(OpenCLKernelManager& manager, const Name& caller, dtype startPoint, size_t size, cl::Buffer& out);

void nonZeroMask(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);

void reduceTargets(OpenCLKernelManager& manager,
                   const Name& caller,
                   size_t batch,
                   size_t idepth,
                   size_t odepth,
                   size_t iheight,
                   size_t oheight,
                   size_t width,
                   size_t reductionFactor,
                   const cl::Buffer& in,
                   cl::Buffer& out);

void ReLU(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void ReLU6(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void ReLUBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& delta, cl::Buffer& prevDelta);
void ReLU6Backward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& delta, cl::Buffer& prevDelta);

void reverse(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void reverse(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& length, cl::Buffer& out);

void round(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);

void selectForward(OpenCLKernelManager& manager,
                   const Name& caller,
                   size_t batch,
                   size_t depth,
                   size_t height,
                   size_t width,
                   const cl::Buffer& cond,
                   const cl::Buffer& in0,
                   const cl::Buffer& in1,
                   cl::Buffer& out);

/**
 * @param index - 0 for first argument, 1 for second
 */
void selectBackward(OpenCLKernelManager& manager,
                    const Name& caller,
                    size_t index,
                    size_t batch,
                    size_t depth,
                    size_t height,
                    size_t width,
                    const cl::Buffer& cond,
                    const cl::Buffer& deltas,
                    cl::Buffer& prevDelta);

void sequenceMask(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& lengths, cl::Buffer& mask);

// ijk -> kji
// in has size dim3 x dim2 x dim1
// out has size dim1 x dim2 x dim3
void swapIndices3d(OpenCLKernelManager& manager, const Name& caller, size_t dim1, size_t dim2, size_t dim3, const cl::Buffer& in, cl::Buffer& out);

void tile(OpenCLKernelManager& manager,
          const Name& caller,
          size_t iDepth,
          size_t iHeight,
          size_t iWidth,
          size_t oDepth,
          size_t oHeight,
          size_t oWidth,
          size_t inOff,
          size_t outOff,
          bool isForward,
          const cl::Buffer& in,
          cl::Buffer& out);

void transpose(OpenCLKernelManager& manager,
               const Name& caller,
               size_t batch,
               size_t iDepth,
               size_t iHeight,
               size_t iWidth,
               size_t oDepth,
               size_t oHeight,
               size_t oWidth,
               size_t dim0,
               size_t dim1,
               size_t dim2,
               size_t dim3,
               const cl::Buffer& in,
               cl::Buffer& out);

void zeroOutput(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& length, cl::Buffer& out);

void globalL2SquareNorm(OpenCLKernelManager& manager, const Name& caller, size_t inputSize, const cl::Buffer& in, cl::Buffer& out);

void clipGradients(OpenCLKernelManager& manager, const Name& caller, size_t inputSize, dtype clipNorm, const cl::Buffer& currGlobalNorm, cl::Buffer& grad);

void adam(OpenCLKernelManager& manager,
          const Name& caller,
          size_t size,
          dtype alpha,
          dtype beta1,
          dtype beta2,
          dtype epsilon,
          size_t useSimpleEpsilon,
          const cl::Buffer& grad,
          const cl::Buffer& betaT1,
          const cl::Buffer& betaT2,
          cl::Buffer& m,
          cl::Buffer& v,
          cl::Buffer& param);

void dynamicDepthwiseConv2DForward(OpenCLKernelManager& manager,
                                   const Name& caller,
                                   size_t batchSize,
                                   size_t inputC,
                                   size_t outputH,
                                   size_t outputW,
                                   size_t channelMultiplier,
                                   size_t filterH,
                                   size_t filterW,
                                   const cl::Buffer& in0,
                                   const cl::Buffer& in1,
                                   cl::Buffer& out);

void dynamicDepthwiseConv2DBackward(OpenCLKernelManager& manager,
                                    const Name& caller,
                                    size_t batchSize,
                                    size_t inputC,
                                    size_t outputH,
                                    size_t outputW,
                                    size_t channelMultiplier,
                                    size_t filterH,
                                    size_t filterW,
                                    bool isForInput,
                                    const cl::Buffer& in0,
                                    const cl::Buffer& in1,
                                    cl::Buffer& out);

void tile(OpenCLKernelManager& manager,
          const Name& caller,
          size_t iDepth,
          size_t iHeight,
          size_t iWidth,
          size_t oDepth,
          size_t oHeight,
          size_t oWidth,
          size_t inOff,
          size_t outOff,
          bool isForward,
          const cl::Buffer& in,
          cl::Buffer& out);

void reduceDimForward(OpenCLKernelManager& manager,
                      const Name& caller,
                      size_t batchSize,
                      size_t iDepth,
                      size_t iHeight,
                      size_t iWidth,
                      size_t oDepth,
                      size_t oHeight,
                      size_t oWidth,
                      size_t dimension,
                      const std::string& reductionType,
                      const cl::Buffer& in,
                      cl::Buffer& out);

void reduceDefaultForward(OpenCLKernelManager& manager, const Name& caller, size_t inputSize, dtype divisor, size_t countNonZeroElems, const cl::Buffer& in, cl::Buffer& out);

void reduceDefaultBackward(OpenCLKernelManager& manager,
                           const Name& caller,
                           size_t batchSize,
                           size_t oDepth,
                           size_t oHeight,
                           size_t oWidth,
                           dtype divisor,
                           const cl::Buffer& deltas,
                           cl::Buffer& prevDelta);

void reduceDimBackward(OpenCLKernelManager& manager,
                       const Name& caller,
                       size_t batchSize,
                       size_t iDepth,
                       size_t iHeight,
                       size_t iWidth,
                       size_t oDepth,
                       size_t oHeight,
                       size_t oWidth,
                       size_t dimension,
                       dtype divisor,
                       const cl::Buffer& deltas,
                       cl::Buffer& prevDelta);

void expForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void expBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void sqrtForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void sqrtBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void rsqrtForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void rsqrtBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void squareForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void squareBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void logForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void logBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void addBias(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, dtype bias, const cl::Buffer& in, cl::Buffer& out);

void sigmoidForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void sigmoidBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void softplusForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, dtype beta, dtype threshold, const cl::Buffer& in, cl::Buffer& out);
void softplusBackward(OpenCLKernelManager& manager,
                      const Name& caller,
                      size_t batch,
                      size_t depth,
                      size_t height,
                      size_t width,
                      dtype beta,
                      dtype threshold,
                      const cl::Buffer& out,
                      const cl::Buffer& deltas,
                      cl::Buffer& prevDelta);

void tanhForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void tanhBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void swishForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void swishBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void splitterForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void splitterBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void softmaxForward(OpenCLKernelManager& manager, const Name& caller, size_t externalDimSize, size_t internalDimSize, const cl::Buffer& in, cl::Buffer& out);
void softmaxBackward(OpenCLKernelManager& manager, const Name& caller, size_t externalDimSize, size_t internalDimSize, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void geluErfForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void geluErfBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void geluTanhForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void geluTanhBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void dropoutForward(OpenCLKernelManager& manager,
                    const Name& caller,
                    size_t batch,
                    size_t depth,
                    size_t height,
                    size_t width,
                    dtype scale,
                    const cl::Buffer& in,
                    const cl::Buffer& proba,
                    cl::Buffer& out);
void dropoutBackward(OpenCLKernelManager& manager,
                     const Name& caller,
                     size_t batch,
                     size_t depth,
                     size_t height,
                     size_t width,
                     dtype scale,
                     const cl::Buffer& deltas,
                     const cl::Buffer& proba,
                     cl::Buffer& prevDelta);

void hsigmoidForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void hsigmoidBackward(OpenCLKernelManager& manager, const Name& caller, size_t size, dtype leftDivisor, dtype rightDivisor, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta);

void hswishForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out);
void hswishBackward(OpenCLKernelManager& manager, const Name& caller, size_t size, dtype a, dtype b, dtype c, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta);
} // raul::gpu namespace
} // raul namespace

#endif // GEMM_GPU_H
