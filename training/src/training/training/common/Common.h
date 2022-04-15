// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <optional>
#include <yato/array_view.h>

//#define CHECK_ASSERT

#include "Errors.h"
#include "Name.h"
#include "OpenCLInclude.h"
#include "TypeHalf.h"
#include "Types.h"

#include <map>
#include <utility>

#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
extern "C"
{
#include <cblas.h>
}
#else
#ifndef OPENBLAS_CONST
#define OPENBLAS_CONST const
#endif

typedef enum CBLAS_TRANSPOSE
{
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113,
    CblasConjNoTrans = 114
} CBLAS_TRANSPOSE;

typedef enum CBLAS_UPLO
{
    CblasUpper = 121,
    CblasLower = 122
} CBLAS_UPLO;
#endif

#ifdef CHECK_ASSERT
#define CHECK_NEAR ASSERT_NEAR
#else
#define CHECK_NEAR EXPECT_NEAR
#endif

#define RAUL_E 2.71828182845904523536        // e
#define RAUL_LOG2E 1.44269504088896340736    // log2(e)
#define RAUL_LOG10E 0.434294481903251827651  // log10(e)
#define RAUL_LN2 0.693147180559945309417     // ln(2)
#define RAUL_LN10 2.30258509299404568402     // ln(10)
#define RAUL_PI 3.14159265358979323846       // pi
#define RAUL_PI_2 1.57079632679489661923     // pi/2
#define RAUL_PI_4 0.785398163397448309616    // pi/4
#define RAUL_1_PI 0.318309886183790671538    // 1/pi
#define RAUL_2_PI 0.636619772367581343076    // 2/pi
#define RAUL_2_SQRTPI 1.12837916709551257390 // 2/sqrt(pi)
#define RAUL_SQRT2_PI 0.79788456080286535588 // sqrt(2/pi)
#define RAUL_SQRT2 1.41421356237309504880    // sqrt(2)
#define RAUL_SQRT1_2 0.707106781186547524401 // 1/sqrt(2)

#define GELU_CONST 0.044715

namespace raul
{

enum class Limit : int
{
    Left = 0,
    Middle = 1,
    Right = 2
};

enum class Dimension : int
{
    Default = -1,
    Batch = 0,
    Depth = 1,
    Height = 2,
    Width = 3
};

#if defined(_MSC_VER)
#define INLINE __forceinline
#else
#define INLINE __attribute__((always_inline))
#endif

template<typename Type>
class TensorImpl;
typedef TensorImpl<dtype> Tensor;
typedef TensorImpl<half> TensorFP16;
class TensorGPU;
class TensorGPUHelper;

#define TOMMTYPE(var) static_cast<typename MM::type>(var)

using shape = yato::dimensionality<4U, size_t>;
} // raul namespace

namespace raul
{

enum class NetworkMode
{
    Train = 0,
    Test = 1,
    TrainCheckpointed = 2
};

enum class CompressionMode
{
    NONE = -1,
    FP16 = 0,
    INT8 = 1
};

enum class CalculationMode
{
    DETERMINISTIC = 0,
#if defined(_OPENMP)
    FAST = 1,
#endif
};

/**
 * @brief Hardware target platform
 *
 * \note Use GPU for all supported operations
 */
enum class ExecutionTarget
{
    CPU = 0,
    GPU = 1,
    CPUFP16 = 2
};

/**
 * @brief Hardware target platform per layer
 *
 * \note Might override execution target for workflow, useful for mixed precision
 */
enum class LayerExecutionTarget
{
    Default = -1, // use same as ExecutionTarget
    CPU = 0,      // from this point enums should be aligned with ExecutionTarget (due to LayerExecutionTarget = static_cast<ExecutionTarget>(enum))
    GPU = 1,
    CPUFP16 = 2
};

/**
 * @brief Memory allocation mode
 */
enum class AllocationMode
{
    STANDARD,
    POOL
};

enum class DeclarationType
{
    Tensor = 0,
    Shape = 1,
    //    Alias = 2
};

class OpenclInitializer;

class Common
{
  public:
    static bool hasOpenCL();
    // calling this function after creating workflow will lead to errors (due to different contexts)
    static void setGpuPlatformAndDevice(std::optional<size_t> platform_id = std::nullopt, std::optional<size_t> device_id = std::nullopt);
    static std::tuple<cl::Platform, cl::Device, cl::Context> getGpuPlatformDeviceAndContext();

    static void checkOpenCLStatus(cl_int status, const std::string& caller, const std::string& message);

    /*
     * [cols x rows]
     * A[k x m]
     * B[n x k]
     * C[n x m]
     * https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm
     * C = alpha * A * B + beta * C
     * bOffset - in elements (not bytes)
     */
    static void gemm(void*,
                     const Name& caller,
                     OPENBLAS_CONST CBLAS_TRANSPOSE transA,
                     OPENBLAS_CONST CBLAS_TRANSPOSE transB,
                     size_t m,
                     size_t n,
                     size_t k,
                     OPENBLAS_CONST dtype alpha,
                     OPENBLAS_CONST dtype* a,
                     OPENBLAS_CONST dtype* b,
                     OPENBLAS_CONST dtype beta,
                     dtype* c,
                     dtype tmpBuf = 0);

    static void gemm(void*,
                     const Name& caller,
                     OPENBLAS_CONST CBLAS_TRANSPOSE transA,
                     OPENBLAS_CONST CBLAS_TRANSPOSE transB,
                     size_t m,
                     size_t n,
                     size_t k,
                     OPENBLAS_CONST dtype alpha,
                     OPENBLAS_CONST half* a,
                     OPENBLAS_CONST half* b,
                     OPENBLAS_CONST dtype beta,
                     half* c,
                     half tmpBuf = 0);

    static void gemm(void* kernelManager,
                     const Name& caller,
                     OPENBLAS_CONST CBLAS_TRANSPOSE transA,
                     OPENBLAS_CONST CBLAS_TRANSPOSE transB,
                     size_t m,
                     size_t n,
                     size_t k,
                     OPENBLAS_CONST dtype alpha,
                     const cl::Buffer a,
                     const cl::Buffer b,
                     OPENBLAS_CONST dtype beta,
                     cl::Buffer c,
                     cl::Buffer& tmp,
                     size_t aOffset = 0,
                     size_t bOffset = 0,
                     size_t cOffset = 0);
    /**
     * @brief : Basic Linear Algebra Subroutine y = y + ax
     *
     *  \f[
     *      \vec{y} = \vec{y} + \alpha * \vec{x},
     *  \f]
     *
     * @param n The number of elements in vectors x and y.
     * @param sa The scalar alpha.
     * @param sx The vector x of length n. Specified as: a one-dimensional array of (at least) length \f$ 1+(n-1)|incx| \f$.
     * @param incx The stride for vector x. Specified as: an integer. It can have any value.
     * @param sy The vector y of length n. Specified as: a one-dimensional array of (at least) length \f$ 1+(n-1)|incy| \f$.
     * @param incy The stride for vector y.
     * @param xOffset The offset for vector x.
     * @param yOffset The offset for vector y.
     * @return The vector y, containing the results of the computation.
     */
    static void axpy(void*, const Name& caller, size_t n, OPENBLAS_CONST dtype sa, OPENBLAS_CONST dtype* sx, size_t incx, dtype* sy, size_t incy, size_t xOffset = 0, size_t yOffset = 0);
    static void axpy(void*, const Name& caller, size_t n, OPENBLAS_CONST dtype sa, OPENBLAS_CONST half* sx, size_t incx, half* sy, size_t incy, size_t xOffset = 0, size_t yOffset = 0);
    static void axpy(void* kernelManager, const Name& caller, size_t n, OPENBLAS_CONST dtype sa, const cl::Buffer x, size_t incx, cl::Buffer y, size_t incy, size_t xOffset = 0, size_t yOffset = 0);

    /**
     * @brief : Basic Linear Algebra Subroutine y = ax + by
     *
     *  \f[
     *      \vec{y} = \alpha \vec{x} + \beta \vec{y},
     *  \f]
     *
     * @param n The number of elements in vectors x and y.
     * @param alpha The scalar alpha.
     * @param x The vector x of length n. Specified as: a one-dimensional array of (at least) length \f$ 1+(n-1)|incx| \f$.
     * @param incx The stride for vector x. Specified as: an integer. It can have any value.
     * @param beta The scalar beta.
     * @param y The vector y of length n. Specified as: a one-dimensional array of (at least) length \f$ 1+(n-1)|incy| \f$.
     * @param incy The stride for vector y.
     * @param xOffset The offset for vector x.
     * @param yOffset The offset for vector y.
     * @return The vector y, containing the results of the computation.
     */
    static int axpby(OPENBLAS_CONST size_t n,
                     OPENBLAS_CONST dtype alpha,
                     OPENBLAS_CONST dtype* x,
                     OPENBLAS_CONST size_t incx,
                     OPENBLAS_CONST dtype beta,
                     dtype* y,
                     OPENBLAS_CONST size_t incy,
                     size_t xOffset,
                     size_t yOffset);

    static int axpby(OPENBLAS_CONST size_t n,
                     OPENBLAS_CONST dtype alpha,
                     OPENBLAS_CONST half* x,
                     OPENBLAS_CONST size_t incx,
                     OPENBLAS_CONST dtype beta,
                     half* y,
                     OPENBLAS_CONST size_t incy,
                     size_t xOffset,
                     size_t yOffset);

    static void axpby(void* kernelManager,
                      const Name& caller,
                      OPENBLAS_CONST size_t n,
                      OPENBLAS_CONST dtype alpha,
                      const cl::Buffer x,
                      OPENBLAS_CONST size_t incx,
                      OPENBLAS_CONST dtype beta,
                      cl::Buffer y,
                      OPENBLAS_CONST size_t incy,
                      size_t xOffset,
                      size_t yOffset);

    /**
     * @brief : Basic Linear Algebra Subroutine y = alpha * a * x + beta * y
     *
     * Vector by vector element wise multiplication
     *
     *  \f[
     *      \vec{y} = \alpha \vec{a} \vec{x} + \beta \vec{y},
     *  \f]
     *
     * @param n The number of elements in vectors x and y.
     * @param alpha The scalar alpha.
     * @param a The vector of length n.
     * @param x The vector x of length n. Specified as: a one-dimensional array of (at least) length \f$ 1+(n-1)|incx| \f$.
     * @param incx The stride for vector x. Specified as: an integer. It can have any value.
     * @param beta The scalar beta.
     * @param y The vector y of length n. Specified as: a one-dimensional array of (at least) length \f$ 1+(n-1)|incy| \f$.
     * @param incy The stride for vector y.
     */

    static void hadamard(OPENBLAS_CONST size_t n,
                         OPENBLAS_CONST dtype alpha,
                         OPENBLAS_CONST dtype* a,
                         OPENBLAS_CONST dtype* x,
                         OPENBLAS_CONST size_t incx,
                         OPENBLAS_CONST dtype beta,
                         dtype* y,
                         OPENBLAS_CONST size_t incy);

    static dtype dot(size_t n, OPENBLAS_CONST dtype* sx, size_t incx, OPENBLAS_CONST dtype* sy, size_t incy);

    static void scal(size_t n, OPENBLAS_CONST dtype sa, dtype* sx, size_t incx);

    static void transpose(Tensor& tensor, size_t cols);
    static void transpose(TensorGPUHelper&& tensor, size_t cols);
    static void transpose(TensorFP16& tensor, size_t cols);

    /*
     * memory for dst should be allocated externaly
     */
    static void addPadding1D(const dtype* src, dtype* dst, size_t srcChannels, size_t srcSize, size_t dstSize, bool reversedOrder = false);
    static void addPadding2D(const dtype* src, dtype* dst, size_t srcChannels, size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight);

    /*
     * memory for dst should be allocated externaly
     */
    static void removePadding1D(const dtype* src, dtype* dst, size_t srcChannels, size_t srcSize, size_t dstSize, bool reversedOrder = false, bool overwrite = true);
    static void removePadding2D(const dtype* src, dtype* dst, size_t srcChannels, size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight, bool overwrite = true);

    /*
     * paddingWidth, paddingHeight - zero padding added for both sides of the input
     * memory for matrix should be allocated externaly
     */
    template<typename T>
    static void im2col(const T* image,
                       size_t imageWidth,
                       size_t imageHeight,
                       size_t imageChannels,
                       size_t filterWidth,
                       size_t filterHeight,
                       size_t strideWidth,
                       size_t strideHeight,
                       size_t paddingWidth,
                       size_t paddingHeight,
                       T* matrix,
                       bool reversedOrder = false);

    static size_t im2colOutputSize(size_t imageWidth,
                                   size_t imageHeight,
                                   size_t imageChannels,
                                   size_t filterWidth,
                                   size_t filterHeight,
                                   size_t strideWidth,
                                   size_t strideHeight,
                                   size_t paddingWidth,
                                   size_t paddingHeight,
                                   size_t dilationWidth,
                                   size_t dilationHeight);

    /*
     * paddingWidth, paddingHeight - zero padding added for both sides of the input
     * memory for image should be allocated externaly
     */
    template<typename T>
    static void col2im(const T* matrix,
                       size_t imageWidth,
                       size_t imageHeight,
                       size_t imageChannels,
                       size_t filterWidth,
                       size_t filterHeight,
                       size_t strideWidth,
                       size_t strideHeight,
                       size_t paddingWidth,
                       size_t paddingHeight,
                       T* image,
                       bool reversedOrder = false,
                       bool zeroOutput = true);

    /*
     * Rectified Linear Unit
     */
    template<typename T>
    static T ReLU(T x)
    {
        return std::max(static_cast<T>(0), x);
    }
    template<typename T>
    static T ReLU6(T x)
    {
        return std::min(std::max(static_cast<T>(0), x), static_cast<T>(6.0_dt));
    }

    template<typename T>
    static void ReLU(void*, const Name&, size_t, size_t, size_t, size_t, const T& in, T& out)
    {
        std::transform(in.begin(), in.end(), out.begin(), [&](typename T::type val) -> typename T::type { return ReLU(val); });
    }
    static void ReLU(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);

    template<typename T>
    static void ReLU6(void*, const Name&, size_t, size_t, size_t, size_t, const T& in, T& out)
    {
        std::transform(in.begin(), in.end(), out.begin(), [&](typename T::type val) -> typename T::type { return ReLU6(val); });
    }
    static void ReLU6(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);

    template<typename T>
    static void ReLUBackward(void*, const Name&, size_t, size_t, size_t, size_t, const T& out, const T& delta, T& prevDelta)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < prevDelta.size(); ++q)
        {
            prevDelta[q] += (out[q] > static_cast<typename T::type>(0)) ? delta[q] : static_cast<typename T::type>(0);
        }
    }
    static void ReLUBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& delta, TensorGPU& prevDelta);

    template<typename T>
    static void ReLU6Backward(void*, const Name&, size_t, size_t, size_t, size_t, const T& out, const T& delta, T& prevDelta)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < prevDelta.size(); ++q)
        {
            prevDelta[q] += (out[q] > static_cast<typename T::type>(0) && out[q] < static_cast<typename T::type>(6.0f)) ? delta[q] : static_cast<typename T::type>(0);
        }
    }
    static void ReLU6Backward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& delta, TensorGPU& prevDelta);

    /*
     * Gaussian error linear unit
     * @see https://arxiv.org/abs/1606.08415
     */
    static dtype GeLU_Erf(dtype x);
    static dtype GeLU_Tanh(dtype x);

    /*
     * Hard Sigmoid
     */
    template<typename T>
    static T HSigmoid(T x)
    {
        return static_cast<T>(ReLU6(TODTYPE(x) + 3.0_dt) / 6.0_dt);
    }

    /*
     * Hard Swish
     */
    template<typename T>
    static T HSwish(T x)
    {
        return x * HSigmoid(x);
    }
    static dtype sign(dtype x) { return TODTYPE((0.0_dt < x) - (x < 0.0_dt)); }

    static void round(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void nonZeroMask(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);

    static void zeroOutput(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& length, TensorGPU& out);

    static void sequenceMask(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& lengths, TensorGPU& mask);

    static void reduceTargets(void* kernelManager,
                              const Name& caller,
                              size_t batch,
                              size_t idepth,
                              size_t odepth,
                              size_t iheight,
                              size_t oheight,
                              size_t width,
                              size_t reductionFactor,
                              const TensorGPU& in,
                              TensorGPU& out);

    static void
    copy(void* kernelManager, const Name& caller, size_t sourceLen, size_t destinationLen, size_t sourceOffset, size_t destinationOffset, bool sumWithOldValues, const TensorGPU& in, TensorGPU& out);

    static void initAlignment(void* kernelManager, const Name& caller, dtype val, size_t batch, size_t height, TensorGPU& out);

    static void reverse(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void reverse(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& length, TensorGPU& out);

    static void gaussianUpsamplingDistributionForward(void* kernelManager,
                                                      const Name& caller,
                                                      size_t batch,
                                                      size_t depth,
                                                      size_t height,
                                                      size_t width,
                                                      const TensorGPU& values,
                                                      const TensorGPU& loc,
                                                      const TensorGPU& scale,
                                                      TensorGPU& out);
    static void gaussianUpsamplingDistributionBackward(void* kernelManager,
                                                       const Name& caller,
                                                       size_t batch,
                                                       size_t depth,
                                                       size_t height,
                                                       size_t width,
                                                       bool backwardForLoc,
                                                       const TensorGPU& values,
                                                       const TensorGPU& loc,
                                                       const TensorGPU& scale,
                                                       const TensorGPU& deltas,
                                                       TensorGPU& prevDelta);

    static void dynamicDepthwiseConv2DForward(void* kernelManager,
                                              const Name& caller,
                                              size_t batchSize,
                                              size_t inputC,
                                              size_t outputH,
                                              size_t outputW,
                                              size_t channelMultiplier,
                                              size_t filterH,
                                              size_t filterW,
                                              const TensorGPU& in0,
                                              const TensorGPU& in1,
                                              TensorGPU& out);

    static void dynamicDepthwiseConv2DBackward(void* kernelManager,
                                               const Name& caller,
                                               size_t batchSize,
                                               size_t inputC,
                                               size_t outputH,
                                               size_t outputW,
                                               size_t channelMultiplier,
                                               size_t filterH,
                                               size_t filterW,
                                               bool isForInput,
                                               const TensorGPU& in0,
                                               const TensorGPU& in1,
                                               TensorGPU& out);

    static void expForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void expBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void sqrtForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void sqrtBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void rsqrtForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void rsqrtBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void squareForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void squareBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void logForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void logBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void addBias(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, dtype bias, const TensorGPU& in, TensorGPU& out);

    static void sigmoidForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void sigmoidBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void softplusForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, dtype beta, dtype threshold, const TensorGPU& in, TensorGPU& out);
    static void softplusBackward(void* kernelManager,
                                 const Name& caller,
                                 size_t batch,
                                 size_t depth,
                                 size_t height,
                                 size_t width,
                                 dtype beta,
                                 dtype threshold,
                                 const TensorGPU& out,
                                 const TensorGPU& deltas,
                                 TensorGPU& prevDelta);

    static void tanhForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void tanhBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void swishForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void swishBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void splitterForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void splitterBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void softmaxForward(void* kernelManager, const Name& caller, size_t externalDimSize, size_t internalDimSize, const TensorGPU& in, TensorGPU& out);
    static void softmaxBackward(void* kernelManager, const Name& caller, size_t externalDimSize, size_t internalDimSize, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void geluErfForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void geluErfBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void geluTanhForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void geluTanhBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void
    selectForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& cond, const TensorGPU& in0, const TensorGPU& in1, TensorGPU& out);
    static void selectBackward(void* kernelManager,
                               const Name& caller,
                               size_t index,
                               size_t batch,
                               size_t depth,
                               size_t height,
                               size_t width,
                               const TensorGPU& cond,
                               const TensorGPU& deltas,
                               TensorGPU& prevDelta);

    static void hsigmoidForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void hsigmoidBackward(void* kernelManager, const Name& caller, size_t size, dtype leftDivisor, dtype rightDivisor, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta);

    static void hswishForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out);
    static void hswishBackward(void* kernelManager, const Name& caller, size_t size, dtype a, dtype b, dtype c, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta);

    template<typename T, typename U>
    static void copyView(const T& view_from, U& view_to, bool overwrite = false)
    {
        std::function<typename T::value_type(const typename T::value_type from, const typename T::value_type to)> operation = std::plus<typename T::value_type>();

        if (overwrite)
        {
            operation = [](const typename T::value_type from, const typename T::value_type) { return from; };
        }

        for (size_t i1 = 0; i1 < view_from.size(0); ++i1)
        {
            for (size_t i2 = 0; i2 < view_from.size(1); ++i2)
            {
                for (size_t i3 = 0; i3 < view_from.size(2); ++i3)
                {
                    for (size_t i4 = 0; i4 < view_from.size(3); ++i4)
                    {
                        view_to[i1][i2][i3][i4] = operation(view_from[i1][i2][i3][i4], view_to[i1][i2][i3][i4]);
                    }
                }
            }
        }
    }

    /*
     * Upper triangle of a rectangular array
     */
    template<typename T>
    static void triu(T* data, size_t nrows, size_t ncols, int diag = 0)
    {
        size_t i = 0;
        int cols = (int)ncols;
        int rows = (int)nrows;
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c, ++i)
            {
                if (c - r - diag < 0)
                {
                    data[i] = static_cast<T>(0);
                }
            }
        }
    }

    /*
     * Applies a 1D convolution over an input signal composed of several input planes.
     * Supports 2 modes:
     *  1. PyTorch style: Input[N, C, 1, L1] (or [N, 1, C, L1]) -> Output[N, FILTERS, 1, L2] (or [N, 1, FILTERS, L2])
     *  2. TensorFlow style: Input[N, L1, 1, C] (or [N, 1, L1, C]) -> Output[N, L2, 1, FILTERS] (or [N, 1, L2, FILTERS])
     * Output is not zeroed prior to convolution (operator += is used)
     */
    static void conv1d(const dtype* input,
                       dtype* output,
                       const dtype* kernel,
                       const dtype* bias,
                       size_t batchSize,
                       size_t inputSize,
                       size_t inputChannels,
                       size_t outputSize,
                       size_t outputChannels,
                       size_t kernelSize,
                       size_t padding,
                       size_t stride,
                       size_t dilation = 1U,
                       size_t groups = 1U,
                       bool tfStyle = false);

    /*
     * Applies 2D convolution over input tensor, all channels convolved
     * Output is not zeroed prior to convolution (operator += is used)
     */
    static void conv2d(const dtype* input,
                       dtype* output,
                       const dtype* kernel,
                       const dtype* bias,
                       size_t batchSize,
                       size_t inputWidth,
                       size_t inputHeight,
                       size_t inputChannels,
                       size_t outputWidth,
                       size_t outputHeight,
                       size_t outputChannels,
                       size_t kernelWidth,
                       size_t kernelHeight,
                       size_t paddingW,
                       size_t paddingH,
                       size_t strideW,
                       size_t strideH,
                       size_t dilationW = 1U,
                       size_t dilationH = 1U,
                       size_t groups = 1U);

    template<typename T = dtype, typename Iterator>
    static void arange(Iterator begin, Iterator end, T start = static_cast<T>(0), T step = static_cast<T>(1))
    {
        auto val = start;
        for (auto p = begin; p != end; ++p)
        {
            *p = static_cast<std::remove_reference_t<decltype(*p)>>(val);
            val += step;
        }
    }

    template<typename T = dtype, typename Iterable>
    static void arange(Iterable& i, T start = static_cast<T>(0), T step = static_cast<T>(1))
    {
        return arange(i.begin(), i.end(), start, step);
    }

    static void replaceAll(std::string& str, const std::string& srcSubstr, const std::string& tgtSubstr)
    {
        size_t start_pos = 0;
        while ((start_pos = str.find(srcSubstr, start_pos)) != std::string::npos)
        {
            str.replace(start_pos, srcSubstr.length(), tgtSubstr);
            start_pos += tgtSubstr.length(); // srcSubstr could be a substring of tgtSubstr
        }
    }

    static bool startsWith(const std::string& str, const std::string& srcSubstr) { return (str.rfind(srcSubstr, 0) == 0); }

    static std::vector<std::string> split(const std::string& string, char delimeter);

    /*
     * @see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
     */
    template<typename T>
    static bool shapeIsBroadcastable(const T& from, const T& to)
    {
        const auto n = to.dimensions_num();
        for (size_t i = 0; i < n; ++i)
        {
            if (from[i] != to[i] && from[i] != 1U && to[i] != 1U)
            {
                return false;
            }
        }
        return true;
    }

    static bool endsWith(std::string const& value, std::string const& ending)
    {
        if (ending.size() > value.size())
        {
            return false;
        }
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    }

    static raul::shape getStrides(const raul::shape& tensor_shape);

    static raul::shape offsetToIndexes(size_t offset, const raul::shape& strides);

    static size_t indexesToOffset(const raul::shape& indexes, const raul::shape& strides);
};

template<class T>
bool if_equals(const std::string&& error, const T val1, const T val2)
{
    if (val1 != val2)
    {
        throw(std::runtime_error(error));
    }
    return val1 == val2;
}

void checkOpenCLStatus(cl_int status, const std::string& caller, const std::string& message);

} // raul namespace

#endif // COMMON_H
