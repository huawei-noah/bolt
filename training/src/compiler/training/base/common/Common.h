// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

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

#include <training/system/Errors.h>
#include <training/system/Name.h>
#include <training/system/TypeHalf.h>
#include <training/system/Types.h>

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

#if defined(ANDROID)
#define TOMMTYPE(var) static_cast<typename MM::type>(var)
#else
#define TOMMTYPE(var) castHelper<typename MM::type>::cast(var)
#endif

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
 */
enum class ExecutionTarget
{
    CPU = 0,
    CPUFP16 = 1
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
    CPUFP16 = 1
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

    // generate vector of random index permutation of [0..n-1]
    static void generate_permutation(size_t n, std::vector<size_t>& ind_vector, unsigned int seed = 0);

    /*
     * [cols x rows]
     * A[k x m]
     * B[n x k]
     * C[n x m]
     * C = alpha * A * B + beta * C
     * bOffset - in elements (not bytes)
     */
    static void gemm(OPENBLAS_CONST CBLAS_TRANSPOSE transA,
                     OPENBLAS_CONST CBLAS_TRANSPOSE transB,
                     size_t m,
                     size_t n,
                     size_t k,
                     OPENBLAS_CONST dtype alpha,
                     OPENBLAS_CONST dtype* a,
                     OPENBLAS_CONST dtype* b,
                     OPENBLAS_CONST dtype beta,
                     dtype* c);

    static void gemm(OPENBLAS_CONST CBLAS_TRANSPOSE transA,
                     OPENBLAS_CONST CBLAS_TRANSPOSE transB,
                     size_t m,
                     size_t n,
                     size_t k,
                     OPENBLAS_CONST dtype alpha,
                     OPENBLAS_CONST half* a,
                     OPENBLAS_CONST half* b,
                     OPENBLAS_CONST dtype beta,
                     half* c);

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
    static void axpy(size_t n, OPENBLAS_CONST dtype sa, OPENBLAS_CONST dtype* sx, size_t incx, dtype* sy, size_t incy, size_t xOffset = 0, size_t yOffset = 0);
    static void axpy(size_t n, OPENBLAS_CONST dtype sa, OPENBLAS_CONST half* sx, size_t incx, half* sy, size_t incy, size_t xOffset = 0, size_t yOffset = 0);

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
    static void transpose(TensorFP16& tensor, size_t cols);

    /*
     * memory for dst should be allocated externaly
     */
    static void addPadding1D(const dtype* src, dtype* dst, size_t srcChannels, size_t srcSize, size_t dstSize, bool reversedOrder = false);
    template<typename T>
    static void addPadding2D(const T* src, T* dst, size_t srcChannels, size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight)
    {
        if ((dstWidth >= srcWidth) && (dstHeight >= srcHeight))
        {
            size_t padWidth = dstWidth - srcWidth;
            size_t padHeight = dstHeight - srcHeight;

            size_t leftPad = padWidth / 2;
            // size_t rightPad = padWidth - leftPad;
            size_t topPad = padHeight / 2;
            size_t bottomPad = padHeight - topPad;

            for (size_t d = 0; d < srcChannels; ++d)
            {
                // top
                for (size_t y = 0; y < topPad; ++y)
                {
                    for (size_t x = 0; x < dstWidth; ++x)
                    {
                        dst[d * dstWidth * dstHeight + dstWidth * y + x] = static_cast<T>(0.0_dt);
                    }
                }

                for (size_t y = topPad; y < topPad + srcHeight; ++y)
                {
                    // left
                    for (size_t x = 0; x < leftPad; ++x)
                    {
                        dst[d * dstWidth * dstHeight + dstWidth * y + x] = static_cast<T>(0.0_dt);
                    }

                    // src
                    for (size_t x = leftPad; x < leftPad + srcWidth; ++x)
                    {
                        dst[d * dstWidth * dstHeight + dstWidth * y + x] = src[d * srcWidth * srcHeight + srcWidth * (y - topPad) + x - leftPad];
                    }

                    // right
                    for (size_t x = leftPad + srcWidth; x < dstWidth; ++x)
                    {
                        dst[d * dstWidth * dstHeight + dstWidth * y + x] = static_cast<T>(0.0_dt);
                    }
                }

                // bottom
                for (size_t y = dstHeight - bottomPad; y < dstHeight; ++y)
                {
                    for (size_t x = 0; x < dstWidth; ++x)
                    {
                        dst[d * dstWidth * dstHeight + dstWidth * y + x] = static_cast<T>(0.0_dt);
                    }
                }
            }
        }
    }

    /*
     * memory for dst should be allocated externaly
     */
    static void removePadding1D(const dtype* src, dtype* dst, size_t srcChannels, size_t srcSize, size_t dstSize, bool reversedOrder = false, bool overwrite = true);
    template<typename T>
    static void removePadding2D(const T* src, T* dst, size_t srcChannels, size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight, bool overwrite = true)
    {
        if ((dstWidth <= srcWidth) && (dstHeight <= srcHeight))
        {
            size_t padWidth = srcWidth - dstWidth;
            size_t padHeight = srcHeight - dstHeight;

            size_t leftPad = padWidth / 2;
            // size_t rightPad = padWidth - leftPad;
            size_t topPad = padHeight / 2;
            // size_t bottomPad = padHeight - topPad;

            if (overwrite)
            {
                for (size_t d = 0; d < srcChannels; ++d)
                {
                    for (size_t y = 0; y < dstHeight; ++y)
                    {
                        for (size_t x = 0; x < dstWidth; ++x)
                        {
                            dst[d * dstWidth * dstHeight + dstWidth * y + x] = src[d * srcWidth * srcHeight + srcWidth * (y + topPad) + x + leftPad];
                        }
                    }
                }
            }
            else
            {
                for (size_t d = 0; d < srcChannels; ++d)
                {
                    for (size_t y = 0; y < dstHeight; ++y)
                    {
                        for (size_t x = 0; x < dstWidth; ++x)
                        {
                            dst[d * dstWidth * dstHeight + dstWidth * y + x] += src[d * srcWidth * srcHeight + srcWidth * (y + topPad) + x + leftPad];
                        }
                    }
                }
            }
        }
    }

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
    static void ReLU(const T& in, T& out)
    {
        std::transform(in.begin(), in.end(), out.begin(), [&](typename T::type val) -> typename T::type { return ReLU(val); });
    }

    template<typename T>
    static void ReLU6(const T& in, T& out)
    {
        std::transform(in.begin(), in.end(), out.begin(), [&](typename T::type val) -> typename T::type { return ReLU6(val); });
    }

    template<typename T>
    static void ReLUBackward(const T& out, const T& delta, T& prevDelta)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < prevDelta.size(); ++q)
        {
            prevDelta[q] += (out[q] > static_cast<typename T::type>(0)) ? delta[q] : static_cast<typename T::type>(0);
        }
    }

    template<typename T>
    static void ReLU6Backward(const T& out, const T& delta, T& prevDelta)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < prevDelta.size(); ++q)
        {
            prevDelta[q] += (out[q] > static_cast<typename T::type>(0) && out[q] < static_cast<typename T::type>(6.0f)) ? delta[q] : static_cast<typename T::type>(0);
        }
    }

    /*
     * Gaussian error linear unit
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

    template<typename T, typename U>
    static void copyView(const T& view_from, U& view_to, const bool overwrite = false)
    {
        auto retLhs = [](typename T::value_type& lhs, [[maybe_unused]] typename T::value_type& rhs) { return lhs; };

        auto copyViewImpl = [](const T& view_from, U& view_to, auto&& func)
        {
            for (size_t i1 = 0; i1 < view_from.size(0); ++i1)
            {
                for (size_t i2 = 0; i2 < view_from.size(1); ++i2)
                {
                    for (size_t i3 = 0; i3 < view_from.size(2); ++i3)
                    {
                        for (size_t i4 = 0; i4 < view_from.size(3); ++i4)
                        {
                            view_to[i1][i2][i3][i4] = func(view_from[i1][i2][i3][i4], view_to[i1][i2][i3][i4]);
                        }
                    }
                }
            }
        };

        if (overwrite)
        {
            copyViewImpl(view_from, view_to, retLhs);
        }
        else
        {
            copyViewImpl(view_from, view_to, std::plus<typename T::value_type>());
        }
    }

    template<typename T>
    static void unpack4D(const T& src, T& dst, Dimension dir, size_t index, const Name& layerType, const Name& layerName, bool overwrite)
    {
        auto input4d = src.get4DView();
        auto inputDims = yato::dims(src.getDepth(), src.getHeight(), src.getWidth());

        auto outputDims = dst.getShape();

        const typename T::type* startEl = nullptr;
        switch (dir)
        {
            case Dimension::Depth:
                startEl = &input4d[0][index][0][0];
                break;
            case Dimension::Height:
                startEl = &input4d[0][0][index][0];
                break;
            default:
                throw std::runtime_error(layerType + "[" + layerName + "]: unpack4D unknown dim");
        }

        auto srcView = yato::array_view_4d<const typename T::type>(startEl, outputDims, inputDims);
        auto outputView = dst.get4DView();
        Common::copyView(srcView, outputView, overwrite);
    }

    template<typename T>
    static void pack4D(const T& src, T& dst, Dimension dir, size_t index, const Name& layerType, const Name& layerName, bool overwrite)
    {
        auto output4d = dst.get4DView();

        yato::dimensionality<3U, size_t> concatDims(dst.getDepth(), dst.getHeight(), dst.getWidth());

        auto srcView = src.get4DView();
        typename T::type* startEl = nullptr;
        switch (dir)
        {
            case Dimension::Depth:
                startEl = &output4d[0][index][0][0];
                break;
            case Dimension::Height:
                startEl = &output4d[0][0][index][0];
                break;
            default:
                throw std::runtime_error(layerType + "[" + layerName + "]: pack4D unknown dim");
        }

        auto dstView = yato::array_view_4d<typename T::type>(startEl, src.getShape(), concatDims);
        Common::copyView(srcView, dstView, overwrite);
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
    template<typename T>
    static void conv2d(const T* input,
                       T* output,
                       const T* kernel,
                       const T* bias,
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
                       size_t groups = 1U)
    {
        auto inputs3D = yato::array_view_3d<T>(const_cast<T*>(input), yato::dims(batchSize, inputChannels, inputHeight * inputWidth));
        auto outputs3D = yato::array_view_3d<T>(output, yato::dims(batchSize, outputChannels, outputHeight * outputWidth));
        auto kernelsWeights4D = yato::array_view_4d<T>(const_cast<T*>(kernel), yato::dims(outputChannels, inputChannels / groups, kernelHeight, kernelWidth));

        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t d = 0; d < outputChannels; ++d)
            {
                std::fill(outputs3D[q][d].begin(), outputs3D[q][d].end(), static_cast<T>(0.0_dt));
            }

            size_t inputWidthPadded = inputWidth + 2 * paddingW;
            size_t inputHeightPadded = inputHeight + 2 * paddingH;

            std::vector<T> inputPadded(inputChannels * inputHeightPadded * inputWidthPadded);

            Common::addPadding2D(&inputs3D[q][0][0], inputPadded.data(), inputChannels, inputWidth, inputHeight, inputWidthPadded, inputHeightPadded);

            auto inputPadded2D = yato::view(inputPadded).reshape(yato::dims(inputChannels, inputHeightPadded * inputWidthPadded));

            for (size_t group = 0; group < groups; ++group)
            {
                for (size_t kernelIndex = 0; kernelIndex < outputChannels / groups; ++kernelIndex)
                {
                    for (size_t d = 0; d < inputChannels / groups; ++d)
                    {
                        for (size_t oy = 0; oy < outputHeight; ++oy)
                        {
                            for (size_t ox = 0; ox < outputWidth; ++ox)
                            {
                                for (size_t ky = 0; ky < kernelHeight; ++ky)
                                {
                                    for (size_t kx = 0; kx < kernelWidth; ++kx)
                                    {
                                        outputs3D[q][kernelIndex + group * outputChannels / groups][oy * outputWidth + ox] +=
                                            kernelsWeights4D[kernelIndex + group * outputChannels / groups][d][ky][kx] *
                                            inputPadded2D[d + group * inputChannels / groups][oy * inputWidthPadded * strideH + ky * dilationH * inputWidthPadded + ox * strideW + kx * dilationW];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (bias)
        {
            for (size_t q = 0; q < batchSize; ++q)
            {
                for (size_t kernelIndex = 0; kernelIndex < outputChannels; ++kernelIndex)
                {
                    for (size_t oy = 0; oy < outputHeight; ++oy)
                    {
                        for (size_t ox = 0; ox < outputWidth; ++ox)
                        {
                            outputs3D[q][kernelIndex][oy * outputWidth + ox] += bias[kernelIndex];
                        }
                    }
                }
            }
        }
    }

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

    static shape getStrides(const shape& tensor_shape);

    static shape offsetToIndexes(size_t offset, const shape& strides);

    static size_t indexesToOffset(const shape& indexes, const shape& strides);
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

} // raul namespace

#endif // COMMON_H
