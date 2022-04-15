// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Common.h"

#include "Tensor.h"
#include "TensorGPU.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#ifdef _BLAS_ENHANCE
#include "blas_enhance.h"
#include "error.h"
#endif

#include <training/common/OpenclInitializer.h>
#include <training/common/TypeHalf.h>
#include <training/opencl/GPUCommon.h>
#include <training/opencl/GemmGPU.h>

using namespace std;
using namespace raul;

namespace
{

#if !defined(RAUL_USE_OPENBLAS) // && (!defined(_BLAS_ENHANCE)
/**
 * @brief  Performs one of the matrix-matrix operations
 *
 *   C := alpha*op( A )*op( B ) + beta*C,
 *   where  op( X ) is one of
 *   op( X ) = X   or   op( X ) = X',
 *
 *   alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *   an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 *
 * @param transA specifies the form of op( A )
 * @param transB specifies the form of op( B )
 * @param matA pointer to matrix A
 * @param matB pointer to matrix B
 * @param matC pointer to matrix C
 * @param m pecifies  the number  of rows  of the matrix op( A )
 * @param n specifies the number  of columns of the matrix  op( B )
 * @param k specifies  the number of columns of the matrix op( A ) and the number of rows of the matrix op( B )
 * @param alpha specifies the scalar alpha
 * @param beta pecifies the scalar  beta
 * @param lda stride in matrix A
 * @param ldb stride in matrix B
 * @param ldc stride in matrix C
 * @param bOffset offset in matrix b
 *
 * @see http://www.netlib.org/clapack/cblas/dgemm.c
 */
template<typename T>
void matrixMul(const CBLAS_TRANSPOSE transA,
               const CBLAS_TRANSPOSE transB,
               const T* matA,
               const T* matB,
               T* matC,
               size_t m,
               size_t n,
               size_t k,
               const raul::dtype alpha,
               const raul::dtype beta,
               size_t lda,
               size_t ldb,
               size_t ldc,
               size_t bOffset)
{
    assert(transA != CblasConjTrans);
    assert(transB != CblasConjTrans);
    assert(transA != CblasConjNoTrans);
    assert(transB != CblasConjNoTrans);

    if (transB == CblasNoTrans)
    {
        if (transA == CblasNoTrans)
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t r = 0; r < m; ++r)
            {
                for (size_t c = 0; c < n; ++c)
                {
                    T acc = 0;
                    for (size_t i = 0; i < k; i++)
                    {
                        acc += static_cast<T>(alpha) * matA[i + r * lda] * matB[c + i * ldb + bOffset];
                    }

                    matC[c + r * ldc] = static_cast<T>(acc) + static_cast<T>(beta) * matC[c + r * ldc];
                }
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t r = 0; r < m; ++r)
            {
                for (size_t c = 0; c < n; ++c)
                {
                    T acc = 0;
                    for (size_t i = 0; i < k; i++)
                    {
                        acc += static_cast<T>(alpha) * matA[r + i * lda] * matB[c + i * ldb + bOffset];
                    }

                    matC[c + r * ldc] = static_cast<T>(acc) + static_cast<T>(beta) * matC[c + r * ldc];
                }
            }
        }
    }
    else
    {
        if (transA == CblasNoTrans)
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t r = 0; r < m; ++r)
            {
                for (size_t c = 0; c < n; ++c)
                {
                    T acc = 0;
                    for (size_t i = 0; i < k; i++)
                    {
                        acc += static_cast<T>(alpha) * matA[i + r * lda] * matB[i + c * ldb + bOffset];
                    }

                    matC[c + r * ldc] = static_cast<T>(acc) + static_cast<T>(beta) * matC[c + r * ldc];
                }
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t r = 0; r < m; ++r)
            {
                for (size_t c = 0; c < n; ++c)
                {
                    T acc = 0;
                    for (size_t i = 0; i < k; i++)
                    {
                        acc += static_cast<T>(alpha) * matA[r + i * lda] * matB[i + c * ldb + bOffset];
                    }

                    matC[c + r * ldc] = static_cast<T>(acc) + static_cast<T>(beta) * matC[c + r * ldc];
                }
            }
        }
    }
}
#endif

template<typename Out>
void split(const string& st, char delimeter, Out result)
{
    stringstream stream;
    stream.str(st);
    string item;
    while (getline(stream, item, delimeter))
    {
        *(result++) = item;
    }
}

template<typename T>
void transpose(T* matrix, size_t cols, size_t size)
{
    // https://stackoverflow.com/questions/9227747/in-place-transposition-of-a-matrix
    const size_t mn1 = size - 1;
    const size_t n = size / cols;
    vector<bool> visited(size);
    T* cycle = matrix;
    while (++cycle != (matrix + size))
    {
        if (visited[cycle - matrix]) continue;
        size_t a = cycle - matrix;
        do
        {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            swap(*(matrix + a), *cycle);
            visited[a] = true;
        } while ((matrix + a) != cycle);
    }
}

void gemmCPUImpl(OPENBLAS_CONST CBLAS_TRANSPOSE transA,
                 OPENBLAS_CONST CBLAS_TRANSPOSE transB,
                 size_t m,
                 size_t n,
                 size_t k,
                 OPENBLAS_CONST raul::dtype alpha,
                 OPENBLAS_CONST raul::dtype* a,
                 OPENBLAS_CONST raul::dtype* b,
                 OPENBLAS_CONST raul::dtype beta,
                 raul::dtype* c)
{
    size_t bOffset = 0;
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    size_t lda = (transA == CblasNoTrans) ? k : m;
    size_t ldb = (transB == CblasNoTrans) ? n : k;

    cblas_sgemm(CblasRowMajor,
                transA,
                transB,
                static_cast<blasint>(m),
                static_cast<blasint>(n),
                static_cast<blasint>(k),
                alpha,
                a,
                static_cast<blasint>(lda),
                b + bOffset,
                static_cast<blasint>(ldb),
                beta,
                c,
                static_cast<blasint>(n));
#else
#if defined(_BLAS) && defined(_BLAS_ENHANCE)
    if ((alpha != 1 && beta != 0) || (beta != 0 && beta != 1))
    {
        size_t lda = (transA == CblasNoTrans) ? k : m;
        size_t ldb = (transB == CblasNoTrans) ? n : k;
        matrixMul(transA, transB, a, b, c, m, n, k, alpha, beta, lda, ldb, n, bOffset);
        return;
    }
    CHECK_REQUIREMENT(1 == alpha || 0 == beta);
    CHECK_REQUIREMENT(1 == beta || 0 == beta);
    TensorDesc matrixADesc, matrixBDesc;
    if (transA == CblasNoTrans)
    {
        matrixADesc = tensor2df(DT_F32, DF_NORMAL, static_cast<U32>(m), static_cast<U32>(k));
    }
    else
    {
        matrixADesc = tensor2df(DT_F32, DF_TRANSPOSE, static_cast<U32>(k), static_cast<U32>(m));
    }
    if (transB == CblasNoTrans)
    {
        matrixBDesc = tensor2df(DT_F32, DF_NORMAL, static_cast<U32>(k), static_cast<U32>(n));
    }
    else
    {
        matrixBDesc = tensor2df(DT_F32, DF_TRANSPOSE, static_cast<U32>(n), static_cast<U32>(k));
    }

    if (1 == m && transB == CblasTrans)
    {
        matrixADesc = tensor1d(DT_F32, static_cast<U32>(k));
        matrixBDesc.df = DF_NORMAL;
        TensorDesc matrixCDesc = tensor1d(DT_F32, static_cast<U32>(n));
        if (0 == beta)
        {
            memset(c, 0, tensorNumBytes(matrixCDesc));
        }
        matrix_vector_multiply(matrixBDesc, b + bOffset, matrixADesc, a, 0, nullptr, matrixCDesc, c, ARM_A76);
    }
    else if (1 == n && transA == CblasNoTrans)
    {
        matrixBDesc = tensor1d(DT_F32, static_cast<U32>(k));
        TensorDesc matrixCDesc = tensor1d(DT_F32, static_cast<U32>(m));
        if (0 == beta)
        {
            memset(c, 0, tensorNumBytes(matrixCDesc));
        }
        matrix_vector_multiply(matrixADesc, a, matrixBDesc, b + bOffset, 0, nullptr, matrixCDesc, c, ARM_A76);
    }
    else
    {
        TensorDesc matrixCDesc = tensor2df(DT_F32, DF_NORMAL, static_cast<U32>(m), static_cast<U32>(n));
        unsigned int bytes;
        matrix_matrix_multiply_tmp_bytes(matrixADesc, matrixBDesc, &bytes, ARM_A76);
        vector<char> tmp(bytes);
        if (0 == beta)
        {
            memset(c, 0, tensorNumBytes(matrixCDesc));
        }
        matrix_matrix_multiply(matrixADesc, a, matrixBDesc, b + bOffset, bytes, tmp.data(), matrixCDesc, c, ARM_A76);
    }

    if (alpha != 1_dt)
    {
        size_t sizeC = m * n;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < sizeC; ++i)
        {
            c[i] *= alpha;
        }
    }
#else
    size_t lda = (transA == CblasNoTrans) ? k : m;
    size_t ldb = (transB == CblasNoTrans) ? n : k;
    matrixMul(transA, transB, a, b, c, m, n, k, alpha, beta, lda, ldb, n, bOffset);
#endif
#endif
}

void gemmCPUImpl(OPENBLAS_CONST CBLAS_TRANSPOSE transA,
                 OPENBLAS_CONST CBLAS_TRANSPOSE transB,
                 size_t m,
                 size_t n,
                 size_t k,
                 OPENBLAS_CONST raul::dtype alpha,
                 OPENBLAS_CONST raul::half* a,
                 OPENBLAS_CONST raul::half* b,
                 OPENBLAS_CONST raul::dtype beta,
                 raul::half* c)
{
    size_t bOffset = 0;
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    size_t lda = (transA == CblasNoTrans) ? k : m;
    size_t ldb = (transB == CblasNoTrans) ? n : k;

    std::vector<raul::dtype> mA(k * m);
    std::vector<raul::dtype> mB(n * k + bOffset);
    std::vector<raul::dtype> mC(n * m);

    for (size_t q = 0; q < k * m; ++q)
    {
        mA[q] = raul::toFloat32(a[q]);
    }

    for (size_t q = bOffset; q < n * k; ++q)
    {
        mB[q] = raul::toFloat32(b[q]);
    }

    for (size_t q = 0; q < n * m; ++q)
    {
        mC[q] = raul::toFloat32(c[q]);
    }

    cblas_sgemm(CblasRowMajor,
                transA,
                transB,
                static_cast<blasint>(m),
                static_cast<blasint>(n),
                static_cast<blasint>(k),
                alpha,
                mA.data(),
                static_cast<blasint>(lda),
                mB.data() + bOffset,
                static_cast<blasint>(ldb),
                beta,
                mC.data(),
                static_cast<blasint>(n));

    for (size_t q = 0; q < n * m; ++q)
    {
        c[q] = raul::toFloat16(mC[q]);
    }
#else
#if defined(_BLAS_ENHANCE)
    if (alpha != 1 && beta != 0)
    {
        size_t lda = (transA == CblasNoTrans) ? k : m;
        size_t ldb = (transB == CblasNoTrans) ? n : k;
        matrixMul(transA, transB, a, b, c, m, n, k, alpha, beta, lda, ldb, n, bOffset);
        return;
    }
    CHECK_REQUIREMENT(1 == alpha || 0 == beta);
    CHECK_REQUIREMENT(1 == beta || 0 == beta);
    TensorDesc matrixADesc, matrixBDesc;
    if (transA == CblasNoTrans)
    {
        matrixADesc = tensor2df(DT_F16, DF_NORMAL, static_cast<U32>(m), static_cast<U32>(k));
    }
    else
    {
        matrixADesc = tensor2df(DT_F16, DF_TRANSPOSE, static_cast<U32>(k), static_cast<U32>(m));
    }
    if (transB == CblasNoTrans)
    {
        matrixBDesc = tensor2df(DT_F16, DF_NORMAL, static_cast<U32>(k), static_cast<U32>(n));
    }
    else
    {
        matrixBDesc = tensor2df(DT_F16, DF_TRANSPOSE, static_cast<U32>(n), static_cast<U32>(k));
    }

    if (1 == m && transB == CblasTrans)
    {
        matrixADesc = tensor1d(DT_F16, static_cast<U32>(k));
        matrixBDesc.df = DF_NORMAL;
        TensorDesc matrixCDesc = tensor1d(DT_F16, static_cast<U32>(n));
        if (0 == beta)
        {
            memset(c, 0, tensorNumBytes(matrixCDesc));
        }
        matrix_vector_multiply(matrixBDesc, b + bOffset, matrixADesc, a, 0, nullptr, matrixCDesc, c, ARM_A76);
    }
    else if (1 == n && transA == CblasNoTrans)
    {
        matrixBDesc = tensor1d(DT_F16, static_cast<U32>(k));
        TensorDesc matrixCDesc = tensor1d(DT_F16, static_cast<U32>(m));
        if (0 == beta)
        {
            memset(c, 0, tensorNumBytes(matrixCDesc));
        }
        matrix_vector_multiply(matrixADesc, a, matrixBDesc, b + bOffset, 0, nullptr, matrixCDesc, c, ARM_A76);
    }
    else
    {
        TensorDesc matrixCDesc = tensor2df(DT_F16, DF_NORMAL, static_cast<U32>(m), static_cast<U32>(n));
        unsigned int bytes;
        matrix_matrix_multiply_tmp_bytes(matrixADesc, matrixBDesc, &bytes, ARM_A76);
        vector<char> tmp(bytes);
        if (0 == beta)
        {
            memset(c, 0, tensorNumBytes(matrixCDesc));
        }
        matrix_matrix_multiply(matrixADesc, a, matrixBDesc, b + bOffset, bytes, tmp.data(), matrixCDesc, c, ARM_A76);
    }

    if (alpha != 1_dt)
    {
        size_t sizeC = m * n;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < sizeC; ++i)
        {
            c[i] = static_cast<raul::half>(static_cast<float>(c[i]) * alpha);
        }
    }
#else
    size_t lda = (transA == CblasNoTrans) ? k : m;
    size_t ldb = (transB == CblasNoTrans) ? n : k;

    std::vector<raul::dtype> mA(k * m);
    std::vector<raul::dtype> mB(n * k + bOffset);
    std::vector<raul::dtype> mC(n * m);

    for (size_t q = 0; q < k * m; ++q)
    {
        mA[q] = raul::toFloat32(a[q]);
    }

    for (size_t q = bOffset; q < n * k; ++q)
    {
        mB[q] = raul::toFloat32(b[q]);
    }

    for (size_t q = 0; q < n * m; ++q)
    {
        mC[q] = raul::toFloat32(c[q]);
    }

    matrixMul(transA, transB, mA.data(), mB.data(), mC.data(), m, n, k, alpha, beta, lda, ldb, n, bOffset);

    for (size_t q = 0; q < n * m; ++q)
    {
        c[q] = raul::toFloat16(mC[q]);
    }
#endif
#endif
}

void axpyCPUImpl(size_t n, OPENBLAS_CONST raul::dtype sa, OPENBLAS_CONST raul::dtype* sx, size_t incx, raul::dtype* sy, size_t incy, size_t xOffset, size_t yOffset)
{
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    cblas_saxpy(static_cast<blasint>(n), sa, &sx[xOffset], static_cast<blasint>(incx), &sy[yOffset], static_cast<blasint>(incy));
#else
#if defined(_BLAS_ENHANCE)
    CHECK_REQUIREMENT(1 == incx && 1 == incy);
    TensorDesc vDesc = tensor1d(DT_F32, static_cast<U32>(n));
    vector_vector_axpby(sa, vDesc, &sx[xOffset], 1, vDesc, &sy[yOffset], ARM_A76);
#else
    size_t indexX = 0;
    size_t indexY = 0;
    for (size_t index = 0; index < n; ++index)
    {
        sy[yOffset + indexY] += sa * sx[xOffset + indexX];
        indexX += incx;
        indexY += incy;
    }
#endif
#endif
}

void axpyCPUImpl(size_t n, OPENBLAS_CONST raul::dtype sa, OPENBLAS_CONST raul::half* sx, size_t incx, raul::half* sy, size_t incy, size_t xOffset, size_t yOffset)
{
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    std::vector<raul::dtype> mSX(n * incx + xOffset);
    std::vector<raul::dtype> mSY(n * incy + yOffset);

    for (size_t q = xOffset; q < mSX.size(); q += incx)
    {
        mSX[q] = raul::toFloat32(sx[q]);
    }

    for (size_t q = yOffset; q < mSY.size(); q += incy)
    {
        mSY[q] = raul::toFloat32(sy[q]);
    }

    cblas_saxpy(static_cast<blasint>(n), sa, &mSX[xOffset], static_cast<blasint>(incx), &mSY[yOffset], static_cast<blasint>(incy));

    for (size_t q = yOffset; q < mSY.size(); q += incy)
    {
        sy[q] = raul::toFloat16(mSY[q]);
    }
#else
#if defined(_BLAS_ENHANCE)
    CHECK_REQUIREMENT(1 == incx && 1 == incy);
    TensorDesc vDesc = tensor1d(DT_F16, static_cast<U32>(n));
    vector_vector_axpby(sa, vDesc, &sx[xOffset], 1, vDesc, &sy[yOffset], ARM_A76);
#else
    std::vector<raul::dtype> mSX(n * incx + xOffset);
    std::vector<raul::dtype> mSY(n * incy + yOffset);

    for (size_t q = xOffset; q < mSX.size(); q += incx)
    {
        mSX[q] = raul::toFloat32(sx[q]);
    }

    for (size_t q = yOffset; q < mSY.size(); q += incy)
    {
        mSY[q] = raul::toFloat32(sy[q]);
    }

    size_t indexX = 0;
    size_t indexY = 0;
    for (size_t index = 0; index < n; ++index)
    {
        mSY[yOffset + indexY] += sa * mSX[xOffset + indexX];
        indexX += incx;
        indexY += incy;
    }

    for (size_t q = yOffset; q < mSY.size(); q += incy)
    {
        sy[q] = raul::toFloat16(mSY[q]);
    }
#endif
#endif
}
} // anonymous namespace

namespace raul
{

std::tuple<cl::Platform, cl::Device, cl::Context> Common::getGpuPlatformDeviceAndContext()
{
    return OpenCLInitializer::Instance().getGpuPlatformDeviceAndContext();
}

bool Common::hasOpenCL()
{
    return OpenCLInitializer::Instance().hasOpenCL();
}

void Common::setGpuPlatformAndDevice(std::optional<size_t> platform, std::optional<size_t> device)
{
    OpenCLInitializer::Instance().setGpuPlatformAndDevice(platform, device);
}

void Common::checkOpenCLStatus(cl_int status, const std::string& caller, const std::string& message)
{
    if (status != CL_SUCCESS)
    {
        std::map<int, std::string> map = {
            { 0, "SUCCESS" },
            { -1, "DEVICE_NOT_FOUND" },
            { -2, "DEVICE_NOT_AVAILABLE" },
            { -3, "COMPILER_NOT_AVAILABLE" },
            { -4, "MEM_OBJECT_ALLOCATION_FAILURE" },
            { -5, "OUT_OF_RESOURCES" },
            { -6, "OUT_OF_HOST_MEMORY" },
            { -7, "PROFILING_INFO_NOT_AVAILABLE" },
            { -8, "MEM_COPY_OVERLAP" },
            { -9, "IMAGE_FORMAT_MISMATCH" },
            { -10, "IMAGE_FORMAT_NOT_SUPPORTED" },
            { -11, "BUILD_PROGRAM_FAILURE" },
            { -12, "MAP_FAILURE" },
            { -13, "MISALIGNED_SUB_BUFFER_OFFSET" },
            { -14, "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST" },
            { -15, "COMPILE_PROGRAM_FAILURE" },
            { -16, "LINKER_NOT_AVAILABLE" },
            { -17, "LINK_PROGRAM_FAILURE" },
            { -18, "DEVICE_PARTITION_FAILED" },
            { -19, "KERNEL_ARG_INFO_NOT_AVAILABLE" },

            { -30, "INVALID_VALUE" },
            { -31, "INVALID_DEVICE_TYPE" },
            { -32, "INVALID_PLATFORM" },
            { -33, "INVALID_DEVICE" },
            { -34, "INVALID_CONTEXT" },
            { -35, "INVALID_QUEUE_PROPERTIES" },
            { -36, "INVALID_COMMAND_QUEUE" },
            { -37, "INVALID_HOST_PTR" },
            { -38, "INVALID_MEM_OBJECT" },
            { -39, "INVALID_IMAGE_FORMAT_DESCRIPTOR" },
            { -40, "INVALID_IMAGE_SIZE" },
            { -41, "INVALID_SAMPLER" },
            { -42, "INVALID_BINARY" },
            { -43, "INVALID_BUILD_OPTIONS" },
            { -44, "INVALID_PROGRAM" },
            { -45, "INVALID_PROGRAM_EXECUTABLE" },
            { -46, "INVALID_KERNEL_NAME" },
            { -47, "INVALID_KERNEL_DEFINITION" },
            { -48, "INVALID_KERNEL" },
            { -49, "INVALID_ARG_INDEX" },
            { -50, "INVALID_ARG_VALUE" },
            { -51, "INVALID_ARG_SIZE" },
            { -52, "INVALID_KERNEL_ARGS" },
            { -53, "INVALID_WORK_DIMENSION" },
            { -54, "INVALID_WORK_GROUP_SIZE" },
            { -55, "INVALID_WORK_ITEM_SIZE" },
            { -56, "INVALID_GLOBAL_OFFSET" },
            { -57, "INVALID_EVENT_WAIT_LIST" },
            { -58, "INVALID_EVENT" },
            { -59, "INVALID_OPERATION" },
            { -60, "INVALID_GL_OBJECT" },
            { -61, "INVALID_BUFFER_SIZE" },
            { -62, "INVALID_MIP_LEVEL" },
            { -63, "INVALID_GLOBAL_WORK_SIZE" },
            { -64, "INVALID_PROPERTY" },
            { -65, "INVALID_IMAGE_DESCRIPTOR" },
            { -66, "INVALID_COMPILER_OPTIONS" },
            { -67, "INVALID_LINKER_OPTIONS" },
            { -68, "INVALID_DEVICE_PARTITION_COUNT" },
        };

        auto it = map.find(status);
        if (it != map.end())
        {
            THROW_NONAME(caller, message + " (" + to_string(status) + " '" + it->second + "')");
        }
        else
        {
            THROW_NONAME(caller, message + " (" + to_string(status) + ")");
        }
    }
}

void Common::conv1d(const dtype* input,
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
                    size_t dilation,
                    size_t groups,
                    bool tfStyle)
{
    auto inputs3D = tfStyle ? yato::array_view_3d<dtype>(const_cast<dtype*>(input), yato::dims(batchSize, inputSize, inputChannels))
                            : yato::array_view_3d<dtype>(const_cast<dtype*>(input), yato::dims(batchSize, inputChannels, inputSize));
    auto outputs3D =
        tfStyle ? yato::array_view_3d<dtype>(output, yato::dims(batchSize, outputSize, outputChannels)) : yato::array_view_3d<dtype>(output, yato::dims(batchSize, outputChannels, outputSize));

    auto kernelsWeights3D = tfStyle ? yato::array_view_3d<dtype>(const_cast<dtype*>(kernel), yato::dims(kernelSize, inputChannels / groups, outputChannels))
                                    : yato::array_view_3d<dtype>(const_cast<dtype*>(kernel), yato::dims(outputChannels, inputChannels / groups, kernelSize));

    const auto firstOutputDimension = tfStyle ? outputSize : outputChannels;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < batchSize; ++q)
    {
        for (size_t d = 0; d < firstOutputDimension; ++d)
        {
            fill(outputs3D[q][d].begin(), outputs3D[q][d].end(), 0.0_dt);
        }

        size_t inputSizePadded = inputSize + 2 * padding;

        vector<dtype> inputPadded(inputChannels * inputSizePadded);

        Common::addPadding1D(&inputs3D[q][0][0], inputPadded.data(), inputChannels, inputSize, inputSizePadded, tfStyle);

        auto inputPadded2D = tfStyle ? yato::view(inputPadded).reshape(yato::dims(inputSizePadded, inputChannels)) : yato::view(inputPadded).reshape(yato::dims(inputChannels, inputSizePadded));
        for (size_t group = 0; group < groups; ++group)
        {
            for (size_t kernelIndex = 0; kernelIndex < outputChannels / groups; ++kernelIndex)
            {
                for (size_t d = 0; d < inputChannels / groups; ++d)
                {
                    for (size_t ox = 0; ox < outputSize; ++ox)
                    {
                        for (size_t kx = 0; kx < kernelSize; ++kx)
                        {
                            if (tfStyle)
                            {
                                outputs3D[q][ox][kernelIndex + group * outputChannels / groups] +=
                                    kernelsWeights3D[kx][d][kernelIndex + group * outputChannels / groups] * inputPadded2D[ox * stride + kx * dilation][d + group * inputChannels / groups];
                            }
                            else
                            {
                                outputs3D[q][kernelIndex + group * outputChannels / groups][ox] +=
                                    kernelsWeights3D[kernelIndex + group * outputChannels / groups][d][kx] * inputPadded2D[d + group * inputChannels / groups][ox * stride + kx * dilation];
                            }
                        }
                    }
                }
            }
        }
    }

    if (bias)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t kernelIndex = 0; kernelIndex < outputChannels; ++kernelIndex)
            {
                for (size_t ox = 0; ox < outputSize; ++ox)
                {
                    if (tfStyle)
                    {
                        outputs3D[q][ox][kernelIndex] += bias[kernelIndex];
                    }
                    else
                    {
                        outputs3D[q][kernelIndex][ox] += bias[kernelIndex];
                    }
                }
            }
        }
    }
}

void Common::conv2d(const dtype* input,
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
                    size_t dilationW,
                    size_t dilationH,
                    size_t groups)
{
    auto inputs3D = yato::array_view_3d<dtype>(const_cast<dtype*>(input), yato::dims(batchSize, inputChannels, inputHeight * inputWidth));
    auto outputs3D = yato::array_view_3d<dtype>(output, yato::dims(batchSize, outputChannels, outputHeight * outputWidth));
    auto kernelsWeights4D = yato::array_view_4d<dtype>(const_cast<dtype*>(kernel), yato::dims(outputChannels, inputChannels / groups, kernelHeight, kernelWidth));

    for (size_t q = 0; q < batchSize; ++q)
    {
        for (size_t d = 0; d < outputChannels; ++d)
        {
            fill(outputs3D[q][d].begin(), outputs3D[q][d].end(), 0.0_dt);
        }

        size_t inputWidthPadded = inputWidth + 2 * paddingW;
        size_t inputHeightPadded = inputHeight + 2 * paddingH;

        vector<dtype> inputPadded(inputChannels * inputHeightPadded * inputWidthPadded);

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

void Common::gemm(void*,
                  const Name&,
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
                  dtype)
{
    gemmCPUImpl(transA, transB, m, n, k, alpha, a, b, beta, c);
}

void Common::gemm(void*,
                  const Name&,
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
                  half)
{
    gemmCPUImpl(transA, transB, m, n, k, alpha, a, b, beta, c);
}

void Common::gemm(void* kernelManager,
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
                  size_t aOffset,
                  size_t bOffset,
                  size_t cOffset)
{
    gpu::gemm(*static_cast<OpenCLKernelManager*>(kernelManager), caller, transA, transB, m, n, k, alpha, a, b, beta, c, tmp, aOffset, bOffset, cOffset);
}

void Common::hadamard(OPENBLAS_CONST size_t n,
                      OPENBLAS_CONST dtype alpha,
                      OPENBLAS_CONST dtype* a,
                      OPENBLAS_CONST dtype* x,
                      OPENBLAS_CONST size_t incx,
                      OPENBLAS_CONST dtype beta,
                      dtype* y,
                      OPENBLAS_CONST size_t incy)
{
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    size_t k = 0;
    size_t lda = 1;

    cblas_ssbmv(CblasRowMajor, CblasUpper, static_cast<blasint>(n), static_cast<blasint>(k), alpha, a, static_cast<blasint>(lda), x, static_cast<blasint>(incx), beta, y, static_cast<blasint>(incy));

#else
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; ++i)
    {
        y[i * incy] = alpha * a[i] * x[i * incx] + beta * y[i * incy];
    }
#endif
}

void Common::axpy(void*, const Name&, size_t n, OPENBLAS_CONST dtype sa, OPENBLAS_CONST dtype* sx, size_t incx, dtype* sy, size_t incy, size_t xOffset, size_t yOffset)
{
    axpyCPUImpl(n, sa, sx, incx, sy, incy, xOffset, yOffset);
}

void Common::axpy(void*, const Name&, size_t n, OPENBLAS_CONST dtype sa, OPENBLAS_CONST half* sx, size_t incx, half* sy, size_t incy, size_t xOffset, size_t yOffset)
{
    axpyCPUImpl(n, sa, sx, incx, sy, incy, xOffset, yOffset);
}

void Common::axpy(void* kernelManager, const Name& caller, size_t n, OPENBLAS_CONST dtype sa, const cl::Buffer x, size_t incx, cl::Buffer y, size_t incy, size_t xOffset, size_t yOffset)
{
    gpu::axpy(*static_cast<OpenCLKernelManager*>(kernelManager), caller, n, sa, x, incx, y, incy, xOffset, yOffset);
}

int Common::axpby(OPENBLAS_CONST size_t n,
                  OPENBLAS_CONST dtype alpha,
                  OPENBLAS_CONST dtype* x,
                  OPENBLAS_CONST size_t incx,
                  OPENBLAS_CONST dtype beta,
                  dtype* y,
                  OPENBLAS_CONST size_t incy,
                  size_t xOffset,
                  size_t yOffset)
{
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    cblas_saxpby(static_cast<blasint>(n), alpha, &x[xOffset], static_cast<blasint>(incx), beta, &y[yOffset], static_cast<blasint>(incy));
#else
#if defined(_BLAS_ENHANCE)
    (void)xOffset;
    (void)yOffset;
    CHECK_REQUIREMENT(1 == incx && 1 == incy);
    TensorDesc vDesc = tensor1d(DT_F32, static_cast<U32>(n));
    vector_vector_axpby(alpha, vDesc, x, beta, vDesc, y, ARM_A76);
#else
    size_t indexX = 0;
    size_t indexY = 0;
    for (size_t index = 0; index < n; ++index)
    {
        y[yOffset + indexY] = alpha * x[xOffset + indexX] + beta * y[yOffset + indexY];
        indexX += incx;
        indexY += incy;
    }
#endif
#endif
    return 0;
}

int Common::axpby(OPENBLAS_CONST size_t n,
                  OPENBLAS_CONST dtype alpha,
                  OPENBLAS_CONST half* x,
                  OPENBLAS_CONST size_t incx,
                  OPENBLAS_CONST dtype beta,
                  half* y,
                  OPENBLAS_CONST size_t incy,
                  size_t xOffset,
                  size_t yOffset)
{
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    std::vector<raul::dtype> mX(n * incx + xOffset);
    std::vector<raul::dtype> mY(n * incy + yOffset);

    for (size_t q = xOffset; q < mX.size(); q += incx)
    {
        mX[q] = raul::toFloat32(x[q]);
    }

    for (size_t q = yOffset; q < mY.size(); q += incy)
    {
        mY[q] = raul::toFloat32(y[q]);
    }

    cblas_saxpby(static_cast<blasint>(n), alpha, &mX[xOffset], static_cast<blasint>(incx), beta, &mY[yOffset], static_cast<blasint>(incy));

    for (size_t q = yOffset; q < mY.size(); q += incy)
    {
        y[q] = raul::toFloat16(mY[q]);
    }
#else
#if defined(_BLAS_ENHANCE)
    CHECK_REQUIREMENT(1 == incx && 1 == incy);
    TensorDesc vDesc = tensor1d(DT_F16, static_cast<U32>(n));
    vector_vector_axpby(alpha, vDesc, &x[xOffset], beta, vDesc, &y[yOffset], ARM_A76);
#else
    std::vector<raul::dtype> mX(n * incx + xOffset);
    std::vector<raul::dtype> mY(n * incy + yOffset);

    for (size_t q = xOffset; q < mX.size(); q += incx)
    {
        mX[q] = raul::toFloat32(x[q]);
    }

    for (size_t q = yOffset; q < mY.size(); q += incy)
    {
        mY[q] = raul::toFloat32(y[q]);
    }

    size_t indexX = 0;
    size_t indexY = 0;
    for (size_t index = 0; index < n; ++index)
    {
        mY[yOffset + indexY] = alpha * mX[xOffset + indexX] + beta * mY[yOffset + indexY];
        indexX += incx;
        indexY += incy;
    }

    for (size_t q = yOffset; q < mY.size(); q += incy)
    {
        y[q] = raul::toFloat16(mY[q]);
    }

#endif
#endif
    return 0;
}

void Common::axpby(void* kernelManager,
                   const Name& caller,
                   OPENBLAS_CONST size_t n,
                   OPENBLAS_CONST dtype alpha,
                   const cl::Buffer x,
                   OPENBLAS_CONST size_t incx,
                   OPENBLAS_CONST dtype beta,
                   cl::Buffer y,
                   OPENBLAS_CONST size_t incy,
                   size_t xOffset,
                   size_t yOffset)
{
    gpu::axpby(*static_cast<OpenCLKernelManager*>(kernelManager), caller, n, alpha, x, incx, beta, y, incy, xOffset, yOffset);
}

dtype Common::dot(size_t n, OPENBLAS_CONST dtype* sx, size_t incx, OPENBLAS_CONST dtype* sy, size_t incy)
{
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    return cblas_sdot(static_cast<blasint>(n), sx, static_cast<blasint>(incx), sy, static_cast<blasint>(incy));
#else
    dtype res = 0.0;
    size_t indexX = 0;
    size_t indexY = 0;
    for (size_t index = 0; index < n; ++index)
    {
        res += sx[indexX] * sy[indexY];
        indexX += incx;
        indexY += incy;
    }
    return res;
#endif
}

void Common::scal(size_t n, OPENBLAS_CONST dtype sa, dtype* sx, size_t incx)
{
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    cblas_sscal(static_cast<blasint>(n), sa, sx, static_cast<blasint>(incx));
#else
    size_t indexX = 0;
    for (size_t index = 0; index < n; ++index)
    {
        sx[indexX] *= sa;
        indexX += incx;
    }
#endif
}

void Common::transpose(Tensor& tensor, size_t cols)
{
    ::transpose(&tensor[0], cols, tensor.size());
}

void Common::transpose(TensorGPUHelper&& tensor, size_t cols)
{
    // d.polubotko: implement using GPU

    Tensor t(tensor);
    ::transpose(&t[0], cols, t.size());
    tensor = t;
}

void Common::transpose(TensorFP16& tensor, size_t cols)
{
    ::transpose(&tensor[0], cols, tensor.size());
}

void Common::addPadding2D(const dtype* src, dtype* dst, size_t srcChannels, size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight)
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
                    dst[d * dstWidth * dstHeight + dstWidth * y + x] = 0.0_dt;
                }
            }

            for (size_t y = topPad; y < topPad + srcHeight; ++y)
            {
                // left
                for (size_t x = 0; x < leftPad; ++x)
                {
                    dst[d * dstWidth * dstHeight + dstWidth * y + x] = 0.0_dt;
                }

                // src
                for (size_t x = leftPad; x < leftPad + srcWidth; ++x)
                {
                    dst[d * dstWidth * dstHeight + dstWidth * y + x] = src[d * srcWidth * srcHeight + srcWidth * (y - topPad) + x - leftPad];
                }

                // right
                for (size_t x = leftPad + srcWidth; x < dstWidth; ++x)
                {
                    dst[d * dstWidth * dstHeight + dstWidth * y + x] = 0.0_dt;
                }
            }

            // bottom
            for (size_t y = dstHeight - bottomPad; y < dstHeight; ++y)
            {
                for (size_t x = 0; x < dstWidth; ++x)
                {
                    dst[d * dstWidth * dstHeight + dstWidth * y + x] = 0.0_dt;
                }
            }
        }
    }
}

void Common::addPadding1D(const dtype* src, dtype* dst, size_t srcChannels, size_t srcSize, size_t dstSize, bool reversedOrder)
{
    if (dstSize >= srcSize)
    {
        size_t padSize = dstSize - srcSize;

        size_t leftPad = padSize / 2;

        if (reversedOrder)
        {
            for (size_t d = 0; d < srcChannels; ++d)
            {
                for (size_t x = leftPad; x < leftPad + srcSize; ++x)
                {
                    dst[x * srcChannels + d] = src[(x - leftPad) * srcChannels + d];
                }
            }
        }
        else
        {
            for (size_t d = 0; d < srcChannels; ++d)
            {
                for (size_t x = leftPad; x < leftPad + srcSize; ++x)
                {
                    dst[d * dstSize + x] = src[d * srcSize + x - leftPad];
                }
            }
        }
    }
}

void Common::removePadding2D(const dtype* src, dtype* dst, size_t srcChannels, size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight, bool overwrite)
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

void Common::removePadding1D(const dtype* src, dtype* dst, size_t srcChannels, size_t srcSize, size_t dstSize, bool reversedOrder, bool overwrite)
{
    if (dstSize <= srcSize)
    {
        size_t padSize = srcSize - dstSize;

        size_t leftPad = padSize / 2;
        if (overwrite)
        {
            if (reversedOrder)
            {
                for (size_t d = 0; d < srcChannels; ++d)
                {
                    for (size_t x = 0; x < dstSize; ++x)
                    {
                        dst[x * srcChannels + d] = src[(x + leftPad) * srcChannels + d];
                    }
                }
            }
            else
            {
                for (size_t d = 0; d < srcChannels; ++d)
                {
                    for (size_t x = 0; x < dstSize; ++x)
                    {
                        dst[d * dstSize + x] = src[d * srcSize + x + leftPad];
                    }
                }
            }
        }
        else
        {
            if (reversedOrder)
            {
                for (size_t d = 0; d < srcChannels; ++d)
                {
                    for (size_t x = 0; x < dstSize; ++x)
                    {
                        dst[x * srcChannels + d] += src[(x + leftPad) * srcChannels + d];
                    }
                }
            }
            else
            {
                for (size_t d = 0; d < srcChannels; ++d)
                {
                    for (size_t x = 0; x < dstSize; ++x)
                    {
                        dst[d * dstSize + x] += src[d * srcSize + x + leftPad];
                    }
                }
            }
        }
    }
}

template<typename T>
void Common::im2col(const T* image,
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
                    bool reversedOrder)
{
    // https://github.com/pluskid/Mocha.jl/blob/master/deps/im2col.cpp

    // resulted matrix width (widthCol * heightCol)
    const size_t widthCol = (imageWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;
    const size_t heightCol = (imageHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;

    // resulted matrix height
    const size_t channelsCol = imageChannels * filterWidth * filterHeight;

    for (size_t c = 0; c < channelsCol; ++c)
    {
        size_t w_offset = c % filterWidth;
        size_t h_offset = (c / filterWidth) % filterHeight;
        size_t c_im = c / (filterHeight * filterWidth);

        for (size_t h = 0; h < heightCol; ++h)
        {
            for (size_t w = 0; w < widthCol; ++w)
            {
                long long w_pad = w * strideWidth - paddingWidth + w_offset;
                long long h_pad = h * strideHeight - paddingHeight + h_offset;

                if (h_pad >= 0 && h_pad < static_cast<long long>(imageHeight) && w_pad >= 0 && w_pad < static_cast<long long>(imageWidth))
                {
                    if (reversedOrder)
                    {
                        matrix[(c * heightCol + h) * widthCol + w] = image[w_pad * imageHeight * imageChannels + h_pad * imageChannels + c_im];
                    }
                    else
                    {
                        matrix[(c * heightCol + h) * widthCol + w] = image[c_im * imageHeight * imageWidth + h_pad * imageWidth + w_pad];
                    }
                }
                else
                {
                    matrix[(c * heightCol + h) * widthCol + w] = static_cast<T>(0);
                }
            }
        }
    }
}

size_t Common::im2colOutputSize(size_t imageWidth,
                                size_t imageHeight,
                                size_t imageChannels,
                                size_t filterWidth,
                                size_t filterHeight,
                                size_t strideWidth,
                                size_t strideHeight,
                                size_t paddingWidth,
                                size_t paddingHeight,
                                size_t dilationWidth,
                                size_t dilationHeight)
{
    size_t mEffectiveReceptiveFieldW = dilationWidth * (filterWidth - 1) + 1;
    size_t mEffectiveReceptiveFieldH = dilationHeight * (filterHeight - 1) + 1;
    size_t mOutputWidth = (imageWidth + 2 * paddingWidth - mEffectiveReceptiveFieldW) / strideWidth + 1;
    size_t mOutputHeight = (imageHeight + 2 * paddingHeight - mEffectiveReceptiveFieldH) / strideHeight + 1;

    return mOutputHeight * mOutputWidth * imageChannels * mEffectiveReceptiveFieldH * mEffectiveReceptiveFieldW;
}

template<typename T>
void Common::col2im(const T* matrix,
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
                    bool reversedOrder,
                    bool zeroOutput)
{
    // https://github.com/pluskid/Mocha.jl/blob/master/deps/im2col.cpp

    // input matrix width (widthCol * heightCol)
    size_t widthCol = (imageWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;
    size_t heightCol = (imageHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;

    // input matrix height
    size_t channelsCol = imageChannels * filterHeight * filterWidth;

    if (zeroOutput)
    {
        fill(&image[0], &image[imageChannels * imageHeight * imageWidth], static_cast<T>(0));
    }

    for (size_t c = 0; c < channelsCol; ++c)
    {
        size_t w_offset = c % filterWidth;
        size_t h_offset = (c / filterWidth) % filterHeight;
        size_t c_im = c / (filterHeight * filterWidth);

        for (size_t h = 0; h < heightCol; ++h)
        {
            for (size_t w = 0; w < widthCol; ++w)
            {
                long long w_pad = w * strideWidth - paddingWidth + w_offset;
                long long h_pad = h * strideHeight - paddingHeight + h_offset;

                if (h_pad >= 0 && h_pad < static_cast<long long>(imageHeight) && w_pad >= 0 && w_pad < static_cast<long long>(imageWidth))
                {
                    if (reversedOrder)
                    {
                        image[w_pad * imageHeight * imageChannels + h_pad * imageChannels + c_im] += matrix[(c * heightCol + h) * widthCol + w];
                    }
                    else
                    {
                        image[c_im * imageHeight * imageWidth + h_pad * imageWidth + w_pad] += matrix[(c * heightCol + h) * widthCol + w];
                    }
                }
            }
        }
    }
}

void Common::ReLU(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::ReLU(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::ReLU6(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::ReLU6(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::ReLUBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& delta, TensorGPU& prevDelta)
{
    gpu::ReLUBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, out.getBuffer(), delta.getBuffer(), prevDelta.getBuffer());
}

void Common::ReLU6Backward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& delta, TensorGPU& prevDelta)
{
    gpu::ReLU6Backward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, out.getBuffer(), delta.getBuffer(), prevDelta.getBuffer());
}

dtype Common::GeLU_Erf(dtype x)
{
    return static_cast<dtype>(x * 0.5 * (1.0 + erf(x * RAUL_SQRT1_2)));
}

dtype Common::GeLU_Tanh(dtype x)
{
    return static_cast<dtype>(0.5 * x * (1 + tanh(RAUL_SQRT2_PI * (x + GELU_CONST * pow(x, 3)))));
}

void Common::round(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::round(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::zeroOutput(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& length, TensorGPU& out)
{
    gpu::zeroOutput(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), length.getBuffer(), out.getBuffer());
}

void Common::nonZeroMask(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::nonZeroMask(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::sequenceMask(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& lengths, TensorGPU& mask)
{
    gpu::sequenceMask(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, lengths.getBuffer(), mask.getBuffer());
}

void Common::reverse(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::reverse(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::reverse(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& length, TensorGPU& out)
{
    gpu::reverse(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), length.getBuffer(), out.getBuffer());
}

void Common::reduceTargets(void* kernelManager,
                           const Name& caller,
                           size_t batch,
                           size_t idepth,
                           size_t odepth,
                           size_t iheight,
                           size_t oheight,
                           size_t width,
                           size_t reductionFactor,
                           const TensorGPU& in,
                           TensorGPU& out)
{
    gpu::reduceTargets(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, idepth, odepth, iheight, oheight, width, reductionFactor, in.getBuffer(), out.getBuffer());
}

void Common::copy(void* kernelManager,
                  const Name& caller,
                  size_t sourceLen,
                  size_t destinationLen,
                  size_t sourceOffset,
                  size_t destinationOffset,
                  bool sumWithOldValues,
                  const TensorGPU& in,
                  TensorGPU& out)
{
    gpu::copy(*static_cast<OpenCLKernelManager*>(kernelManager), caller, sourceLen, destinationLen, sourceOffset, destinationOffset, sumWithOldValues, in.getBuffer(), out.getBuffer());
}

void Common::initAlignment(void* kernelManager, const Name& caller, dtype val, size_t batch, size_t height, TensorGPU& out)
{
    gpu::initAlignment(*static_cast<OpenCLKernelManager*>(kernelManager), caller, val, batch, height, out.getBuffer());
}

void Common::gaussianUpsamplingDistributionForward(void* kernelManager,
                                                   const Name& caller,
                                                   size_t batch,
                                                   size_t depth,
                                                   size_t height,
                                                   size_t width,
                                                   const TensorGPU& values,
                                                   const TensorGPU& loc,
                                                   const TensorGPU& scale,
                                                   TensorGPU& out)
{
    gpu::gaussianUpsamplingDistributionForward(
        *static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, values.getBuffer(), loc.getBuffer(), scale.getBuffer(), out.getBuffer());
}

void Common::gaussianUpsamplingDistributionBackward(void* kernelManager,
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
                                                    TensorGPU& prevDelta)
{
    gpu::gaussianUpsamplingDistributionBackward(*static_cast<OpenCLKernelManager*>(kernelManager),
                                                caller,
                                                batch,
                                                depth,
                                                height,
                                                width,
                                                backwardForLoc,
                                                values.getBuffer(),
                                                loc.getBuffer(),
                                                scale.getBuffer(),
                                                deltas.getBuffer(),
                                                prevDelta.getBuffer());
}

void Common::expForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::expForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::expBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::expBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, out.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::sqrtForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::sqrtForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::sqrtBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::sqrtBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, out.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::rsqrtForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::rsqrtForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::rsqrtBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::rsqrtBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, out.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::squareForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::squareForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::squareBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::squareBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::logForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::logForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::logBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::logBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::addBias(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, dtype bias, const TensorGPU& in, TensorGPU& out)
{
    gpu::addBias(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, bias, in.getBuffer(), out.getBuffer());
}

void Common::sigmoidForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::sigmoidForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::sigmoidBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::sigmoidBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, out.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::softplusForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, dtype beta, dtype threshold, const TensorGPU& in, TensorGPU& out)
{
    gpu::softplusForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, beta, threshold, in.getBuffer(), out.getBuffer());
}

void Common::softplusBackward(void* kernelManager,
                              const Name& caller,
                              size_t batch,
                              size_t depth,
                              size_t height,
                              size_t width,
                              dtype beta,
                              dtype threshold,
                              const TensorGPU& out,
                              const TensorGPU& deltas,
                              TensorGPU& prevDelta)
{
    gpu::softplusBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, beta, threshold, out.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::tanhForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::tanhForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::tanhBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::tanhBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, out.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::swishForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::swishForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::swishBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::swishBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::splitterForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::splitterForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::splitterBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::splitterBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::softmaxForward(void* kernelManager, const Name& caller, size_t externalDimSize, size_t internalDimSize, const TensorGPU& in, TensorGPU& out)
{
    gpu::softmaxForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, externalDimSize, internalDimSize, in.getBuffer(), out.getBuffer());
}

void Common::softmaxBackward(void* kernelManager, const Name& caller, size_t externalDimSize, size_t internalDimSize, const TensorGPU& out, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::softmaxBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, externalDimSize, internalDimSize, out.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::geluErfForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::geluErfForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::geluErfBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::geluErfBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::geluTanhForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::geluTanhForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::geluTanhBackward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::geluTanhBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::selectForward(void* kernelManager,
                           const Name& caller,
                           size_t batch,
                           size_t depth,
                           size_t height,
                           size_t width,
                           const TensorGPU& cond,
                           const TensorGPU& in0,
                           const TensorGPU& in1,
                           TensorGPU& out)
{
    gpu::selectForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, cond.getBuffer(), in0.getBuffer(), in1.getBuffer(), out.getBuffer());
}

void Common::selectBackward(void* kernelManager,
                            const Name& caller,
                            size_t index,
                            size_t batch,
                            size_t depth,
                            size_t height,
                            size_t width,
                            const TensorGPU& cond,
                            const TensorGPU& deltas,
                            TensorGPU& prevDelta)
{
    gpu::selectBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, index, batch, depth, height, width, cond.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::dynamicDepthwiseConv2DForward(void* kernelManager,
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
                                           TensorGPU& out)
{
    gpu::dynamicDepthwiseConv2DForward(
        *static_cast<OpenCLKernelManager*>(kernelManager), caller, batchSize, inputC, outputH, outputW, channelMultiplier, filterH, filterW, in0.getBuffer(), in1.getBuffer(), out.getBuffer());
}

void Common::dynamicDepthwiseConv2DBackward(void* kernelManager,
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
                                            TensorGPU& out)
{
    gpu::dynamicDepthwiseConv2DBackward(*static_cast<OpenCLKernelManager*>(kernelManager),
                                        caller,
                                        batchSize,
                                        inputC,
                                        outputH,
                                        outputW,
                                        channelMultiplier,
                                        filterH,
                                        filterW,
                                        isForInput,
                                        in0.getBuffer(),
                                        in1.getBuffer(),
                                        out.getBuffer());
}

void Common::hsigmoidForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::hsigmoidForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::hsigmoidBackward(void* kernelManager, const Name& caller, size_t size, dtype leftDivisor, dtype rightDivisor, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::hsigmoidBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, size, leftDivisor, rightDivisor, in.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

void Common::hswishForward(void* kernelManager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const TensorGPU& in, TensorGPU& out)
{
    gpu::hswishForward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, batch, depth, height, width, in.getBuffer(), out.getBuffer());
}

void Common::hswishBackward(void* kernelManager, const Name& caller, size_t size, dtype a, dtype b, dtype c, const TensorGPU& in, const TensorGPU& deltas, TensorGPU& prevDelta)
{
    gpu::hswishBackward(*static_cast<OpenCLKernelManager*>(kernelManager), caller, size, a, b, c, in.getBuffer(), deltas.getBuffer(), prevDelta.getBuffer());
}

vector<string> Common::split(const string& str, char delimeter)
{
    vector<string> elements;
    ::split(str, delimeter, back_inserter(elements));
    return elements;
}

raul::shape Common::getStrides(const raul::shape& tensor_shape)
{
    raul::shape strides;
    size_t offset = 1U;

    for (ptrdiff_t i = 3; i >= 0; --i)
    {
        strides[i] = (tensor_shape[i] != 1U) ? offset : 0U;
        offset *= tensor_shape[i];
    }
    return strides;
}

raul::shape Common::offsetToIndexes(size_t offset, const raul::shape& strides)
{
    raul::shape indexes;
    size_t q = 0;
    while (q < strides.dimensions_num())
    {
        if (strides[q] != 0)
        {

            indexes[q] = offset / strides[q];
            offset %= strides[q];
        }
        else
        {
            indexes[q] = 0U;
        }
        ++q;
    }
    indexes[q - 1] += offset;
    return indexes;
}

size_t Common::indexesToOffset(const raul::shape& indexes, const raul::shape& strides)
{
    size_t offset = 0;
    for (size_t q = 0; q < indexes.dimensions_num(); q++)
    {
        offset += indexes[q] * strides[q];
    }
    return offset;
}

template void Common::im2col(const dtype* image,
                             size_t imageWidth,
                             size_t imageHeight,
                             size_t imageChannels,
                             size_t filterWidth,
                             size_t filterHeight,
                             size_t strideWidth,
                             size_t strideHeight,
                             size_t paddingWidth,
                             size_t paddingHeight,
                             dtype* matrix,
                             bool reversedOrder);

template void Common::im2col(const half* image,
                             size_t imageWidth,
                             size_t imageHeight,
                             size_t imageChannels,
                             size_t filterWidth,
                             size_t filterHeight,
                             size_t strideWidth,
                             size_t strideHeight,
                             size_t paddingWidth,
                             size_t paddingHeight,
                             half* matrix,
                             bool reversedOrder);

template void Common::col2im(const dtype* matrix,
                             size_t imageWidth,
                             size_t imageHeight,
                             size_t imageChannels,
                             size_t filterWidth,
                             size_t filterHeight,
                             size_t strideWidth,
                             size_t strideHeight,
                             size_t paddingWidth,
                             size_t paddingHeight,
                             dtype* image,
                             bool reversedOrder,
                             bool zeroOutput);

template void Common::col2im(const half* matrix,
                             size_t imageWidth,
                             size_t imageHeight,
                             size_t imageChannels,
                             size_t filterWidth,
                             size_t filterHeight,
                             size_t strideWidth,
                             size_t strideHeight,
                             size_t paddingWidth,
                             size_t paddingHeight,
                             half* image,
                             bool reversedOrder,
                             bool zeroOutput);

} // namespace raul
