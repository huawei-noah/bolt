// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

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

#include <cassert>
#include <fstream>
#include <random>

#ifdef _BLAS_ENHANCE
#include "blas_enhance.h"
#include "thread_affinity.h"
#endif

#include <training/system/TypeHalf.h>

namespace
{

using namespace std;
using namespace raul;

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
               const dtype alpha,
               const dtype beta,
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
                    float acc = 0.f;
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
                    float acc = 0.f;
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
                    float acc = 0.f;
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
                    float acc = 0.f;
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
                 OPENBLAS_CONST dtype alpha,
                 OPENBLAS_CONST dtype* a,
                 OPENBLAS_CONST dtype* b,
                 OPENBLAS_CONST dtype beta,
                 dtype* c)
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
            memset(static_cast<void*>(c), 0, tensorNumBytes(matrixCDesc));
        }
        CHECK_STATUS(matrix_vector_multiply(matrixBDesc, b + bOffset, matrixADesc, a, 0, nullptr, matrixCDesc, c, nullptr, get_cpu_arch()));
    }
    else if (1 == n && transA == CblasNoTrans)
    {
        matrixBDesc = tensor1d(DT_F32, static_cast<U32>(k));
        TensorDesc matrixCDesc = tensor1d(DT_F32, static_cast<U32>(m));
        if (0 == beta)
        {
            memset(static_cast<void*>(c), 0, tensorNumBytes(matrixCDesc));
        }
        CHECK_STATUS(matrix_vector_multiply(matrixADesc, a, matrixBDesc, b + bOffset, 0, nullptr, matrixCDesc, c, nullptr, get_cpu_arch()));
    }
    else
    {
        TensorDesc matrixCDesc = tensor2df(DT_F32, DF_NORMAL, static_cast<U32>(m), static_cast<U32>(n));
        unsigned int bytes;
        CHECK_STATUS(matrix_matrix_multiply_tmp_bytes(matrixADesc, matrixBDesc, &bytes, get_cpu_arch()));
        vector<char> tmp(bytes);
        if (0 == beta)
        {
            memset(static_cast<void*>(c), 0, tensorNumBytes(matrixCDesc));
        }
        CHECK_STATUS(matrix_matrix_multiply(matrixADesc, a, matrixBDesc, b + bOffset, bytes, tmp.data(), matrixCDesc, c, nullptr, get_cpu_arch()));
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
                 OPENBLAS_CONST dtype alpha,
                 OPENBLAS_CONST half* a,
                 OPENBLAS_CONST half* b,
                 OPENBLAS_CONST dtype beta,
                 half* c)
{
    size_t bOffset = 0;
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    size_t lda = (transA == CblasNoTrans) ? k : m;
    size_t ldb = (transB == CblasNoTrans) ? n : k;

    std::vector<dtype> mA(k * m);
    std::vector<dtype> mB(n * k + bOffset);
    std::vector<dtype> mC(n * m);

    for (size_t q = 0; q < k * m; ++q)
    {
        mA[q] = toFloat32(a[q]);
    }

    for (size_t q = bOffset; q < n * k; ++q)
    {
        mB[q] = toFloat32(b[q]);
    }

    for (size_t q = 0; q < n * m; ++q)
    {
        mC[q] = toFloat32(c[q]);
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
        c[q] = toFloat16(mC[q]);
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
        CHECK_STATUS(matrix_vector_multiply(matrixBDesc, b + bOffset, matrixADesc, a, 0, nullptr, matrixCDesc, c, nullptr, get_cpu_arch()));
    }
    else if (1 == n && transA == CblasNoTrans)
    {
        matrixBDesc = tensor1d(DT_F16, static_cast<U32>(k));
        TensorDesc matrixCDesc = tensor1d(DT_F16, static_cast<U32>(m));
        if (0 == beta)
        {
            memset(c, 0, tensorNumBytes(matrixCDesc));
        }
        CHECK_STATUS(matrix_vector_multiply(matrixADesc, a, matrixBDesc, b + bOffset, 0, nullptr, matrixCDesc, c, nullptr, get_cpu_arch()));
    }
    else
    {
        TensorDesc matrixCDesc = tensor2df(DT_F16, DF_NORMAL, static_cast<U32>(m), static_cast<U32>(n));
        unsigned int bytes;
        CHECK_STATUS(matrix_matrix_multiply_tmp_bytes(matrixADesc, matrixBDesc, &bytes, get_cpu_arch()));
        vector<char> tmp(bytes);
        if (0 == beta)
        {
            memset(c, 0, tensorNumBytes(matrixCDesc));
        }
        CHECK_STATUS(matrix_matrix_multiply(matrixADesc, a, matrixBDesc, b + bOffset, bytes, tmp.data(), matrixCDesc, c, nullptr, get_cpu_arch()));
    }

    if (alpha != 1_dt)
    {
        size_t sizeC = m * n;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < sizeC; ++i)
        {
            c[i] = static_cast<half>(static_cast<float>(c[i]) * alpha);
        }
    }
#else
    size_t lda = (transA == CblasNoTrans) ? k : m;
    size_t ldb = (transB == CblasNoTrans) ? n : k;

    std::vector<dtype> mA(k * m);
    std::vector<dtype> mB(n * k + bOffset);
    std::vector<dtype> mC(n * m);

    for (size_t q = 0; q < k * m; ++q)
    {
        mA[q] = toFloat32(a[q]);
    }

    for (size_t q = bOffset; q < n * k; ++q)
    {
        mB[q] = toFloat32(b[q]);
    }

    for (size_t q = 0; q < n * m; ++q)
    {
        mC[q] = toFloat32(c[q]);
    }

    matrixMul(transA, transB, mA.data(), mB.data(), mC.data(), m, n, k, alpha, beta, lda, ldb, n, bOffset);

    for (size_t q = 0; q < n * m; ++q)
    {
        c[q] = toFloat16(mC[q]);
    }
#endif
#endif
}

void axpyCPUImpl(size_t n, OPENBLAS_CONST dtype sa, OPENBLAS_CONST dtype* sx, size_t incx, dtype* sy, size_t incy, size_t xOffset, size_t yOffset)
{
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    cblas_saxpy(static_cast<blasint>(n), sa, &sx[xOffset], static_cast<blasint>(incx), &sy[yOffset], static_cast<blasint>(incy));
#else
#if defined(_BLAS_ENHANCE)
    CHECK_REQUIREMENT(1 == incx && 1 == incy);
    TensorDesc vDesc = tensor1d(DT_F32, static_cast<U32>(n));
    CHECK_STATUS(vector_vector_axpby(sa, vDesc, &sx[xOffset], 1, vDesc, &sy[yOffset], get_cpu_arch()));
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

void axpyCPUImpl(size_t n, OPENBLAS_CONST dtype sa, OPENBLAS_CONST half* sx, size_t incx, half* sy, size_t incy, size_t xOffset, size_t yOffset)
{
#if defined(_BLAS) && !defined(_BLAS_ENHANCE)
    std::vector<dtype> mSX(n * incx + xOffset);
    std::vector<dtype> mSY(n * incy + yOffset);

    for (size_t q = xOffset; q < mSX.size(); q += incx)
    {
        mSX[q] = toFloat32(sx[q]);
    }

    for (size_t q = yOffset; q < mSY.size(); q += incy)
    {
        mSY[q] = toFloat32(sy[q]);
    }

    cblas_saxpy(static_cast<blasint>(n), sa, &mSX[xOffset], static_cast<blasint>(incx), &mSY[yOffset], static_cast<blasint>(incy));

    for (size_t q = yOffset; q < mSY.size(); q += incy)
    {
        sy[q] = toFloat16(mSY[q]);
    }
#else
#if defined(_BLAS_ENHANCE)
    CHECK_REQUIREMENT(1 == incx && 1 == incy);
    TensorDesc vDesc = tensor1d(DT_F16, static_cast<U32>(n));
    CHECK_STATUS(vector_vector_axpby(sa, vDesc, &sx[xOffset], 1, vDesc, &sy[yOffset], get_cpu_arch()));
#else
    std::vector<dtype> mSX(n * incx + xOffset);
    std::vector<dtype> mSY(n * incy + yOffset);

    for (size_t q = xOffset; q < mSX.size(); q += incx)
    {
        mSX[q] = toFloat32(sx[q]);
    }

    for (size_t q = yOffset; q < mSY.size(); q += incy)
    {
        mSY[q] = toFloat32(sy[q]);
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
        sy[q] = toFloat16(mSY[q]);
    }
#endif
#endif
}
} // anonymous namespace

namespace raul
{

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

void Common::generate_permutation(size_t n, vector<size_t>& ind_vector, unsigned int seed)
{
    ind_vector.clear();
    ind_vector.resize(n);
    std::iota(ind_vector.begin(), ind_vector.end(), size_t(0));

    std::mt19937 g(seed);

    std::shuffle(ind_vector.begin(), ind_vector.end(), g);
}

void Common::gemm(OPENBLAS_CONST CBLAS_TRANSPOSE transA,
                  OPENBLAS_CONST CBLAS_TRANSPOSE transB,
                  size_t m,
                  size_t n,
                  size_t k,
                  OPENBLAS_CONST dtype alpha,
                  OPENBLAS_CONST dtype* a,
                  OPENBLAS_CONST dtype* b,
                  OPENBLAS_CONST dtype beta,
                  dtype* c)
{
    gemmCPUImpl(transA, transB, m, n, k, alpha, a, b, beta, c);
}

void Common::gemm(OPENBLAS_CONST CBLAS_TRANSPOSE transA,
                  OPENBLAS_CONST CBLAS_TRANSPOSE transB,
                  size_t m,
                  size_t n,
                  size_t k,
                  OPENBLAS_CONST dtype alpha,
                  OPENBLAS_CONST half* a,
                  OPENBLAS_CONST half* b,
                  OPENBLAS_CONST dtype beta,
                  half* c)
{
    gemmCPUImpl(transA, transB, m, n, k, alpha, a, b, beta, c);
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

void Common::axpy(size_t n, OPENBLAS_CONST dtype sa, OPENBLAS_CONST dtype* sx, size_t incx, dtype* sy, size_t incy, size_t xOffset, size_t yOffset)
{
    axpyCPUImpl(n, sa, sx, incx, sy, incy, xOffset, yOffset);
}

void Common::axpy(size_t n, OPENBLAS_CONST dtype sa, OPENBLAS_CONST half* sx, size_t incx, half* sy, size_t incy, size_t xOffset, size_t yOffset)
{
    axpyCPUImpl(n, sa, sx, incx, sy, incy, xOffset, yOffset);
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
    CHECK_STATUS(vector_vector_axpby(alpha, vDesc, x, beta, vDesc, y, get_cpu_arch()));
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
    std::vector<dtype> mX(n * incx + xOffset);
    std::vector<dtype> mY(n * incy + yOffset);

    for (size_t q = xOffset; q < mX.size(); q += incx)
    {
        mX[q] = toFloat32(x[q]);
    }

    for (size_t q = yOffset; q < mY.size(); q += incy)
    {
        mY[q] = toFloat32(y[q]);
    }

    cblas_saxpby(static_cast<blasint>(n), alpha, &mX[xOffset], static_cast<blasint>(incx), beta, &mY[yOffset], static_cast<blasint>(incy));

    for (size_t q = yOffset; q < mY.size(); q += incy)
    {
        y[q] = toFloat16(mY[q]);
    }
#else
#if defined(_BLAS_ENHANCE)
    CHECK_REQUIREMENT(1 == incx && 1 == incy);
    TensorDesc vDesc = tensor1d(DT_F16, static_cast<U32>(n));
    CHECK_STATUS(vector_vector_axpby(alpha, vDesc, &x[xOffset], beta, vDesc, &y[yOffset], get_cpu_arch()));
#else
    std::vector<dtype> mX(n * incx + xOffset);
    std::vector<dtype> mY(n * incy + yOffset);

    for (size_t q = xOffset; q < mX.size(); q += incx)
    {
        mX[q] = toFloat32(x[q]);
    }

    for (size_t q = yOffset; q < mY.size(); q += incy)
    {
        mY[q] = toFloat32(y[q]);
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
        y[q] = toFloat16(mY[q]);
    }

#endif
#endif
    return 0;
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

void Common::transpose(TensorFP16& tensor, size_t cols)
{
    ::transpose(&tensor[0], cols, tensor.size());
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

dtype Common::GeLU_Erf(dtype x)
{
    return static_cast<dtype>(x * 0.5 * (1.0 + erf(x * RAUL_SQRT1_2)));
}

dtype Common::GeLU_Tanh(dtype x)
{
    return static_cast<dtype>(0.5 * x * (1 + tanh(RAUL_SQRT2_PI * (x + GELU_CONST * pow(x, 3)))));
}

vector<string> Common::split(const string& str, char delimeter)
{
    vector<string> elements;
    ::split(str, delimeter, back_inserter(elements));
    return elements;
}

shape Common::getStrides(const shape& tensor_shape)
{
    shape strides;
    size_t offset = 1U;

    for (ptrdiff_t i = 3; i >= 0; --i)
    {
        strides[i] = (tensor_shape[i] != 1U) ? offset : 0U;
        offset *= tensor_shape[i];
    }
    return strides;
}

shape Common::offsetToIndexes(size_t offset, const shape& strides)
{
    shape indexes;
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

size_t Common::indexesToOffset(const shape& indexes, const shape& strides)
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
