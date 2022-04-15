// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GEMM_GPU_H
#define GEMM_GPU_H

#include <training/common/Common.h>
#include <training/opencl/OpenCLKernelManager.h>

namespace raul
{
namespace gpu
{

/*
 * [cols x rows]
 * A[k x m]
 * B[n x k]
 * C[n x m]
 * https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm
 * C = alpha * A * B + beta * C
 */

void gemm(OpenCLKernelManager& manager,
          const Name& caller,
          OPENBLAS_CONST CBLAS_TRANSPOSE transA,
          OPENBLAS_CONST CBLAS_TRANSPOSE transB,
          size_t m,
          size_t n,
          size_t k,
          OPENBLAS_CONST dtype alpha,
          const cl::Buffer d_a,
          const cl::Buffer d_b,
          OPENBLAS_CONST dtype beta,
          cl::Buffer d_c,
          cl::Buffer& d_tmp,
          size_t a_offset = 0,
          size_t b_offset = 0,
          size_t c_offset = 0);

void gemm_padded_b(OpenCLKernelManager& manager,
                   const Name& caller,
                   OPENBLAS_CONST CBLAS_TRANSPOSE transA,
                   size_t m,
                   size_t n,
                   size_t k,
                   OPENBLAS_CONST dtype alpha,
                   const cl::Buffer d_a,
                   const cl::Buffer d_b,
                   OPENBLAS_CONST dtype beta,
                   cl::Buffer d_c,
                   cl::Buffer& d_tmp,
                   size_t a_offset = 0,
                   size_t b_offset = 0,
                   size_t c_offset = 0);

/*
    @brief returns 4 buffer sizes: a_transposed, b_transposed, a_transposed_padded, b_transposed_padded
*/
std::vector<size_t> gemm_temp_buffer_sizes(OPENBLAS_CONST CBLAS_TRANSPOSE transA, OPENBLAS_CONST CBLAS_TRANSPOSE transB, size_t m, size_t n, size_t k);
void gemm_bolt_aligned_sizes(size_t m, size_t n, size_t& aligned_m, size_t& aligned_n);
size_t gemm_temp_buffer_size(OPENBLAS_CONST CBLAS_TRANSPOSE transA, OPENBLAS_CONST CBLAS_TRANSPOSE transB, size_t m, size_t n, size_t k);

void axpy(OpenCLKernelManager& manager, const Name& caller, size_t n, OPENBLAS_CONST dtype sa, const cl::Buffer d_x, size_t, cl::Buffer d_y, size_t, size_t xOffset, size_t yOffset);
void axpby(OpenCLKernelManager& manager,
           const Name& caller,
           size_t n,
           OPENBLAS_CONST dtype sa,
           const cl::Buffer d_x,
           size_t,
           OPENBLAS_CONST dtype sb,
           cl::Buffer d_y,
           size_t,
           size_t xOffset,
           size_t yOffset);

void col2im(OpenCLKernelManager& manager,
            const Name& caller,
            const cl::Buffer matrix,
            size_t imageWidth,
            size_t imageHeight,
            size_t imageChannels,
            size_t filterWidth,
            size_t filterHeight,
            size_t strideWidth,
            size_t strideHeight,
            size_t paddingWidth,
            size_t paddingHeight,
            cl::Buffer image,
            bool reversedOrder = false,
            bool zeroOutput = true,
            size_t matrix_offset = 0,
            size_t image_offset = 0);

void im2col(OpenCLKernelManager& manager,
            const Name& caller,
            const cl::Buffer image,
            size_t imageWidth,
            size_t imageHeight,
            size_t imageChannels,
            size_t filterWidth,
            size_t filterHeight,
            size_t strideWidth,
            size_t strideHeight,
            size_t paddingWidth,
            size_t paddingHeight,
            cl::Buffer matrix,
            bool reversedOrder = false,
            size_t image_offset = 0,
            size_t matrix_offset = 0);

} // raul::gpu namespace
} // raul namespace

#endif // GEMM_GPU_H