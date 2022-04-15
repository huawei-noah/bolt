// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GemmGPU.h"
#include <training/common/Common.h>
#include <training/common/OpenclInitializer.h>

using namespace std;

namespace
{
using namespace raul;

inline size_t memory_aligned_size(size_t memorySize, size_t alignmentSize)
{
    return memorySize + ((alignmentSize - (memorySize % alignmentSize)) % alignmentSize);
}

cl::Kernel getTransposeKernel(OpenCLKernelManager& manager)
{
    string name = "transpose_nchw";

    if (!manager.hasKernel(name))
    {
        string source =
#include "kernels/transpose_nchw.cl"
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, "getTransposeKernel");
}

cl::Kernel getPaddingKernel(OpenCLKernelManager& manager)
{
    string name = "padding_nchw_constant";

    if (!manager.hasKernel(name))
    {
        string source =
#include "kernels/padding_nchw.cl"
            ;
        manager.registerProgram(name, source, "-DUSE_CONSTANT");
    }

    return manager.getKernel(name, "getPaddingKernel");
}

cl::Kernel getGemmKernel(OpenCLKernelManager& manager, size_t LM, size_t LN, size_t vectorization = 1)
{
    string name = "gemm_tn_nobias_" + to_string(LM) + to_string(LN);

    if (!manager.hasKernel(name))
    {
        string source =
#include "kernels/gemm_tn.cl"
            ;
        string opt = "-DLM=" + to_string(LM) + " -DLN=" + to_string(LN) + " -DUSE_V" + to_string(vectorization) + " -DNO_BIAS";
        manager.registerProgram(name, source, opt);
    }

    return manager.getKernel(name, "getGemmKernel");
    ;
}

cl::Kernel getAxpyKernel(OpenCLKernelManager& manager)
{
    string name = "axpy";

    if (!manager.hasKernel(name))
    {
        string source =
#include "kernels/axpy.cl"
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, "getAxpyKernel");
}

cl::Kernel getAxpbyKernel(OpenCLKernelManager& manager)
{
    string name = "axpby";

    if (!manager.hasKernel(name))
    {
        string source =
#include "kernels/axpby.cl"
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, "getAxpbyKernel");
}

cl::Kernel getCol2ImKernel(OpenCLKernelManager& manager, bool reversed)
{
    string pname = "col2im";
    string name = reversed ? "col2im_reversed" : "col2im";

    if (!manager.hasKernel(pname, name))
    {
        string source =
#include "kernels/col2im.cl"
            ;
        manager.registerProgram(pname, source);
    }

    return manager.getKernel(pname, name, "getCol2ImKernel");
}

cl::Kernel getIm2ColKernel(OpenCLKernelManager& manager, bool reversed)
{
    string pname = "im2col";
    string name = reversed ? "im2col_reversed" : "im2col";

    if (!manager.hasKernel(pname, name))
    {
        string source =
#include "kernels/im2col.cl"
            ;
        manager.registerProgram(pname, source);
    }

    return manager.getKernel(pname, name, "getIm2ColKernel");
}

}

namespace raul
{
namespace gpu
{

void gemm_bolt_aligned_sizes(size_t m, size_t n, size_t& aligned_m, size_t& aligned_n)
{
    aligned_m = m;
    aligned_n = n;
    for (size_t i = 1; i <= 8; i++)
    {
        size_t j = (m + i - 1) / i * i;
        if (aligned_m < j)
        {
            aligned_m = j;
        }
        j = (n + i - 1) / i * i;
        if (aligned_n < j)
        {
            aligned_n = j;
        }
    }
}

std::vector<size_t> gemm_temp_buffer_sizes(OPENBLAS_CONST CBLAS_TRANSPOSE transA, OPENBLAS_CONST CBLAS_TRANSPOSE transB, size_t m, size_t n, size_t k)
{
    std::vector<size_t> v(4, 0);

    static size_t alignment = 0;
    if (alignment == 0)
    {
        auto [p, d, c] = Common::getGpuPlatformDeviceAndContext();
        alignment = d.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() / 8;
    }
    if (transA == CblasNoTrans)
    {
        v[0] = memory_aligned_size(m * k * sizeof(dtype), alignment);
    }
    if (transB == CblasTrans)
    {
        v[1] = memory_aligned_size(n * k * sizeof(dtype), alignment);
    }

    size_t m_align = 0;
    size_t n_align = 0;
    gemm_bolt_aligned_sizes(m, n, m_align, n_align);

    if (m_align != m)
    {
        v[2] = memory_aligned_size(m_align * k * sizeof(dtype), alignment);
    }

    if (n_align != n)
    {
        v[3] = memory_aligned_size(n_align * k * sizeof(dtype), alignment);
    }

    return v;
}

// buffer for TN kernel
size_t gemm_temp_buffer_size(OPENBLAS_CONST CBLAS_TRANSPOSE transA, OPENBLAS_CONST CBLAS_TRANSPOSE transB, size_t m, size_t n, size_t k)
{
    size_t tmpSize = 0;

    auto v = gemm_temp_buffer_sizes(transA, transB, m, n, k);

    tmpSize = std::accumulate(v.begin(), v.end(), tmpSize);

    return tmpSize;
}

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
          cl::Buffer& d_tmp0,
          size_t a_offset,
          size_t b_offset,
          size_t c_offset)
{
    auto queue = manager.getCommandQueue();
    cl_int status = CL_SUCCESS;
    auto context = queue.getInfo<CL_QUEUE_CONTEXT>(&status);
    Common::checkOpenCLStatus(status, caller, "Error quering context from queue");

    auto tmpSizes = gemm_temp_buffer_sizes(transA, transB, m, n, k);
    auto tmpSize = std::accumulate(tmpSizes.begin(), tmpSizes.end(), size_t(0));
    size_t bufSize = 0;
    cl::Buffer& d_tmp = d_tmp0;
    if (tmpSize > 0)
    {
        if (d_tmp0())
        {
            // d_tmp = d_tmp0;
            bufSize = d_tmp.getInfo<CL_MEM_SIZE>(&status);
            Common::checkOpenCLStatus(status, caller, "Error quering buffer size");
        }
        if (bufSize < tmpSize)
        {
            throw runtime_error("[gpu::gemm(" + caller + ")]: Provided buffer size (" + to_string(bufSize) + ") is less then required minimum (" + to_string(tmpSize) + ")");
        }
    }

    // alignment
    size_t m_align = 0;
    size_t n_align = 0;
    size_t batch = 1;
    gemm_bolt_aligned_sizes(m, n, m_align, n_align);

    if (beta == 0)
    {
        float fill = 0.f;

        Common::checkOpenCLStatus(queue.enqueueFillBuffer(d_c, fill, c_offset * sizeof(dtype), m * n * sizeof(dtype)), caller, "Error zeroing matrix C");
    }

    cl::Buffer d_a_padded, d_b_padded;
    if (m_align != m)
    {
        d_a_padded = manager.createSubBuffer(d_tmp, tmpSizes[0] + tmpSizes[1], tmpSizes[2], caller);
    }
    if (n_align != n)
    {
        d_b_padded = manager.createSubBuffer(d_tmp, tmpSizes[0] + tmpSizes[1] + tmpSizes[2], tmpSizes[3], caller);
    }

    cl::Buffer d_a_transposed = d_a;
    cl::Buffer d_b_transposed = d_b;

    if (transA == CblasNoTrans || transB == CblasTrans)
    {
        auto kernelTranspose = getTransposeKernel(manager);

        if (transA == CblasNoTrans)
        {
            auto d_a_t = manager.createSubBuffer(d_tmp, 0, tmpSizes[0], caller);

            cl::NDRange workSize{ (k + 3) / 4, m, batch };

            cl_int dimTran[3] = { 1, 0, 2 };

            manager.callKernel(kernelTranspose,
                               workSize,
                               caller / "transpose_a",
                               (cl_int)k,
                               (cl_int)m,
                               (cl_int)a_offset,
                               0,
                               (cl_int)m,
                               (cl_int)k,
                               0,
                               0,
                               dimTran[0],
                               dimTran[1],
                               dimTran[2],
                               (cl_int)k,
                               (cl_int)workSize[0],
                               (cl_int)workSize[1],
                               d_a(),
                               d_a_t());

            d_a_transposed = d_a_t;
            a_offset = 0;
        }
        if (transB == CblasTrans)
        {
            auto d_b_t = manager.createSubBuffer(d_tmp, tmpSizes[0], tmpSizes[1], caller);

            cl::NDRange workSize{ (k + 3) / 4, n, batch };

            cl_int dimTran[3] = { 1, 0, 2 };

            manager.callKernel(kernelTranspose,
                               workSize,
                               caller / "transpose_b",
                               (cl_int)k,
                               (cl_int)n,
                               (cl_int)b_offset,
                               0,
                               (cl_int)n,
                               (cl_int)k,
                               0,
                               0,
                               dimTran[0],
                               dimTran[1],
                               dimTran[2],
                               (cl_int)k,
                               (cl_int)workSize[0],
                               (cl_int)workSize[1],
                               d_b(),
                               d_b_t());

            d_b_transposed = d_b_t;
            b_offset = 0;
        }
    }

    // padding
    cl::Buffer matrixAPadded = d_a_transposed;
    cl::Buffer matrixBPadded = d_b_transposed;

    if (m_align != m || n_align != n)
    {
        auto kernelPad = getPaddingKernel(manager);

        if (m_align != m)
        {
            cl::NDRange workSize{ (m_align + 3) / 4, k, batch };
            manager.callKernel(kernelPad,
                               workSize,
                               caller / "pad_a",
                               (cl_int)m,
                               (cl_int)k,
                               (cl_int)a_offset,
                               0,
                               (cl_int)m_align,
                               (cl_int)k,
                               0,
                               0,
                               (cl_int)m,
                               (cl_int)k,
                               (cl_int)m_align,
                               (cl_int)k,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               (cl_int)workSize[0],
                               (cl_int)workSize[1],
                               d_a_transposed(),
                               d_a_padded());
            matrixAPadded = d_a_padded;
            a_offset = 0;
        }
        if (n_align != n)
        {
            cl::NDRange workSize{ (n_align + 3) / 4, k, batch };
            manager.callKernel(kernelPad,
                               workSize,
                               caller / "pad_b",
                               (cl_int)n,
                               (cl_int)k,
                               (cl_int)b_offset,
                               0,
                               (cl_int)n_align,
                               (cl_int)k,
                               0,
                               0,
                               (cl_int)n,
                               (cl_int)k,
                               (cl_int)n_align,
                               (cl_int)k,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               (cl_int)workSize[0],
                               (cl_int)workSize[1],
                               d_b_transposed(),
                               d_b_padded());
            matrixBPadded = d_b_padded;
            b_offset = 0;
        }
    }

    size_t a_str = m_align * k;
    size_t b_str = n_align * k;
    size_t c_str = m * n;

    /*item_m: calculate points number on matrix A for each thread*/
    /*item_n: calculate points number on matrix B for each thread*/
    /*each thread calculate item_m * item_n points on matrix C*/
    size_t item_m = 4;
    size_t item_n = 4;
    if (manager.getExecutionPolicy() == KernelExecutionPolicy::SelectBestParams)
    {
        auto& info = (*manager.getExecutionProfile())[caller / "gemm"];
        size_t best_item_m = 0;
        size_t best_item_n = 0;
        float bestTime = FLT_MAX;
        cl::Buffer tmp_c;

        if (beta != 0)
        {
            tmp_c = manager.createBuffer(m * n * sizeof(dtype), caller / "gemm" / "create_tmp_c");
            manager.copyBuffer(d_c, tmp_c, caller / "gemm" / "store_c");
            Common::checkOpenCLStatus(queue.finish(), caller, "error saving matrix c");
        }

        {

            for (item_m = 1; item_m <= 8; item_m++)
            {
                for (item_n = 1; item_n <= 8; item_n++)
                {
                    if (item_m * item_n == 1)
                    {
                        continue;
                    }

                    auto kernelGemm = getGemmKernel(manager, item_m, item_n);

                    cl::NDRange workSize{ (n + item_n - 1) / item_n, (m + item_m - 1) / item_m, batch };

                    manager.callKernel(kernelGemm,
                                       workSize,
                                       caller / "gemm",
                                       (cl_int)m_align,
                                       (cl_int)n_align,
                                       (cl_int)k,
                                       (cl_int)n,
                                       (cl_int)a_str,
                                       (cl_int)b_str,
                                       (cl_int)c_str,
                                       (cl_int)a_offset,
                                       (cl_int)b_offset,
                                       (cl_int)n,
                                       (cl_int)m,
                                       (cl_int)workSize[0],
                                       (cl_int)workSize[1],
                                       alpha,
                                       beta,
                                       matrixAPadded(),
                                       matrixBPadded(),
                                       d_c(),
                                       (cl_int)c_offset);
                    if (info.BestTimeNS < bestTime)
                    {
                        bestTime = info.BestTimeNS;
                        best_item_m = item_m;
                        best_item_n = item_n;
                        info.Data = to_string(item_m) + "_" + to_string(item_n);
                    }
                }
            }
            item_m = best_item_m;
            item_n = best_item_n;
        }

        if (beta != 0)
        {
            manager.copyBuffer(tmp_c, d_c, caller / "gemm" / "restore_c");
            Common::checkOpenCLStatus(queue.finish(), caller, "Error restoring matrix c");
        }
    }
    else if (manager.getExecutionPolicy() == KernelExecutionPolicy::ProfiledParams)
    {
        const auto& info = (*manager.getExecutionProfile())[caller / "gemm"];
        if (!info.Data.empty())
        {
            auto pos = info.Data.find("_");
            if (pos != string::npos)
            {
                item_m = stoi(info.Data.substr(0, pos));
                item_n = stoi(info.Data.substr(pos + 1));
            }
        }
    }
    auto kernelGemm = getGemmKernel(manager, item_m, item_n);

    cl::NDRange workSize{ (n + item_n - 1) / item_n, (m + item_m - 1) / item_m, batch };
    auto storedPolicy = manager.getExecutionPolicy();
    if (storedPolicy == KernelExecutionPolicy::SelectBestParams)
    {
        manager.setExecutionPolicy(KernelExecutionPolicy::DefaultParams);
    }

    manager.callKernel(kernelGemm,
                       workSize,
                       caller / "gemm",
                       (cl_int)m_align,
                       (cl_int)n_align,
                       (cl_int)k,
                       (cl_int)n,
                       (cl_int)a_str,
                       (cl_int)b_str,
                       (cl_int)c_str,
                       (cl_int)a_offset,
                       (cl_int)b_offset,
                       (cl_int)n,
                       (cl_int)m,
                       (cl_int)workSize[0],
                       (cl_int)workSize[1],
                       alpha,
                       beta,
                       matrixAPadded(),
                       matrixBPadded(),
                       d_c(),
                       (cl_int)c_offset);

    manager.setExecutionPolicy(storedPolicy);
}

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
                   cl::Buffer& d_tmp0,
                   size_t a_offset,
                   size_t b_offset,
                   size_t c_offset)
{
    auto queue = manager.getCommandQueue();
    cl_int status = CL_SUCCESS;
    auto context = queue.getInfo<CL_QUEUE_CONTEXT>(&status);
    Common::checkOpenCLStatus(status, caller, "Error quering context from queue");

    auto tmpSizes = gemm_temp_buffer_sizes(transA, CblasNoTrans, m, n, k);
    auto tmpSize = tmpSizes[0] + tmpSizes[2];
    size_t bufSize = 0;
    cl::Buffer& d_tmp = d_tmp0;
    if (tmpSize > 0)
    {
        if (d_tmp0())
        {
            // d_tmp = d_tmp0;
            bufSize = d_tmp.getInfo<CL_MEM_SIZE>(&status);
            Common::checkOpenCLStatus(status, caller, "Error quering buffer size");
        }
        if (bufSize < tmpSize)
        {
            throw runtime_error("[gpu::gemm(" + caller + ")]: Provided buffer size (" + to_string(bufSize) + ") is less then required minimum (" + to_string(tmpSize) + ")");
        }
    }

    // alignment
    size_t m_align = 0;
    size_t n_align = 0;
    size_t batch = 1;
    gemm_bolt_aligned_sizes(m, n, m_align, n_align);

    if (beta == 0)
    {
        float fill = 0.f;

        Common::checkOpenCLStatus(queue.enqueueFillBuffer(d_c, fill, c_offset * sizeof(dtype), m * n * sizeof(dtype)), caller, "Error zeroing matrix C");
    }

    cl::Buffer d_a_padded;
    if (m_align != m)
    {
        cl_buffer_region rgn = { tmpSizes[0] + tmpSizes[1], tmpSizes[2] };
        d_a_padded = clCreateSubBuffer(d_tmp(), CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rgn, &status);
        Common::checkOpenCLStatus(status,
                                  caller,
                                  "error creating sub-buffer for a-padded (size " + to_string(rgn.size) + " bytes, offset " + to_string(rgn.origin) + " bytes) from buffer of size " +
                                      to_string(bufSize) + " bytes");
    }

    cl::Buffer d_a_transposed = d_a;

    if (transA == CblasNoTrans)
    {
        auto kernelTranspose = getTransposeKernel(manager);

        auto d_a_t = manager.createSubBuffer(d_tmp, 0, tmpSizes[0], caller);

        cl::NDRange workSize{ (k + 3) / 4, m, batch };

        cl_int dimTran[3] = { 1, 0, 2 };

        manager.callKernel(kernelTranspose,
                           workSize,
                           caller / "transpose_a",
                           (cl_int)k,
                           (cl_int)m,
                           (cl_int)a_offset,
                           0,
                           (cl_int)m,
                           (cl_int)k,
                           0,
                           0,
                           dimTran[0],
                           dimTran[1],
                           dimTran[2],
                           (cl_int)k,
                           (cl_int)workSize[0],
                           (cl_int)workSize[1],
                           d_a(),
                           d_a_t());

        d_a_transposed = d_a_t;
        a_offset = 0;
    }

    // padding
    cl::Buffer matrixAPadded = d_a_transposed;
    cl::Buffer matrixBPadded = d_b;

    if (m_align != m)
    {
        auto kernelPad = getPaddingKernel(manager);

        if (m_align != m)
        {
            cl::NDRange workSize{ (m_align + 3) / 4, k, batch };
            manager.callKernel(kernelPad,
                               workSize,
                               caller / "pad_a",
                               (cl_int)m,
                               (cl_int)k,
                               (cl_int)a_offset,
                               0,
                               (cl_int)m_align,
                               (cl_int)k,
                               0,
                               0,
                               (cl_int)m,
                               (cl_int)k,
                               (cl_int)m_align,
                               (cl_int)k,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               (cl_int)workSize[0],
                               (cl_int)workSize[1],
                               d_a_transposed(),
                               d_a_padded());
            matrixAPadded = d_a_padded;
            a_offset = 0;
        }
    }

    size_t a_str = m_align * k;
    size_t b_str = n_align * k;
    size_t c_str = m * n;

    /*item_m: calculate points number on matrix A for each thread*/
    /*item_n: calculate points number on matrix B for each thread*/
    /*each thread calculate item_m * item_n points on matrix C*/
    size_t item_m = 4;
    size_t item_n = 4;
    if (manager.getExecutionPolicy() == KernelExecutionPolicy::SelectBestParams)
    {
        auto& info = (*manager.getExecutionProfile())[caller / "gemm"];
        size_t best_item_m = 0;
        size_t best_item_n = 0;
        float bestTime = FLT_MAX;
        cl::Buffer tmp_c;

        if (beta != 0)
        {
            tmp_c = manager.createBuffer(m * n * sizeof(dtype), caller / "gemm" / "create_tmp_c");
            Common::checkOpenCLStatus(status, caller, "Error creating buffer with size " + to_string(m * n * sizeof(dtype)) + " bytes");
            manager.copyBuffer(d_c, tmp_c, caller / "gemm" / "store_c");
            Common::checkOpenCLStatus(queue.finish(), caller, "error saving matrix c");
        }

        {

            for (item_m = 1; item_m <= 8; item_m++)
            {
                for (item_n = 1; item_n <= 8; item_n++)
                {
                    if (item_m * item_n == 1)
                    {
                        continue;
                    }

                    auto kernelGemm = getGemmKernel(manager, item_m, item_n);

                    cl::NDRange workSize{ (n + item_n - 1) / item_n, (m + item_m - 1) / item_m, batch };

                    manager.callKernel(kernelGemm,
                                       workSize,
                                       caller / "gemm",
                                       (cl_int)m_align,
                                       (cl_int)n_align,
                                       (cl_int)k,
                                       (cl_int)n,
                                       (cl_int)a_str,
                                       (cl_int)b_str,
                                       (cl_int)c_str,
                                       (cl_int)a_offset,
                                       (cl_int)b_offset,
                                       (cl_int)n,
                                       (cl_int)m,
                                       (cl_int)workSize[0],
                                       (cl_int)workSize[1],
                                       alpha,
                                       beta,
                                       matrixAPadded(),
                                       matrixBPadded(),
                                       d_c(),
                                       (cl_int)c_offset);
                    if (info.BestTimeNS < bestTime)
                    {
                        bestTime = info.BestTimeNS;
                        best_item_m = item_m;
                        best_item_n = item_n;
                        info.Data = to_string(item_m) + "_" + to_string(item_n);
                    }
                }
            }
            item_m = best_item_m;
            item_n = best_item_n;
        }

        if (beta != 0)
        {
            manager.copyBuffer(tmp_c, d_c, caller / "gemm" / "restore_c");
            Common::checkOpenCLStatus(queue.finish(), caller, "Error restoring matrix c");
        }
    }
    else if (manager.getExecutionPolicy() == KernelExecutionPolicy::ProfiledParams)
    {
        const auto& info = (*manager.getExecutionProfile())[caller / "gemm"];
        if (!info.Data.empty())
        {
            auto pos = info.Data.find("_");
            if (pos != string::npos)
            {
                item_m = stoi(info.Data.substr(0, pos));
                item_n = stoi(info.Data.substr(pos + 1));
            }
        }
    }
    auto kernelGemm = getGemmKernel(manager, item_m, item_n);

    cl::NDRange workSize{ (n + item_n - 1) / item_n, (m + item_m - 1) / item_m, batch };
    auto storedPolicy = manager.getExecutionPolicy();
    if (storedPolicy == KernelExecutionPolicy::SelectBestParams)
    {
        manager.setExecutionPolicy(KernelExecutionPolicy::DefaultParams);
    }

    manager.callKernel(kernelGemm,
                       workSize,
                       caller / "gemm",
                       (cl_int)m_align,
                       (cl_int)n_align,
                       (cl_int)k,
                       (cl_int)n,
                       (cl_int)a_str,
                       (cl_int)b_str,
                       (cl_int)c_str,
                       (cl_int)a_offset,
                       (cl_int)b_offset,
                       (cl_int)n,
                       (cl_int)m,
                       (cl_int)workSize[0],
                       (cl_int)workSize[1],
                       alpha,
                       beta,
                       matrixAPadded(),
                       matrixBPadded(),
                       d_c(),
                       (cl_int)c_offset);

    manager.setExecutionPolicy(storedPolicy);
}

void axpy(OpenCLKernelManager& manager, const Name& caller, size_t n, OPENBLAS_CONST dtype sa, const cl::Buffer d_x, size_t, cl::Buffer d_y, size_t, size_t xOffset, size_t yOffset)
{
    auto kernel = getAxpyKernel(manager);

    cl::NDRange workSize{ (n + 3) / 4, 1, 1 };

    manager.callKernel(kernel, workSize, caller / "axpy", (cl_int)n, sa, (cl_int)xOffset, (cl_int)yOffset, (cl_int)((n + 3) / 4), d_x(), d_y());
}

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
           size_t yOffset)
{
    auto kernel = getAxpbyKernel(manager);

    cl::NDRange workSize{ (n + 3) / 4, 1, 1 };

    manager.callKernel(kernel, workSize, caller / "axpby", (cl_int)n, sa, sb, (cl_int)xOffset, (cl_int)yOffset, (cl_int)((n + 3) / 4), d_x(), d_y());
}

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
            bool reversedOrder,
            bool zeroOutput,
            size_t matrix_offset,
            size_t image_offset)
{
    auto queue = manager.getCommandQueue();
    if (zeroOutput)
    {
        float fill = 0.f;
        Common::checkOpenCLStatus(queue.enqueueFillBuffer(image, fill, 0, imageChannels * imageHeight * imageWidth * sizeof(dtype)), caller, "Error zeroing output image");
    }

    auto kernel = getCol2ImKernel(manager, reversedOrder);

    cl::NDRange workSize{ imageChannels, imageHeight, imageWidth };

    manager.callKernel(kernel,
                       workSize,
                       caller / "col2im",
                       (cl_int)imageWidth,
                       (cl_int)imageHeight,
                       (cl_int)imageChannels,
                       (cl_int)filterWidth,
                       (cl_int)filterHeight,
                       (cl_int)paddingWidth,
                       (cl_int)paddingHeight,
                       (cl_int)strideWidth,
                       (cl_int)strideHeight,
                       matrix(),
                       (cl_int)matrix_offset,
                       image(),
                       (cl_int)image_offset);
}

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
            bool reversedOrder,
            size_t image_offset,
            size_t matrix_offset)
{
    auto queue = manager.getCommandQueue();

    auto kernel = getIm2ColKernel(manager, reversedOrder);

    size_t widthCol = (imageWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;
    size_t heightCol = (imageHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;
    size_t channelsCol = imageChannels * filterHeight * filterWidth;

    cl::NDRange workSize{ channelsCol, heightCol, widthCol };

    manager.callKernel(kernel,
                       workSize,
                       caller / "im2col",
                       (cl_int)imageWidth,
                       (cl_int)imageHeight,
                       (cl_int)imageChannels,
                       (cl_int)filterWidth,
                       (cl_int)filterHeight,
                       (cl_int)paddingWidth,
                       (cl_int)paddingHeight,
                       (cl_int)strideWidth,
                       (cl_int)strideHeight,
                       image(),
                       (cl_int)image_offset,
                       matrix(),
                       (cl_int)matrix_offset);
}

} // namespace gpu

} // namespace raul
