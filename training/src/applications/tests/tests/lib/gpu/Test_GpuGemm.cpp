// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <cstdio>
#include <tests/tools/TestTools.h>

#include <training/common/Common.h>
#include <training/common/Tensor.h>
#include <training/opencl/GemmGPU.h>

namespace UT
{
using namespace std;
using namespace raul;

struct GpuGemmTest : public testing::TestWithParam<tuple<size_t, size_t, size_t, float, float, bool, bool, size_t>>
{
    const float EPSILON_ACCURACY = 1e-3f;

    const size_t m = get<0>(GetParam());
    const size_t n = get<1>(GetParam());
    const size_t k = get<2>(GetParam());
    const float alpha = get<3>(GetParam());
    const float beta = get<4>(GetParam());

    const bool transA = get<5>(GetParam());
    const bool transB = get<6>(GetParam());

    const size_t ITERS = get<7>(GetParam());

    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    const size_t platform_id = 0;
    const size_t device_id = 0;

    cl::Buffer device_a;
    cl::Buffer device_b;
    cl::Buffer device_c;
    cl::Buffer device_tmp;

    vector<float> host_a;
    vector<float> host_b;
    vector<float> host_c;

    OpenCLKernelManager manager;

    void SetUp() final
    {
        if (!Common::hasOpenCL())
        {
            return;
        }

        // Initializes the OpenCL platform
        std::tie(platform, device, context) = Common::getGpuPlatformDeviceAndContext();

        // Creates the OpenCL context, queue, and an event
        queue = cl::CommandQueue(context, device);
        manager.setCommandQueue(queue);

        cout << device.getInfo<CL_DEVICE_NAME>() << endl;
        cout << "M = " << m << ", N = " << n << ", K = " << k << ", alpha = " << alpha << ", beta = " << beta << ", " << (transA ? "T" : "N") << (transB ? "T" : "N") << " (" << ITERS << " iters)"
             << endl;

        size_t aSize = k * m * sizeof(float);
        size_t bSize = n * k * sizeof(float);
        size_t cSize = n * m * sizeof(float);

        size_t bufSize = gpu::gemm_temp_buffer_size(transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans, m, n, k);
        if (bufSize > 0)
        {
            device_tmp = manager.createBuffer(bufSize, "");
        }

        device_a = manager.createBuffer(aSize, "");
        device_b = manager.createBuffer(bSize, "");
        device_c = manager.createBuffer(cSize, "");

        host_a.resize(m * k);
        host_b.resize(n * k);
        host_c.resize(m * n);
    }
};

TEST_P(GpuGemmTest, GemmPerfGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    const auto random_range = pair<dtype, dtype>(-4.0f, 4.0f);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(random_range.first, nextafter(random_range.second, numeric_limits<dtype>::max()));

    // Populate host matrices with some example data
    vector<float> cl_host_c(m * n);
    for (size_t i = 0; i < m * k; ++i)
    {
        host_a[i] = static_cast<float>(dis(gen));
    }
    for (size_t i = 0; i < n * k; ++i)
    {
        host_b[i] = static_cast<float>(dis(gen));
    }
    for (size_t i = 0; i < m * n; ++i)
    {
        host_c[i] = static_cast<float>(dis(gen));
        cl_host_c[i] = host_c[i];
    }

    queue.enqueueWriteBuffer(device_a, CL_TRUE, 0, host_a.size() * sizeof(float), host_a.data());
    queue.enqueueWriteBuffer(device_b, CL_TRUE, 0, host_b.size() * sizeof(float), host_b.data());
    queue.enqueueWriteBuffer(device_c, CL_TRUE, 0, host_c.size() * sizeof(float), host_c.data());

    raul::gpu::gemm(manager, TEST_NAME, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans, m, n, k, alpha, device_a, device_b, beta, device_c, device_tmp);

    queue.finish();
    auto timeStart = chrono::steady_clock::now();
    // Call the SGEMM routine.
    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        gpu::gemm(manager, TEST_NAME, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans, m, n, k, alpha, device_a, device_b, beta, device_c, device_tmp);
    }
    // Measure time
    queue.finish();
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

    queue.enqueueReadBuffer(device_c, CL_TRUE, 0, host_c.size() * sizeof(float), cl_host_c.data());

    if (ITERS > 0)
    {
        cout << "Bolt:" << endl;
        cout << "  Total cl cpu time: " << elapsed / static_cast<float>(ITERS) << " ms" << endl;
    }
    Common::gemm(
        nullptr, TEST_NAME, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans, m, n, k, alpha, host_a.data(), host_b.data(), beta, host_c.data()); // to get the same results when
                                                                                                                                                                        // beta != 0
    timeStart = chrono::steady_clock::now();
    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        Common::gemm(nullptr, TEST_NAME, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans, m, n, k, alpha, host_a.data(), host_b.data(), beta, host_c.data());
    }
    auto cpu_elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    if (ITERS > 0)
    {
        cout << "BLAS:" << endl;
        cout << "  Total cpu time: " << cpu_elapsed / static_cast<float>(ITERS) << " ms" << endl;
        cout << "BOLT is " << (cpu_elapsed > elapsed ? "faster: " : "slower: ") << elapsed / static_cast<float>(ITERS) << " <-> " << cpu_elapsed / static_cast<float>(ITERS) << endl;
        cout << "(" << m << ", " << n << ", " << k << ", " << alpha << ".f, " << beta << ".f, " << ITERS << ", " << elapsed / static_cast<float>(ITERS) << ")," << endl;
    }
    size_t differentValues = 0;

    for (size_t i = 0; i < m * n; ++i)
    {
        if (fabs(host_c[i] - cl_host_c[i]) >= EPSILON_ACCURACY)
        {
            if (differentValues == 0)
            {
                cout << "First difference at index " << i << ": " << host_c[i] << " <-> " << cl_host_c[i] << endl;
            }
            ++differentValues;
        }
        else
        {
            differentValues += 0;
        }
    }

    if (differentValues > 0)
    {
        cout << differentValues << "/" << m * n << " values (" << 100.f * static_cast<float>(differentValues) / static_cast<float>(m * n) << "%) has relative difference more then " << EPSILON_ACCURACY
             << endl;
    }
}

INSTANTIATE_TEST_SUITE_P(TestGpu,
                         GpuGemmTest,
                         testing::Values(make_tuple(1, 1, 1, 1.f, 3.f, false, false, 3),
                                         make_tuple(2, 3, 4, 1.f, 1.f, false, false, 3),
                                         make_tuple(11, 1, 11, 1.f, 0.f, false, false, 20),
                                         make_tuple(11, 1, 11, 2.f, 0.f, false, false, 20),
                                         make_tuple(11, 1, 11, 2.f, 1.f, false, false, 3),
                                         make_tuple(14, 1, 14, 1.f, 0.f, false, false, 20),
                                         make_tuple(32, 1, 32, 1.f, 0.f, false, false, 20),
                                         make_tuple(16, 1, 16, 1.f, 0.f, false, false, 20),
                                         make_tuple(3, 4, 2, 1.f, 0.f, false, false, 20),
                                         make_tuple(32, 32, 32, 1.f, 0.f, false, false, 20),
                                         make_tuple(32, 56, 24, 1.f, 0.f, false, false, 20),
                                         make_tuple(32, 56, 24, 1.f, 0.f, true, false, 20),
                                         make_tuple(32, 56, 24, 1.f, 0.f, false, true, 20),
                                         make_tuple(32, 56, 24, 1.f, 0.f, true, true, 20),
                                         make_tuple(32, 56, 24, 3.f, 2.f, false, true, 2),
                                         make_tuple(64, 64, 64, 1.f, 0.f, false, false, 20),
                                         make_tuple(128, 128, 128, 1.f, 0.f, false, false, 20),
                                         make_tuple(256, 128, 19, 1.f, 0.f, false, false, 20),
                                         make_tuple(765, 129, 75, 2.f, -2.f, false, true, 2),
                                         make_tuple(1, 1, 256, 1.f, 0.f, false, false, 20)));

} // UT namespace
