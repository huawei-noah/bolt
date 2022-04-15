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

struct GpuAxpyTest : public testing::TestWithParam<tuple<size_t, dtype, size_t, size_t, size_t>>
{
    const float EPSILON_ACCURACY = 1e-3f;
    const float RELATIVE_ACCURACY = 1e-6f;

    const size_t n = get<0>(GetParam());
    const dtype sa = get<1>(GetParam());
    const size_t xOffset = get<2>(GetParam());
    const size_t yOffset = get<3>(GetParam());
    const size_t ITERS = get<4>(GetParam());

    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    const size_t platform_id = 0;
    const size_t device_id = 0;

    cl::Buffer device_x;
    cl::Buffer device_y;

    vector<float> host_x;
    vector<float> host_y;

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

        ASSERT_TRUE(context() != NULL);
        ASSERT_TRUE(queue() != NULL);
        cl_int error;

        cout << device.getInfo<CL_DEVICE_NAME>(&error) << endl;
        ASSERT_EQ(CL_SUCCESS, error);
        cout << "n = " << n << ", sa = " << sa << ", xOffset = " << xOffset << ", yOffset = " << yOffset << " (" << ITERS << " iters)" << endl;

        size_t xSize = (n + xOffset + n);
        size_t ySize = (n + yOffset + n);

        device_x = manager.createBuffer(xSize * sizeof(dtype), "");
        device_y = manager.createBuffer(ySize * sizeof(dtype), "");

        host_x.resize(xSize);
        host_y.resize(ySize);
    }
};

TEST_P(GpuAxpyTest, AxpyPerfGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    const auto random_range = pair<dtype, dtype>(-4.0f, 4.0f);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(random_range.first, nextafter(random_range.second, numeric_limits<dtype>::max()));

    // Populate host matrices with some example data
    vector<float> cl_host_y(host_y.size());
    for (auto& x : host_x)
    {
        x = static_cast<float>(dis(gen));
    }
    for (auto& y : host_y)
    {
        y = static_cast<float>(dis(gen));
    }

    Common::checkOpenCLStatus(queue.enqueueWriteBuffer(device_x, CL_TRUE, 0, host_x.size() * sizeof(dtype), host_x.data()), TEST_NAME, "enqueueWriteBuffer failed for x");
    Common::checkOpenCLStatus(queue.enqueueWriteBuffer(device_y, CL_TRUE, 0, host_y.size() * sizeof(dtype), host_y.data()), TEST_NAME, "enqueueWriteBuffer failed for y");

    raul::gpu::axpy(manager, TEST_NAME, n, sa, device_x, 1, device_y, 1, xOffset, yOffset);

    Common::checkOpenCLStatus(queue.finish(), TEST_NAME, "finish failed");
    auto timeStart = chrono::steady_clock::now();
    // Call the SGEMM routine.
    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        raul::gpu::axpy(manager, TEST_NAME, n, sa, device_x, 1, device_y, 1, xOffset, yOffset);
    }
    // Measure time
    Common::checkOpenCLStatus(queue.finish(), TEST_NAME, "finish failed");
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

    Common::checkOpenCLStatus(queue.enqueueReadBuffer(device_y, CL_TRUE, 0, host_y.size() * sizeof(dtype), cl_host_y.data()), TEST_NAME, "enqueueReadBuffer failed for y");

    if (ITERS > 0)
    {
        cout << "Bolt:" << endl;
        cout << "  Total cl cpu time: " << elapsed / static_cast<float>(ITERS) << " ms" << endl;
    }
    Common::axpy(nullptr, TEST_NAME, n, sa, host_x.data(), 1, host_y.data(), 1, xOffset, yOffset);
    timeStart = chrono::steady_clock::now();
    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        Common::axpy(nullptr, TEST_NAME, n, sa, host_x.data(), 1, host_y.data(), 1, xOffset, yOffset);
    }
    auto cpu_elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    if (ITERS > 0)
    {
        cout << "BLAS:" << endl;
        cout << "  Total cpu time: " << cpu_elapsed / static_cast<float>(ITERS) << " ms" << endl;
        cout << "BOLT is " << (cpu_elapsed > elapsed ? "faster: " : "slower: ") << elapsed / static_cast<float>(ITERS) << " <-> " << cpu_elapsed / static_cast<float>(ITERS) << endl;
        cout << "(" << n << ", " << sa << ", " << xOffset << ", " << yOffset << ", " << ITERS << ", " << elapsed / static_cast<float>(ITERS) << ")," << endl;
    }
    size_t differentValues = 0;

    for (size_t i = 0; i < host_y.size(); ++i)
    {
        if (fabs(host_y[i] - cl_host_y[i]) >= EPSILON_ACCURACY)
        {
            if (fabs(host_y[i] - cl_host_y[i]) / std::max(fabs(host_y[i]), fabs(cl_host_y[i])) >= RELATIVE_ACCURACY)
            {
                if (differentValues == 0)
                {
                    cout << "First difference at index " << i << ": " << host_y[i] << " <-> " << cl_host_y[i] << endl;
                }
                ++differentValues;
            }
        }
    }

    if (differentValues > 0)
    {
        cout << differentValues << "/" << host_y.size() << " values (" << 100.f * static_cast<float>(differentValues) / static_cast<float>(host_y.size()) << "%) has relative absolute more then "
             << EPSILON_ACCURACY << " or relative difference more then " << RELATIVE_ACCURACY << endl;
    }
    ASSERT_EQ(differentValues, 0u);
}

INSTANTIATE_TEST_SUITE_P(TestGpu,
                         GpuAxpyTest,
                         testing::Values(make_tuple(1, 1.f, 0, 0, 1000),
                                         make_tuple(3, 1.f, 0, 0, 1000),
                                         make_tuple(5, 1.f, 0, 0, 1000),
                                         make_tuple(9, 1.f, 0, 0, 1000),
                                         make_tuple(10, 1.f, 0, 0, 1000),
                                         make_tuple(11, 1.f, 0, 0, 1000),
                                         make_tuple(1000, 2.f, 0, 0, 1000),
                                         make_tuple(10, 2.f, 10, 0, 1000),
                                         make_tuple(10, 2.f, 0, 10, 1000),
                                         make_tuple(10, 3.f, 20, 10, 1000),
                                         make_tuple(10, 3.f, 10, 20, 1000),
                                         make_tuple(1000, 0.f, 1000, 1000, 1000),
                                         make_tuple(100000, 1.f, 1000, 1000, 1000),
                                         make_tuple(100000, 2.f, 1000, 1000, 1000)));

} // UT namespace
