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

// col_data, iw, ih, ic, kw, kh, pw, ph, sw, sh, reversed
struct GpuCol2ImTest : public testing::TestWithParam<tuple<vector<dtype>, array<size_t, 9>, bool>>
{
    const float EPSILON_ACCURACY = 1e-3f;

    const vector<dtype> colData = get<0>(GetParam());
    const array<size_t, 9> params = get<1>(GetParam());
    const bool reversed = get<2>(GetParam());

    size_t ITERS = 10000;

    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    const size_t platform_id = 0;
    const size_t device_id = 0;

    cl::Buffer device_col;
    cl::Buffer device_im;

    vector<float> host_col;
    vector<float> host_im;

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

        cl_int error;

        cout << device.getInfo<CL_DEVICE_NAME>(&error) << endl;
        ASSERT_EQ(CL_SUCCESS, error);

        auto [iw, ih, ic, kw, kh, pw, ph, sw, sh] = params;

        //cout << "n = " << n << ", sa = " << sa << ", xOffset = " << xOffset << ", yOffset = " << yOffset << " (" << ITERS << " iters)" << endl;

        size_t colSize = colData.size();
        size_t imSize = iw * ih * ic;

        device_col = manager.createBuffer(colSize * sizeof(dtype), "");
        device_im = manager.createBuffer(imSize * sizeof(dtype), "");

        host_col = colData;
        host_im.resize(imSize);
    }
};

struct GpuIm2ColTest : public testing::TestWithParam<tuple<vector<dtype>, array<size_t, 9>, bool>>
{
    const float EPSILON_ACCURACY = 1e-3f;

    const vector<dtype> imData = get<0>(GetParam());
    const array<size_t, 9> params = get<1>(GetParam());
    const bool reversed = get<2>(GetParam());

    size_t ITERS = imData.empty() ? 10 : 1000;

    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    const size_t platform_id = 0;
    const size_t device_id = 0;

    cl::Buffer device_col;
    cl::Buffer device_im;

    vector<float> host_col;
    vector<float> host_im;

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

        cout << device.getInfo<CL_DEVICE_NAME>() << endl;

        auto [iw, ih, ic, kw, kh, pw, ph, sw, sh] = params;

        // cout << "n = " << n << ", sa = " << sa << ", xOffset = " << xOffset << ", yOffset = " << yOffset << " (" << ITERS << " iters)" << endl;

        size_t imSize = imData.empty() ? (iw * ih * ic) : imData.size();
        size_t colSize = Common::im2colOutputSize(iw, ih, ic, kw, kh, pw, ph, sw, sh, 1, 1);

        device_col = cl::Buffer(context, CL_MEM_READ_WRITE, colSize * sizeof(dtype), NULL, &error);
        ASSERT_EQ(CL_SUCCESS, error);
        device_im = cl::Buffer(context, CL_MEM_READ_WRITE, imSize * sizeof(dtype), NULL, &error);
        ASSERT_EQ(CL_SUCCESS, error);

        host_im = imData;
        host_col.resize(colSize);
    }
};

TEST_P(GpuCol2ImTest, Col2ImGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    const auto random_range = pair<dtype, dtype>(-4.0f, 4.0f);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(random_range.first, nextafter(random_range.second, numeric_limits<dtype>::max()));

    // Populate host matrices with some example data
    vector<float> cl_host_im(host_im.size());

    for (auto& y : host_im)
    {
        y = static_cast<float>(dis(gen));
    }

    queue.enqueueWriteBuffer(device_col, CL_TRUE, 0, host_col.size() * sizeof(dtype), host_col.data());
    queue.enqueueWriteBuffer(device_im, CL_TRUE, 0, host_im.size() * sizeof(dtype), host_im.data());

    auto [iw, ih, ic, kw, kh, sw, sh, pw, ph] = params;

    raul::gpu::col2im(manager, TEST_NAME, device_col, iw, ih, ic, kw, kh, sw, sh, pw, ph, device_im, reversed, false);

    queue.finish();
    auto timeStart = chrono::steady_clock::now();

    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        raul::gpu::col2im(manager, TEST_NAME, device_col, iw, ih, ic, kw, kh, sw, sh, pw, ph, device_im, reversed, false);
    }
    // Measure time
    queue.finish();
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

    queue.enqueueReadBuffer(device_im, CL_TRUE, 0, host_im.size() * sizeof(dtype), cl_host_im.data());

    if (ITERS > 0)
    {
        cout << "OpenCL:" << endl;
        cout << "  Total cl cpu time: " << elapsed / static_cast<float>(ITERS) << " ms" << endl;
    }
    Common::col2im(host_col.data(), iw, ih, ic, kw, kh, sw, sh, pw, ph, host_im.data(), reversed, false);
    timeStart = chrono::steady_clock::now();
    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        Common::col2im(host_col.data(), iw, ih, ic, kw, kh, sw, sh, pw, ph, host_im.data(), reversed, false);
    }
    auto cpu_elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    if (ITERS > 0)
    {
        cout << "CPU:" << endl;
        cout << "  Total cpu time: " << cpu_elapsed / static_cast<float>(ITERS) << " ms" << endl;
        cout << "OpenCL is " << (cpu_elapsed > elapsed ? "faster: " : "slower: ") << elapsed / static_cast<float>(ITERS) << " <-> " << cpu_elapsed / static_cast<float>(ITERS) << endl;
    }
    size_t differentValues = 0;

    for (size_t i = 0; i < host_im.size(); ++i)
    {
        if (fabs(host_im[i] - cl_host_im[i]) >= EPSILON_ACCURACY)
        {
            if (differentValues == 0)
            {
                cout << "First difference at index " << i << ": " << host_im[i] << " <-> " << cl_host_im[i] << endl;
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
        cout << differentValues << "/" << host_im.size() << " values (" << 100.f * static_cast<float>(differentValues) / static_cast<float>(host_im.size()) << "%) has relative difference more then "
             << EPSILON_ACCURACY << endl;
    }
    ASSERT_EQ(differentValues, 0u);
}

TEST_P(GpuIm2ColTest, Im2ColGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    const auto random_range = pair<dtype, dtype>(-4.0f, 4.0f);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(random_range.first, nextafter(random_range.second, numeric_limits<dtype>::max()));

    // Populate host matrices with some example data
    auto [iw, ih, ic, kw, kh, sw, sh, pw, ph] = params;
    if (host_im.empty())
    {
        host_im.resize(iw * ih * ic);
        for (auto& x : host_im)
        {
            x = static_cast<float>(dis(gen));
        }
    }

    vector<float> cl_host_im(host_im.size());

    queue.enqueueWriteBuffer(device_col, CL_TRUE, 0, host_col.size() * sizeof(dtype), host_col.data());
    queue.enqueueWriteBuffer(device_im, CL_TRUE, 0, host_im.size() * sizeof(dtype), host_im.data());

    raul::gpu::im2col(manager, TEST_NAME, device_im, iw, ih, ic, kw, kh, sw, sh, pw, ph, device_col, reversed);

    queue.finish();
    auto timeStart = chrono::steady_clock::now();

    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        raul::gpu::im2col(manager, TEST_NAME, device_im, iw, ih, ic, kw, kh, sw, sh, pw, ph, device_col, reversed);
    }
    // Measure time
    queue.finish();
    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

    queue.enqueueReadBuffer(device_im, CL_TRUE, 0, host_im.size() * sizeof(dtype), cl_host_im.data());

    if (ITERS > 0)
    {
        cout << "OpenCL:" << endl;
        cout << "  Total cl cpu time: " << elapsed / static_cast<float>(ITERS) << " ms" << endl;
    }
    Common::im2col(host_im.data(), iw, ih, ic, kw, kh, sw, sh, pw, ph, host_col.data(), reversed);
    timeStart = chrono::steady_clock::now();
    for (size_t iter = 0; iter < ITERS; ++iter)
    {
        Common::im2col(host_im.data(), iw, ih, ic, kw, kh, sw, sh, pw, ph, host_col.data(), reversed);
    }
    auto cpu_elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());
    if (ITERS > 0)
    {
        cout << "CPU:" << endl;
        cout << "  Total cpu time: " << cpu_elapsed / static_cast<float>(ITERS) << " ms" << endl;
        cout << "OpenCL is " << (cpu_elapsed > elapsed ? "faster: " : "slower: ") << elapsed / static_cast<float>(ITERS) << " <-> " << cpu_elapsed / static_cast<float>(ITERS) << endl;
    }
    size_t differentValues = 0;

    for (size_t i = 0; i < host_im.size(); ++i)
    {
        if (fabs(host_im[i] - cl_host_im[i]) >= EPSILON_ACCURACY)
        {
            if (differentValues == 0)
            {
                cout << "First difference at index " << i << ": " << host_im[i] << " <-> " << cl_host_im[i] << endl;
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
        cout << differentValues << "/" << host_im.size() << " values (" << 100.f * static_cast<float>(differentValues) / static_cast<float>(host_im.size()) << "%) has relative difference more then "
             << EPSILON_ACCURACY << endl;
    }
    ASSERT_EQ(differentValues, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    TestGpu,
    GpuCol2ImTest,
    testing::Values(make_tuple(vector<dtype>{ 1.0f, 4.0f, 2.0f, 5.0f, 4.0f, 7.0f, 5.0f, 8.0f, 2.0f, 5.0f, 3.0f, 6.0f, 5.0f, 8.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 2, 2, 1, 1, 0, 0 }, false),
                    make_tuple(vector<dtype>{ 1.0f, 2.0f, 4.0f, 5.0f, 2.0f, 3.0f, 5.0f, 6.0f }, array<size_t, 9>{ 3, 3, 1, 2, 2, 2, 1, 0, 0 }, false),
                    make_tuple(vector<dtype>{ 1.0f, 4.0f, 4.0f, 7.0f, 2.0f, 5.0f, 5.0f, 8.0f }, array<size_t, 9>{ 3, 3, 1, 2, 2, 1, 2, 0, 0 }, false),
                    make_tuple(vector<dtype>{ 1.0f, 2.0f, 4.0f, 5.0f, 7.0f, 8.0f, 2.0f, 3.0f, 5.0f, 6.0f, 8.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 3, 2, 1, 1, 0, 0 }, false),
                    make_tuple(vector<dtype>{ 1.0f, 4.0f, 2.0f, 5.0f, 4.0f, 7.0f, 5.0f, 8.0f, 2.0f, 5.0f, 3.0f, 6.0f, 5.0f, 8.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 2, 3, 1, 1, 0, 0 }, false),
                    make_tuple(vector<dtype>{ 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f }, array<size_t, 9>{ 3, 3, 1, 3, 2, 1, 2, 0, 0 }, false),
                    make_tuple(vector<dtype>{ 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f }, array<size_t, 9>{ 3, 3, 1, 2, 3, 2, 1, 0, 0 }, false),
                    make_tuple(vector<dtype>{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 4.0f,
                                              7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f, 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f,
                                              0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                               array<size_t, 9>{ 3, 3, 1, 2, 2, 1, 1, 1, 1 },
                               false),
                    make_tuple(vector<dtype>{ 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f,
                                              0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f },
                               array<size_t, 9>{ 3, 3, 1, 2, 2, 1, 1, 1, 0 },
                               false),
                    make_tuple(vector<dtype>{ 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f,
                                              0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f },
                               array<size_t, 9>{ 3, 3, 1, 2, 2, 1, 1, 1, 0 },
                               true),
                    make_tuple(vector<dtype>{ 0.0f, 0.0f, 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, 0.0f, 0.0f, 4.0f, 7.0f, 5.0f, 8.0f, 6.0f, 9.0f,
                                              1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, 0.0f, 0.0f, 4.0f, 7.0f, 5.0f, 8.0f, 6.0f, 9.0f, 0.0f, 0.0f },
                               array<size_t, 9>{ 3, 3, 1, 2, 2, 1, 1, 0, 1 },
                               false),
                    make_tuple(vector<dtype>{ 1.0f,  2.0f,  3.0f,  5.0f,  6.0f,  7.0f,  9.0f,  10.0f, 11.0f, 2.0f,  3.0f,  4.0f,  6.0f,  7.0f,  8.0f,  10.0f, 11.0f, 12.0f,
                                              5.0f,  6.0f,  7.0f,  9.0f,  10.0f, 11.0f, 13.0f, 14.0f, 15.0f, 6.0f,  7.0f,  8.0f,  10.0f, 11.0f, 12.0f, 14.0f, 15.0f, 16.0f,

                                              17.0f, 18.0f, 19.0f, 21.0f, 22.0f, 23.0f, 25.0f, 26.0f, 27.0f, 18.0f, 19.0f, 20.0f, 22.0f, 23.0f, 24.0f, 26.0f, 27.0f, 28.0f,
                                              21.0f, 22.0f, 23.0f, 25.0f, 26.0f, 27.0f, 29.0f, 30.0f, 31.0f, 22.0f, 23.0f, 24.0f, 26.0f, 27.0f, 28.0f, 30.0f, 31.0f, 32.0f,

                                              33.0f, 34.0f, 35.0f, 37.0f, 38.0f, 39.0f, 41.0f, 42.0f, 43.0f, 34.0f, 35.0f, 36.0f, 38.0f, 39.0f, 40.0f, 42.0f, 43.0f, 44.0f,
                                              37.0f, 38.0f, 39.0f, 41.0f, 42.0f, 43.0f, 45.0f, 46.0f, 47.0f, 38.0f, 39.0f, 40.0f, 42.0f, 43.0f, 44.0f, 46.0f, 47.0f, 48.0f },
                               array<size_t, 9>{ 4, 4, 3, 2, 2, 1, 1, 0, 0 },
                               false)));

INSTANTIATE_TEST_SUITE_P(TestGpu,
                         GpuIm2ColTest,
                         testing::Values(make_tuple(vector<dtype>{ 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 2, 2, 1, 1, 0, 0 }, false),
                                         make_tuple(vector<dtype>{ 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 2, 2, 2, 1, 0, 0 }, false),
                                         make_tuple(vector<dtype>{ 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 2, 2, 1, 2, 0, 0 }, false),
                                         make_tuple(vector<dtype>{ 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 3, 2, 1, 1, 0, 0 }, false),
                                         make_tuple(vector<dtype>{ 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 2, 3, 1, 1, 0, 0 }, false),
                                         make_tuple(vector<dtype>{ 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 3, 2, 1, 2, 0, 0 }, false),
                                         make_tuple(vector<dtype>{ 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 2, 3, 2, 1, 0, 0 }, false),
                                         make_tuple(vector<dtype>{ 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 2, 2, 1, 1, 1, 1 }, false),
                                         make_tuple(vector<dtype>{ 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 2, 2, 1, 1, 1, 0 }, false),
                                         make_tuple(vector<dtype>{ 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f }, array<size_t, 9>{ 3, 3, 1, 2, 2, 1, 1, 0, 1 }, false),
                                         make_tuple(vector<dtype>{ 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                                                                   17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
                                                                   33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f },
                                                    array<size_t, 9>{ 4, 4, 3, 2, 2, 1, 1, 0, 0 },
                                                    false),
                                         make_tuple(vector<dtype>{}, array<size_t, 9>{ 500, 500, 5, 2, 2, 1, 1, 0, 1 }, false)));
} // UT namespace
