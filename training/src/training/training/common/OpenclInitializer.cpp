// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "OpenclInitializer.h"

#include <training/common/Common.h>
#include <training/opencl/OpenclLoader.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace std;

namespace
{

/*map<string, vector<string>> CreatePerFileOptions()
{
    map<string, vector<string>> m;
    m["padding_nchw"] = {"-DUSE_CONSTANT", "-DUSE_EDGE", "-DUSE_REFLECT", "-DUSE_SYMMETRIC"};
    m["gemm_tn"] = {};
    for (size_t lm = 4; lm <= 8; lm += 4)
    {
        for (size_t ln = 1; ln <= 8; ++ln)
        {
            size_t un = ln - 1;
            string opt = "-DLM=" + to_string(lm) + " -DLN=" + to_string(ln) + " -DUN=" + to_string(un) + " -DUSE_NCWHC4";
            m["gemm_tn"].emplace_back(opt);
            m["gemm_tn"].emplace_back(opt + " -DUSE_RELU");
            m["gemm_tn"].emplace_back(opt + " -DUSE_GELU");
            m["gemm_tn"].emplace_back(opt + " -DUSE_ELTWISE_NCHW");
            m["gemm_tn"].emplace_back(opt + " -DUSE_ELTWISE_NCWHC4");
        }
    }
    for (size_t lm = 1; lm <= 8; ++lm)
    {
        for (size_t ln = 1; ln <= 8; ++ln)
        {
            if (lm * ln != 1)
            {
                string opt = "-DLM=" + to_string(lm) + " -DLN=" + to_string(ln) + " -DNO_BIAS";
                m["gemm_tn"].emplace_back(opt);

                if (lm * ln != 2)
                {
                    opt = "-DLM=" + to_string(lm) + " -DLN=" + to_string(ln);
                    m["gemm_tn"].emplace_back(opt);
                    m["gemm_tn"].emplace_back(opt + " -DUSE_RELU");
                    m["gemm_tn"].emplace_back(opt + " -DUSE_GELU");
                }
            }
        }
    }
    return m;
}
*/
}

namespace raul
{
OpenclInitializer::OpenclInitializer()
    : mHasOpenCL(false)
    , mIsIntel(false)
{
    try
    {
        auto status = OpenCLHelper::Loader::Init();
        mHasOpenCL = status == CL_SUCCESS;

        if (mHasOpenCL)
        {
            // try to get GPU platform
            setGpuPlatformAndDevice();
            std::tie(mPlatform, mDevice, mContext) = getGpuPlatformDeviceAndContext();
        }
        else
        {
            std::cout << "Error loading OpenCL (" << status << ")" << std::endl;
        }
    }
    catch (std::exception& e)
    {
        mHasOpenCL = false;
        std::cout << "Error loading OpenCL: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Error loading OpenCL" << std::endl;
        mHasOpenCL = false;
    }
}

OpenclInitializer::~OpenclInitializer()
{
    if (!mIsIntel)
    {
        // unexpected crash on Intel HD Graphics
        mContext = nullptr;
        mDevice = nullptr;
    }
    mPlatform = nullptr;

    OpenCLHelper::Loader::Exit();
}

bool OpenclInitializer::hasOpenCL() const
{
    return mHasOpenCL;
}

void OpenclInitializer::setGpuPlatformAndDevice(std::optional<size_t> platform_id, std::optional<size_t> device_id)
{
    if (!mHasOpenCL)
    {
        THROW_NONAME("OpenclInitializer", "no OpenCL library loaded");
    }

    if (device_id && !platform_id)
    {
        THROW_NONAME("OpenclInitializer", "can't set device without setting platform");
    }

    if (!platform_id && mPlatform())
    {
        // nothing to change - we already have some platform
        return;
    }

    auto platforms = std::vector<cl::Platform>();
    cl::Platform::get(&platforms);
    if (platform_id)
    {
        if (platforms.empty())
        {
            THROW_NONAME("OpenclInitializer", "no platforms available");
        }
        if (*platform_id >= platforms.size())
        {
            THROW_NONAME("OpenclInitializer", "bad platform id");
        }
        auto platform = platforms[*platform_id];
        auto devices = std::vector<cl::Device>();

        if (device_id)
        {
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            if (*device_id >= devices.size())
            {
                THROW_NONAME("OpenclInitializer", "bad device id");
            }
            auto device = devices[*device_id];
            auto context = cl::Context(device);
            std::tie(mPlatform, mDevice, mContext) = std::make_tuple(platform, device, context);
        }
        else
        {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            if (!devices.empty())
            {
                auto context = cl::Context(devices.front());
                std::tie(mPlatform, mDevice, mContext) = std::make_tuple(platform, devices.front(), context);
            }
        }
    }
    else
    {
        for (const auto& platform : platforms)
        {
            auto devices = std::vector<cl::Device>();
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            if (!devices.empty())
            {
                auto context = cl::Context(devices.front());
                std::tie(mPlatform, mDevice, mContext) = std::make_tuple(platform, devices.front(), context);
                break;
            }
        }
    }
    if (mContext())
    {
        auto deviceName = mDevice.getInfo<CL_DEVICE_NAME>();
        mIsIntel = (deviceName.find("Intel") != string::npos); // TODO(ad): this is temporary solution to avoid crash in destructor
    }
}

std::tuple<cl::Platform, cl::Device, cl::Context> OpenclInitializer::getGpuPlatformDeviceAndContext() const
{
    return { mPlatform, mDevice, mContext };
}
} // raul namespace
