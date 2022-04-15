// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef OPENCL_INITIALIZER_H
#define OPENCL_INITIALIZER_H

#include "OpenCLInclude.h"
#include "Singleton.h"

#include <filesystem>
#include <map>
#include <optional>
#include <set>

namespace raul
{
class OpenclInitializer
{
  public:
    OpenclInitializer();

    ~OpenclInitializer();

    bool hasOpenCL() const;

    std::tuple<cl::Platform, cl::Device, cl::Context> getGpuPlatformDeviceAndContext() const;
    void setGpuPlatformAndDevice(std::optional<size_t> platform_id = std::nullopt, std::optional<size_t> device_id = std::nullopt);

  private:
    bool mHasOpenCL;
    cl::Platform mPlatform;
    cl::Device mDevice;
    cl::Context mContext;

    bool mIsIntel;
};

typedef SingletonHolder<OpenclInitializer, CreateStatic, PhoenixSingleton> OpenCLInitializer;

} // raul namespace

#endif // OPENCL_INITIALIZER_H
