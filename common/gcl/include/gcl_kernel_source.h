// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GCL_KERNEL_SOURCE
#define GCL_KERNEL_SOURCE

#include "gcl_kernel_type.h"
#include "error.h"

#include <string>
#include <unordered_map>
typedef GCLKernelSource KernelSource;
typedef GCLKernelOption KernelOption;

class gcl_kernel_source {
public:
    gcl_kernel_source()
    {
        UNI_DEBUG_LOG("gcl_kernel_source %p constructor\n", (char *)this);
    }
    ~gcl_kernel_source()
    {
        UNI_DEBUG_LOG("gcl_kernel_source %p constructor\n", (char *)this);
    }

    std::unordered_map<std::string, KernelSource> &kernelSourceMap()
    {
        return kernelSourceMap_;
    }
    std::unordered_map<std::string, KernelOption> &kernelOptionMap()
    {
        return kernelOptionMap_;
    }

    void put_source(std::string kernelname, KernelSource kernelSource)
    {
        auto it = kernelSourceMap_.find(kernelname);
        if (it == kernelSourceMap_.end()) {
            kernelSourceMap_.insert({kernelname, kernelSource});
        }
    }

    bool get_source(std::string kernelname, KernelSource **kernelSource_ptr)
    {
        auto it = kernelSourceMap_.find(kernelname);
        if (it == kernelSourceMap_.end()) {
            return false;
        }
        *kernelSource_ptr = &it->second;
        return true;
    }

    void put_option(std::string kernelname, KernelOption kernelOption)
    {
        auto it = kernelOptionMap_.find(kernelname);
        if (it == kernelOptionMap_.end()) {
            kernelOptionMap_.insert({kernelname, kernelOption});
        }
    }

    bool get_option(std::string kernelname, KernelOption **kernelOption_ptr)
    {
        auto it = kernelOptionMap_.find(kernelname);
        if (it == kernelOptionMap_.end()) {
            return false;
        }
        *kernelOption_ptr = &it->second;
        return true;
    }

private:
    std::unordered_map<std::string, KernelSource> kernelSourceMap_;
    std::unordered_map<std::string, KernelOption> kernelOptionMap_;
};
#endif
