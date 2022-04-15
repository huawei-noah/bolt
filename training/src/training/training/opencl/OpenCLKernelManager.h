// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef OPENCL_KERNEL_MANAGER_H
#define OPENCL_KERNEL_MANAGER_H

#include <training/common/Common.h>
#include <training/common/ExecutionProfile.h>
#include <training/common/OpenCLInclude.h>

namespace
{

using namespace std;

template<typename T>
void copyArg(cl::CommandQueue, T from, T to)
{
    to = from;
}

template<>
[[maybe_unused]] void copyArg<cl_mem>(cl::CommandQueue queue, cl_mem from, cl_mem to)
{
    using namespace std;
    size_t size;
    cl_int status = clGetMemObjectInfo(from, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
    if (status != CL_SUCCESS)
    {
        throw runtime_error("copyArg: clGetMemObjectInfo failed (" + to_string(status) + ")");
    }
    auto context = queue.getInfo<CL_QUEUE_CONTEXT>();
    to = clCreateBuffer(context(), CL_MEM_READ_WRITE, size, NULL, &status);
    if (status != CL_SUCCESS)
    {
        throw runtime_error("copyArg: clCreateBuffer failed (" + to_string(status) + ")");
    }
    status = clEnqueueCopyBuffer(queue(), from, to, 0, 0, size, 0, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        throw runtime_error("copyArg: enqueueCopyBuffer failed (" + to_string(status) + ")");
    }
}

template<typename T>
void releaseArg(T)
{
}

template<>
[[maybe_unused]] void releaseArg<cl_mem>(cl_mem arg)
{
    if (arg != NULL)
    {
        cl_int status = clReleaseMemObject(arg);
        if (status != CL_SUCCESS)
        {
            throw runtime_error("releaseArg: clReleaseMemObject failed (" + to_string(status) + ")");
        }
    }
}

template<typename TupleType, size_t N>
struct KernelArgsWrapper
{
    static void copyArgWrapper(cl::CommandQueue queue, TupleType from, TupleType to)
    {
        copyArg(queue, get<N - 1>(from), get<N - 1>(to));
        KernelArgsWrapper<TupleType, N - 1>::copyArgWrapper(queue, from, to);
    }

    static cl_int setArgWrapper(cl::Kernel kernel, TupleType tuple)
    {
        auto val = get<N - 1>(tuple);
        cl_int status = kernel.setArg(N - 1, val);
        if (status != CL_SUCCESS)
        {
            return status;
        }
        return KernelArgsWrapper<TupleType, N - 1>::setArgWrapper(kernel, tuple);
    }

    static void releaseArgWrapper(TupleType tuple)
    {
        releaseArg(get<N - 1>(tuple));
        KernelArgsWrapper<TupleType, N - 1>::releaseArgWrapper(tuple);
    }
};

template<typename TupleType>
struct KernelArgsWrapper<TupleType, 0>
{
    static cl_int setArgWrapper(cl::Kernel /*kernel*/, TupleType /*tuple*/) { return CL_SUCCESS; }

    static void copyArgWrapper(cl::CommandQueue queue, TupleType, TupleType)
    {
        auto status = queue.finish();
        if (status != CL_SUCCESS)
        {
            THROW_NONAME("CopyKernelArgsWrapper<Args, 0>", "opencl queue failed (" + std::to_string(status) + ")");
        }
    }

    static void releaseArgWrapper(TupleType) {}
};

template<typename TupleType>
cl_int setKernelArgs(cl::Kernel kernel, TupleType args)
{
    return KernelArgsWrapper<TupleType, tuple_size<TupleType>::value>::setArgWrapper(kernel, args);
}

template<typename... Args>
cl_int setKernelArgs(cl::Kernel kernel, Args... args)
{
    return setKernelArgs(kernel, make_tuple(args...));
}

template<typename TupleType>
class TupleGuard
{
  public:
    TupleGuard(TupleType t)
        : mTuple(t)
    {
    }

    ~TupleGuard() { KernelArgsWrapper<TupleType, tuple_size<TupleType>::value>::releaseArgWrapper(mTuple); }

  private:
    TupleType mTuple;
};

}

namespace raul
{

enum class KernelExecutionPolicy
{
    DefaultParams,
    ProfiledParams,
    SelectBestParams,
    SkipKernels,
    SkipAll
};

using namespace std;

class OpenCLKernelManager
{
  public:
    OpenCLKernelManager();
    explicit OpenCLKernelManager(cl::CommandQueue& queue);

    void setCommandQueue(cl::CommandQueue& queue);
    const cl::CommandQueue& getCommandQueue() const;
    cl::CommandQueue& getCommandQueue();

    void setExecutionProfile(ExecutionProfile* profile);
    shared_ptr<ExecutionProfile> getExecutionProfile() const;

    void setExecutionPolicy(KernelExecutionPolicy policy);
    KernelExecutionPolicy getExecutionPolicy() const;

    void registerProgram(const string& programName, const string& source, const string& params = "");

    cl::Kernel getKernel(const string& programName, const string& kernelName, const string& caller);
    cl::Kernel getKernel(const string& programAndKernelName, const string& caller);
    bool hasKernel(const string& programName, const string& kernelName = "");

    void setKernelFilter(vector<string> filters);

    template<typename... Args>
    void callKernel(cl::Kernel kernel, cl::NDRange workSize, const string& caller, Args... args)
    {
        if (mPolicy == KernelExecutionPolicy::SkipKernels || mPolicy == KernelExecutionPolicy::SkipAll)
        {
            return;
        }

        auto kernelName = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
        kernelName.erase(std::remove(kernelName.begin(), kernelName.end(), '\0'), kernelName.end());

        if (isFiltered(kernelName))
        {
            return;
        }

        decltype(make_tuple(args...)) tempArgs;
        TupleGuard guard(tempArgs);
        auto oldArgs = make_tuple(args...);
        if (mPolicy == KernelExecutionPolicy::SelectBestParams)
        {
            Common::checkOpenCLStatus(setKernelArgs(kernel, tempArgs), caller, "Error setting arguments for kernel \"" + kernelName + "\"");
        }
        else
        {
            Common::checkOpenCLStatus(setKernelArgs(kernel, oldArgs), caller, "Error setting arguments for kernel \"" + kernelName + "\"");
        }
        callKernelImpl(kernel, kernelName, workSize, caller);
        if (mPolicy == KernelExecutionPolicy::SelectBestParams)
        {
            // TODO(ad): use a guard
            mPolicy = KernelExecutionPolicy::ProfiledParams;
            Common::checkOpenCLStatus(setKernelArgs(kernel, args...), caller, "Error setting arguments for kernel \"" + kernelName + "\"");
            callKernelImpl(kernel, kernelName, workSize, caller);
            mPolicy = KernelExecutionPolicy::SelectBestParams;
        }
    }

    void fillBuffer(cl::Buffer buffer, dtype val, const string& caller);
    void copyBuffer(cl::Buffer from, cl::Buffer to, const string& caller);
    void writeBuffer(cl::Buffer to, size_t size, const dtype* source, const string& caller);
    void readBuffer(cl::Buffer from, dtype* dest, const string& caller);
    void readBuffer(cl::Buffer from, size_t size, dtype* dest, const string& caller);

    cl::Buffer createBuffer(size_t bufSize, const string& caller);
    cl::Buffer createSubBuffer(cl::Buffer buffer, size_t offset, size_t size, const string& caller);

  private:
    void callKernelImpl(cl::Kernel kernel, const string& kernelName, cl::NDRange workSize, const string& caller);
    cl::Program buildProgramFromSource(const string& programName, const string& source, const string& params);
    void createHeader();
    bool isFiltered(Name kernelName);

  private:
    cl::CommandQueue mCommandQueue;
    cl::Context mContext;
    cl::Device mDevice;
    map<string, cl::Kernel> mOpenClKernelMap;
    set<string> mRegisteredPrograms;
    cl::Program mHeaderProgram;
    mutable shared_ptr<ExecutionProfile> mProfile;
    KernelExecutionPolicy mPolicy;

    vector<string> mKernelFilter;
};

} // raul namespace

#endif // OPENCL_KERNEL_MANAGER_H
