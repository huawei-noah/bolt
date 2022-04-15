// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "OpenCLKernelManager.h"
#include <chrono>
#include <training/opencl/kernels/kernel_chunks.h>
#include <utility>

using namespace std;

namespace
{

using namespace raul;

bool match(string_view first, string_view second)
{
    // If we reach at the end of both strings, we are done
    if (first.empty() && second.empty())
    {
        return true;
    }

    if (first.empty())
    {
        return false;
    }

    // Make sure that the characters after '*' are present
    // in second string. This function assumes that the first
    // string will not contain two consecutive '*'
    if (second.empty())
    {
        if (first[0] == '*' && first.size() == 1)
        {
            return true;
        }
        return false;
    }

    // If the first string contains '?', or current characters
    // of both strings match
    if (first[0] == '?' || first[0] == second[0])
    {
        return match(first.substr(1), second.substr(1));
    }

    // If there is *, then there are two possibilities
    // a) We consider current character of second string
    // b) We ignore current character of second string.
    if (first[0] == '*')
    {
        return match(first.substr(1), second) || match(first, second.substr(1));
    }
    return false;
}

cl::NDRange globalSizeFromLocalSize(cl::NDRange workSize, cl::NDRange localSize)
{
    if (localSize.dimensions() == 0)
    {
        return workSize;
    }
    size_t gs[3];
    for (size_t i = 0; i < 3; ++i)
    {
        if (localSize[i] == 0)
        {
            throw runtime_error("globalSizeFromLocalSize: localSize MUST either be NullRange or have three non-zero dimensions");
        }
        gs[i] = (workSize[i] + localSize[i] - 1) / localSize[i] * localSize[i];
    }
    return cl::NDRange(gs[0], gs[1], gs[2]);
}

float measureKernelTime(cl::CommandQueue queue, cl::Kernel kernel, const std::string& caller, cl::NDRange offset, cl::NDRange globalSize, cl::NDRange localSize)
{
    auto kernelName = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
    cl_command_queue_properties prop = queue.getInfo<CL_QUEUE_PROPERTIES>();
    bool profiling = prop & CL_QUEUE_PROFILING_ENABLE;
    // finish all unfinished jobs
    Common::checkOpenCLStatus(queue.finish(), caller, "Error running kernels previous to \"" + kernelName + "\"");

    size_t iters = 10;
    float duration = 0;

    for (size_t iter = 0; iter < iters + 1; ++iter)
    {
        cl::Event event;
        auto start = chrono::high_resolution_clock::now();
        Common::checkOpenCLStatus(queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize, nullptr, &event), caller, "Error running kernel \"" + kernelName + "\"");
        Common::checkOpenCLStatus(queue.finish(), caller, "Error running kernel \"" + kernelName + "\"");
        if (iter == 0)
        {
            continue;
        }
        if (!profiling)
        {
            auto finish = chrono::high_resolution_clock::now();
            duration += static_cast<float>(chrono::duration_cast<chrono::nanoseconds>(finish - start).count());
        }
        else
        {
            auto eventStart = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            auto eventFinish = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            duration += static_cast<float>(eventFinish - eventStart);
        }
    }

    duration /= static_cast<float>(iters);
    return duration;
}

}

namespace raul
{
OpenCLKernelManager::OpenCLKernelManager()
    : mProfile(nullptr)
    , mPolicy(KernelExecutionPolicy::DefaultParams)
{
}

OpenCLKernelManager::OpenCLKernelManager(cl::CommandQueue& queue)
    : mProfile(nullptr)
    , mPolicy(KernelExecutionPolicy::DefaultParams)
{
    setCommandQueue(queue);
}

void OpenCLKernelManager::setKernelFilter(vector<string> filters)
{
    mKernelFilter = std::move(filters);
}

bool OpenCLKernelManager::isFiltered(Name kernelName)
{
    ///@todo(ck): use any_of
    for (const auto& name : mKernelFilter)
    {
        if (match(name, kernelName))
        {
            return true;
        }
    }
    return false;
}

void OpenCLKernelManager::setExecutionProfile(ExecutionProfile* profile)
{
    mProfile.reset(profile);
}

std::shared_ptr<ExecutionProfile> OpenCLKernelManager::getExecutionProfile() const
{
    if (!mProfile)
    {
        mProfile = std::make_shared<ExecutionProfile>();
    }
    return mProfile;
}

void OpenCLKernelManager::setExecutionPolicy(KernelExecutionPolicy policy)
{
    mPolicy = policy;
}

KernelExecutionPolicy OpenCLKernelManager::getExecutionPolicy() const
{
    return mPolicy;
}

void OpenCLKernelManager::callKernelImpl(cl::Kernel kernel, const std::string& kernelName, cl::NDRange workSize, const string& caller)
{
    cl::NDRange offset{ 0, 0, 0 };

    if (mPolicy == KernelExecutionPolicy::SelectBestParams)
    {
        if (!mProfile)
        {
            mProfile = std::make_shared<ExecutionProfile>();
        }

        cl::NDRange testLocalSize;
        cl::NDRange bestLocalSize;
        cl::NDRange testGlobalSize = globalSizeFromLocalSize(workSize, testLocalSize);
        size_t maxSize = 384;
        size_t gs_x = 256;
        size_t gs_y = (workSize.dimensions() > 1) ? 256 : 1;
        size_t gs_z = (workSize.dimensions() > 2) ? workSize[2] : 1;
        float bestTime = measureKernelTime(mCommandQueue, kernel, caller, offset, testGlobalSize, testLocalSize);
        for (size_t z = 1; z <= gs_z; z = z << 1)
        {
            if (gs_z % z != 0)
            {
                continue;
            }
            for (size_t y = 1; y <= gs_y; y = y << 1)
            {
                if (gs_y % y != 0)
                {
                    continue;
                }
                for (size_t x = 1; x <= gs_x; x = x << 1)
                {
                    if (gs_x % x != 0)
                    {
                        continue;
                    }
                    auto total = x * y * z;
                    if (total <= maxSize)
                    {
                        testLocalSize = cl::NDRange{ x, y, z };
                        testGlobalSize = globalSizeFromLocalSize(workSize, testLocalSize);

                        float duration = measureKernelTime(mCommandQueue, kernel, caller, offset, testGlobalSize, testLocalSize);

                        if (duration < bestTime)
                        {
                            bestTime = duration;
                            bestLocalSize = testLocalSize;
                        }
                    }
                }
            }
        }

        (*mProfile)[caller].LocalSize = bestLocalSize;
        (*mProfile)[caller].BestTimeNS = bestTime;
    }
    else
    {
        cl::NDRange localSize = { 1, 1, 1 };
        if (mPolicy == KernelExecutionPolicy::ProfiledParams && mProfile)
        {
            auto& hint = (*mProfile)[caller];
            localSize = hint.LocalSize;
        }
        auto gs = globalSizeFromLocalSize(workSize, localSize);
        Common::checkOpenCLStatus(mCommandQueue.enqueueNDRangeKernel(kernel, offset, gs, localSize), caller, "Error running kernel \"" + kernelName + "\"");
    }
}

void OpenCLKernelManager::fillBuffer(cl::Buffer buffer, dtype val, const string& caller)
{
    if (mPolicy == KernelExecutionPolicy::SkipAll)
    {
        return;
    }
    cl_int status;
    auto size = buffer.getInfo<CL_MEM_SIZE>(&status);
    Common::checkOpenCLStatus(status, caller, "error quering buffer size");
    Common::checkOpenCLStatus(mCommandQueue.enqueueFillBuffer(buffer, val, 0, size), caller, "enqueueFillBuffer failed");
}

void OpenCLKernelManager::copyBuffer(cl::Buffer from, cl::Buffer to, const string& caller)
{
    if (mPolicy == KernelExecutionPolicy::SkipAll)
    {
        return;
    }
    cl_int status;
    auto sizeFrom = from.getInfo<CL_MEM_SIZE>(&status);
    Common::checkOpenCLStatus(status, caller, "error quering source buffer size");
    auto sizeTo = to.getInfo<CL_MEM_SIZE>(&status);
    Common::checkOpenCLStatus(status, caller, "error quering destination buffer size");
    if (sizeFrom > sizeTo)
    {
        throw runtime_error(caller + ": source buffer sis larger then destination buffer')");
    }
    Common::checkOpenCLStatus(mCommandQueue.enqueueCopyBuffer(from, to, 0, 0, sizeFrom), caller, "enqueueCopyBuffer failed");
}

void OpenCLKernelManager::writeBuffer(cl::Buffer to, size_t size, const dtype* source, const string& caller)
{
    if (mPolicy == KernelExecutionPolicy::SkipAll)
    {
        return;
    }
    if (size == 0)
    {
        return;
    }
    cl_int status;
    auto sizeTo = to.getInfo<CL_MEM_SIZE>(&status);
    Common::checkOpenCLStatus(status, caller, "error quering destination buffer size");
    if (size > sizeTo)
    {
        throw runtime_error(caller + ": source buffer is larger then destination buffer')");
    }
    Common::checkOpenCLStatus(mCommandQueue.enqueueWriteBuffer(to, CL_TRUE, 0, size, source, NULL, NULL), caller, "enqueueWriteBuffer failed");
}

void OpenCLKernelManager::readBuffer(cl::Buffer from, dtype* dest, const string& caller)
{
    if (mPolicy == KernelExecutionPolicy::SkipAll)
    {
        return;
    }
    cl_int status;
    auto sizeFrom = from.getInfo<CL_MEM_SIZE>(&status);
    Common::checkOpenCLStatus(status, caller, "error quering source buffer size");
    Common::checkOpenCLStatus(mCommandQueue.enqueueReadBuffer(from, CL_TRUE, 0, sizeFrom, dest, NULL, NULL), caller, "enqueueReadBuffer failed");
}

cl::Buffer OpenCLKernelManager::createBuffer(size_t bufSize, const string& caller)
{
    if (mPolicy == KernelExecutionPolicy::SkipAll)
    {
        return cl::Buffer();
    }
    cl_int status;
    auto mem = cl::Buffer(mContext, CL_MEM_READ_WRITE, bufSize, NULL, &status);
    if (status != CL_SUCCESS)
    {
        Common::checkOpenCLStatus(status, caller, "error creating buffer of size " + std::to_string(bufSize));
    }
    return mem;
}

cl::Buffer OpenCLKernelManager::createSubBuffer(cl::Buffer buffer, size_t offset, size_t size, const string& caller)
{
    if (mPolicy == KernelExecutionPolicy::SkipAll)
    {
        return buffer;
    }
    cl_int status;
    auto sizeSrc = buffer.getInfo<CL_MEM_SIZE>(&status);
    Common::checkOpenCLStatus(status, caller, "error quering source buffer size");
    if (sizeSrc < offset + size)
    {
        throw runtime_error(caller + ": subbuffer (offset: " + to_string(offset) + ", size: " + to_string(size) + ") requested from buffer of size " + to_string(sizeSrc) + ")");
    }
    cl_buffer_region rgn = { offset, size };
    cl::Buffer subBuffer = buffer.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rgn, &status);
    Common::checkOpenCLStatus(status, caller, "error creating subbuffer");
    return subBuffer;
}

void OpenCLKernelManager::readBuffer(cl::Buffer from, size_t size, dtype* dest, const string& caller)
{
    if (mPolicy == KernelExecutionPolicy::SkipAll)
    {
        return;
    }
    if (size == 0)
    {
        return;
    }
    cl_int status;
    auto sizeFrom = from.getInfo<CL_MEM_SIZE>(&status);
    Common::checkOpenCLStatus(status, caller, "error quering source buffer size");
    if (size > sizeFrom)
    {
        throw runtime_error(caller + ": size bigger then source buffer size)");
    }
    Common::checkOpenCLStatus(mCommandQueue.enqueueReadBuffer(from, CL_TRUE, 0, size, dest, NULL, NULL), caller, "enqueueReadBuffer failed");
}

void OpenCLKernelManager::setCommandQueue(cl::CommandQueue& queue)
{
    mCommandQueue = queue;
    if (!mCommandQueue())
    {
        THROW_NONAME("OpenCLKernelManager", "null command queue");
    }
    cl_int status;
    mContext = mCommandQueue.getInfo<CL_QUEUE_CONTEXT>(&status);
    Common::checkOpenCLStatus(status, "OpenCLKernelManager[setCommandQueue]", "error quering context");
    if (!mContext())
    {
        THROW_NONAME("OpenCLKernelManager", "null context");
    }
    auto devices = mContext.getInfo<CL_CONTEXT_DEVICES>(&status);
    Common::checkOpenCLStatus(status, "OpenCLKernelManager[setCommandQueue]", "error quering devices");
    if (devices.size() > 1)
    {
        THROW_NONAME("OpenCLKernelManager", "multiple execution devices currently not supported");
    }
    if (devices.empty())
    {
        THROW_NONAME("OpenCLKernelManager", "no gpu devices");
    }
    mDevice = devices.front();
    if (!mDevice())
    {
        THROW_NONAME("OpenCLKernelManager", "null device");
    }

    createHeader();
}

const cl::CommandQueue& OpenCLKernelManager::getCommandQueue() const
{
    return mCommandQueue;
}

cl::CommandQueue& OpenCLKernelManager::getCommandQueue()
{
    return mCommandQueue;
}

void OpenCLKernelManager::createHeader()
{
    string kernel_def;
    for (const auto& chunk : kernel_def_chunks_init)
    {
        kernel_def += chunk;
    }

    mHeaderProgram = cl::Program(mContext, kernel_def);
}

cl::Program OpenCLKernelManager::buildProgramFromSource(const string& programName, const string& source, const string& params)
{
    vector<string> buildOptions{ params };
    auto clVersion = mDevice.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
    if (Common::startsWith(clVersion, "OpenCL C 2."))
    {
        buildOptions.emplace_back("-cl-std=CL2.0");
    }
    else if (Common::startsWith(clVersion, "OpenCL C 1.2"))
    {
        buildOptions.emplace_back(" -cl-std=CL1.2");
    }
    else if (Common::startsWith(clVersion, "OpenCL C 1.1"))
    {
        buildOptions.emplace_back("-cl-std=CL1.1");
    }
    buildOptions.emplace_back("-D T=float -D T2=float2 -D T3=float3 -D T4=float4 -D T8=float8 -D T16=float16");

    string options;
    for (const auto& opt : buildOptions)
    {
        options += " " + opt;
    }
    cl::Program program(mContext, source);

    auto header = mHeaderProgram();
    auto hname = "kernel_def.h";
    auto err = clCompileProgram(program(), 0, NULL, options.c_str(), 1, &header, &hname, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(mDevice);
        Common::checkOpenCLStatus(err, "OpenCLKernelManager[buildProgramFromSource]", "error compiling \"" + programName + "\" with options " + options + (log.empty() ? "" : "\nBuild log:\n" + log));
    }
    auto linkOptions = " -cl-fast-relaxed-math";
    auto p = program();
    program = clLinkProgram(mContext(), 0, NULL, linkOptions, 1, &p, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(mDevice);
        Common::checkOpenCLStatus(
            err, "OpenCLKernelManager[buildProgramFromSource]", "error linking \"" + programName + "\" with options " + linkOptions + (log.empty() ? "" : "\nBuild log:\n" + log));
    }

    return program;
}

void OpenCLKernelManager::registerProgram(const string& programName, const string& source, const string& params)
{
    if (mRegisteredPrograms.find(programName) != mRegisteredPrograms.end())
    {
        return;
    }
    auto program = buildProgramFromSource(programName, source, params);
    vector<cl::Kernel> kernels;
    program.createKernels(&kernels);
    for (const auto& kernel : kernels)
    {
        auto kernelName = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
        kernelName.erase(find(kernelName.begin(), kernelName.end(), '\0'), kernelName.end());
        auto fullName = programName + "." + kernelName;
        if (mOpenClKernelMap.find(fullName) != mOpenClKernelMap.end())
        {
            throw runtime_error("OpenCLKernelManager[registerProgram]: trying to load kernel \"" + kernelName + "\" for program \"" + programName + "\" twice");
        }
        mOpenClKernelMap[fullName] = kernel;
    }
    mRegisteredPrograms.insert(programName);
}

cl::Kernel OpenCLKernelManager::getKernel(const string& programName, const string& kernelName, const string& caller)
{
    auto it = mOpenClKernelMap.find(programName + "." + kernelName);
    if (it == mOpenClKernelMap.end())
    {
        throw runtime_error(caller + ": kernel \"" + kernelName + "\" in program \"" + programName + "\" not found");
    }

    return it->second;
}

cl::Kernel OpenCLKernelManager::getKernel(const string& programAndKernelName, const string& caller)
{
    return getKernel(programAndKernelName, programAndKernelName, caller);
}

bool OpenCLKernelManager::hasKernel(const string& programName, const string& kernelName)
{
    auto fullName = programName + "." + (kernelName.empty() ? programName : kernelName);
    return mOpenClKernelMap.find(fullName) != mOpenClKernelMap.end();
}

} // raul namespace
