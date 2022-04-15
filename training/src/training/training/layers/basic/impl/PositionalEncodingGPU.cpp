// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "PositionalEncodingGPU.h"
#include "../PositionalEncoding.h"

#include <GemmGPU.h>

namespace raul
{

PositionalEncodingGPU::PositionalEncodingGPU(PositionalEncoding& layer)
    : mLayer(layer)
{
    mLayer.mNetworkParams.mWorkflow.tensorNeeded(mLayer.mName, mLayer.mName / "ranges", WShape{ BS(), 1u, 1u, mLayer.mMaxMelLength }, DEC_FORW_WRIT);
    initKernels();
}

void PositionalEncodingGPU::initKernels()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    string kernelName = "durations_range";
    auto& manager = work.getKernelManager();

    if (!manager.hasKernel(mLayer.mTypeName, kernelName))
    {
        string source =
#include "kernels/positional_encoding.cl"
            ;
        manager.registerProgram(mLayer.mTypeName, source);
    }
}

void PositionalEncodingGPU::initNotBSTensors()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;
    auto& memoryManager = work.getMemoryManager<MemoryManagerGPU>();

    Tensor pe = memoryManager[mLayer.mName / "pe"];

    auto data = pe.reshape(yato::dims(mLayer.mMaxLength, mLayer.mModelSize));
    double w = 1;
    double kw = pow(0.0001, 2. / static_cast<dtype>(mLayer.mModelSize));
    for (size_t i = 0; i < mLayer.mModelSize; i += 2)
    {
        for (size_t t = 0; t < mLayer.mMaxLength; ++t)
        {
            data[t][i] = TODTYPE(sin(w * TODTYPE(t)));
            data[t][i + 1] = TODTYPE(cos(w * TODTYPE(t)));
        }
        w *= kw;
    }
    memoryManager[mLayer.mName / "pe"] = pe;
}

void PositionalEncodingGPU::forwardComputeImpl(NetworkMode)
{
    auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerGPU>();
    auto& kernelManager = mLayer.mNetworkParams.mWorkflow.getKernelManager();
    auto& output = memoryManager(mLayer.mOutputName);

    const auto& inputs = memoryManager(mLayer.mInputName);
    auto batchSize = inputs.getBatchSize();
    const auto& pe = memoryManager(mLayer.mName / "pe");

    if (!mLayer.mDurationEncoding)
    {
        const Tensor inputsCPU = memoryManager[mLayer.mInputName];
        auto inData3D = inputsCPU.reshape(yato::dims(batchSize * inputs.getDepth(), inputs.getHeight(), mLayer.mModelSize));
        Tensor outputCPU = memoryManager[mLayer.mOutputName];
        auto outData3D = outputCPU.reshape(inData3D.dimensions());
        const Tensor peCPU = memoryManager[mLayer.mName / "pe"];
        const auto data = peCPU.reshape(yato::dims(pe.getHeight(), pe.getWidth()));

        size_t N = inData3D.dimensions()[0];
        size_t height = inputs.getHeight();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t e = 0; e < N; ++e)
        {
            for (size_t t = 0; t < height; ++t)
            {
                for (size_t i = 0; i < mLayer.mModelSize; ++i)
                {
                    outData3D[e][t][i] = inData3D[e][t][i] + data[t][i];
                }
            }
        }
        memoryManager[mLayer.mOutputName] = outputCPU;
    }
    else
    {
        auto& ranges = memoryManager(mLayer.mName / "ranges");
        auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]";
        kernelManager.fillBuffer(ranges.getBuffer(), TODTYPE(mLayer.mMaxLength - 1), caller);
        auto durationsRangeKernel = kernelManager.getKernel(mLayer.mTypeName, "durations_range", caller);
        kernelManager.callKernel(
            durationsRangeKernel, cl::NDRange{ batchSize, 1, 1 }, caller, (cl_int)batchSize, (cl_int)inputs.getWidth(), (cl_int)ranges.getWidth(), inputs.getBuffer(), ranges.getBuffer());

        auto rangeLutKernel = kernelManager.getKernel(mLayer.mTypeName, "range_lut", caller);
        kernelManager.callKernel(rangeLutKernel,
                                 cl::NDRange{ batchSize, output.getHeight(), 1 },
                                 caller,
                                 (cl_int)batchSize,
                                 (cl_int)ranges.getWidth(),
                                 (cl_int)output.getWidth(),
                                 (cl_int)pe.getHeight(),
                                 (cl_int)pe.getWidth(),
                                 ranges.getBuffer(),
                                 pe.getBuffer(),
                                 output.getBuffer());
    }
}

void PositionalEncodingGPU::backwardComputeImpl()
{
    // if (mNetworkParams.isGradNeeded(mInputName))
    {
        if (!mLayer.mDurationEncoding)
        {
            auto& memoryManager = mLayer.mNetworkParams.mMemoryManagerGPU;
            Tensor prevLayerDelta = memoryManager[mLayer.mInputName.grad()];
            const Tensor deltas = memoryManager[mLayer.mOutputName.grad()];
            // just copy
            prevLayerDelta += TORANGE(deltas);
            memoryManager[mLayer.mInputName.grad()] = prevLayerDelta;
        }
        //    auto& memoryManager = mLayer.mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerGPU>();
        //    auto& prevLayerDelta = memoryManager(mLayer.mInputName.grad());
        //    auto& delta = memoryManager(mLayer.mOutputName.grad());
        //    // just add
        //    gpu::axpy(mLayer.mNetworkParams.mWorkflow.getKernelManager(), mLayer.mName / "backward", delta.size(), 1.0_dt, delta.getBuffer(), 1, prevLayerDelta.getBuffer(), 1, 0, 0);
        //}
    }
}

} // namespace raul
