// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseCompareLayerGPU.h"
#include "../ElementWiseCompareLayer.h"
#include "ElementWiseCompareLayerCPU.h"

#include <unordered_map>

namespace raul
{

void ElementWiseCompareLayerGPU::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && (mLayer.mBroadcastQuery[0] || mLayer.mBroadcastQuery[1]))
    {
        THROW("ElementWiseCompareLayer", mLayer.mName, "input size mismatch");
    }

    auto outputGPU = mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mOutputs[0]];
    Tensor output(outputGPU);

    const Tensor firstInput(mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mInputs[0]]);
    const Tensor secondInput(mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mInputs[1]]);

    if (mLayer.mBroadcastQuery[0])
    {
        auto firstViewer = firstInput.getBroadcastedViewer(output.getShape());
        if (mLayer.mBroadcastQuery[1])
        {
            auto secondViewer = secondInput.getBroadcastedViewer(output.getShape());
            if (mLayer.mEqual)
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = firstViewer[i] == secondViewer[i];
                }
            }
            else if (mLayer.mLess)
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = firstViewer[i] < secondViewer[i];
                }
            }
            else
            {
                auto comparator = comparators<dtype>[mLayer.mCompName];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = comparator(firstViewer[i], secondViewer[i], mLayer.mTolerance);
                }
            }
        }
        else
        {
            if (mLayer.mEqual)
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = firstViewer[i] == secondInput[i];
                }
            }
            else if (mLayer.mLess)
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = firstViewer[i] < secondInput[i];
                }
            }
            else
            {
                auto comparator = comparators<dtype>[mLayer.mCompName];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = comparator(firstViewer[i], secondInput[i], mLayer.mTolerance);
                }
            }
        }
    }
    else
    {
        if (mLayer.mBroadcastQuery[1])
        {
            auto secondViewer = secondInput.getBroadcastedViewer(output.getShape());
            if (mLayer.mEqual)
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = firstInput[i] == secondViewer[i];
                }
            }
            else if (mLayer.mLess)
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = firstInput[i] < secondViewer[i];
                }
            }
            else
            {
                auto comparator = comparators<dtype>[mLayer.mCompName];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = comparator(firstInput[i], secondViewer[i], mLayer.mTolerance);
                }
            }
        }
        else
        {
            if (mLayer.mEqual)
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = firstInput[i] == secondInput[i];
                }
            }
            else if (mLayer.mLess)
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = firstInput[i] < secondInput[i];
                }
            }
            else
            {
                auto comparator = comparators<dtype>[mLayer.mCompName];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < output.size(); i++)
                {
                    output[i] = comparator(firstInput[i], secondInput[i], mLayer.mTolerance);
                }
            }
        }
    }
    outputGPU = output;
}

} // namespace raul